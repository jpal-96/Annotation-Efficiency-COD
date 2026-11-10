# ================================================================
# MAIN.PY
# Annotation-Efficient Camouflaged Object Detection
#
# Pipeline:
# 1. Dataset splitting
# 2. Model initialization
# 3. Weighted BCE + Weighted IoU loss
# 4. Model training and validation
# 5. Test evaluation
# 6. Results generation
#
# Model: PVTv2COD
# Input: 384 x 384
# Optimizer: AdamW
# Learning Rate: 1e-4
# ================================================================

import os
import random
import argparse
import shutil
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models.model import PVTv2COD


# ================================================================
# 1. GLOBAL CONFIGURATION
# ================================================================

CONFIG = {
    "image_size": 384,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "workers": 4,
    "train_ratio": 0.80,
    "val_ratio": 0.20,
    "backbone": "pvt_v2_b2",
    "checkpoint": "checkpoints/best_model.pth",
    "results_dir": "results",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

DEVICE = torch.device(CONFIG["device"])

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

SIDE_WEIGHTS = [1.0, 0.5, 0.25, 0.125]


# ================================================================
# 2. REPRODUCIBILITY
# ================================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================
# 3. DATASET SPLITTING
# ================================================================

def split_dataset(
    image_dir,
    mask_dir,
    output_dir,
    train_ratio=0.80,
    seed=42
):

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            "Original image or mask directory not found."
        )

    images = sorted([
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    pairs = []

    for image_path in images:

        candidates = [
            p for p in mask_dir.iterdir()
            if p.is_file()
            and p.stem == image_path.stem
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if candidates:
            pairs.append((image_path, candidates[0]))
        else:
            print("Missing mask:", image_path.name)

    if len(pairs) < 2:
        raise ValueError("Insufficient image-mask pairs.")

    random.Random(seed).shuffle(pairs)

    split_index = int(len(pairs) * train_ratio)

    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    for split_name, selected_pairs in [
        ("train", train_pairs),
        ("val", val_pairs)
    ]:

        image_output = output_dir / split_name / "images"
        mask_output = output_dir / split_name / "masks"

        image_output.mkdir(parents=True, exist_ok=True)
        mask_output.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in selected_pairs:

            shutil.copy2(
                image_path,
                image_output / image_path.name
            )

            shutil.copy2(
                mask_path,
                mask_output / f"{image_path.stem}.png"
            )

    print("\nDATASET SPLIT")
    print("Total:", len(pairs))
    print("Training:", len(train_pairs))
    print("Validation:", len(val_pairs))

    return len(train_pairs), len(val_pairs)


# ================================================================
# 4. COD DATASET CLASS
# ================================================================

class CODDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        training=False
    ):

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.training = training

        self.pairs = []

        images = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        for image_path in images:

            mask_candidates = list(
                self.mask_dir.glob(image_path.stem + ".*")
            )

            if mask_candidates:
                self.pairs.append(
                    (image_path, mask_candidates[0])
                )

        if not self.pairs:
            raise RuntimeError(
                f"No matching pairs in {image_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (CONFIG["image_size"], CONFIG["image_size"]),
            Image.Resampling.NEAREST
        )

        image = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

        mask = np.asarray(
            mask,
            dtype=np.float32
        ) / 255.0

        mask = (mask >= 0.5).astype(np.float32)

        # Synchronized horizontal flipping
        if self.training and random.random() < 0.5:

            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # Multi-scale augmentation
        if self.training:

            scale = random.choice(
                [0.75, 1.0, 1.25]
            )

            if scale != 1.0:

                new_size = max(
                    32,
                    int(CONFIG["image_size"] * scale)
                )

                image_pil = Image.fromarray(
                    (image * 255).astype(np.uint8)
                )

                mask_pil = Image.fromarray(
                    (mask * 255).astype(np.uint8)
                )

                image = np.asarray(
                    image_pil.resize(
                        (new_size, new_size),
                        Image.Resampling.BILINEAR
                    ).resize(
                        (CONFIG["image_size"], CONFIG["image_size"]),
                        Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32
                ) / 255.0

                mask = np.asarray(
                    mask_pil.resize(
                        (new_size, new_size),
                        Image.Resampling.NEAREST
                    ).resize(
                        (CONFIG["image_size"], CONFIG["image_size"]),
                        Image.Resampling.NEAREST
                    ),
                    dtype=np.float32
                ) / 255.0

                mask = (mask >= 0.5).astype(np.float32)

        # ImageNet normalization
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        image = (image - mean) / std

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        ).float()

        mask = torch.from_numpy(
            mask[None, :, :].copy()
        ).float()

        return image, mask, image_path.stem


# ================================================================
# 5. WEIGHTED BCE + WEIGHTED IOU LOSS
# ================================================================

class WeightedBCELoss(nn.Module):

    def __init__(
        self,
        gamma=5.0,
        kernel_size=31
    ):

        super().__init__()

        self.gamma = gamma
        self.kernel_size = kernel_size

    def forward(self, prediction, target):

        prediction = prediction.clamp(
            1e-7,
            1.0 - 1e-7
        )

        local_average = F.avg_pool2d(
            target,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2
        )

        weight = 1.0 + self.gamma * torch.abs(
            local_average - target
        )

        bce = F.binary_cross_entropy(
            prediction,
            target,
            reduction="none"
        )

        weighted_bce = (
            weight * bce
        ).sum(dim=(2, 3)) / (
            weight.sum(dim=(2, 3)) + 1e-8
        )

        intersection = (
            weight * prediction * target
        ).sum(dim=(2, 3))

        union = (
            weight * (prediction + target)
        ).sum(dim=(2, 3))

        weighted_iou = 1.0 - (
            (intersection + 1.0)
            /
            (union - intersection + 1.0)
        )

        return (
            weighted_bce + weighted_iou
        ).mean()


def total_loss(outputs, target, criterion):

    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if len(outputs) != len(SIDE_WEIGHTS):
        raise ValueError(
            "Expected four side-output predictions."
        )

    loss = 0.0

    for output, weight in zip(
        outputs,
        SIDE_WEIGHTS
    ):

        output = F.interpolate(
            output,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        loss += weight * criterion(
            output,
            target
        )

    return loss


# ================================================================
# 6. MODEL TRAINING
# ================================================================

def train_model(
    train_loader,
    val_loader,
    epochs=50,
    checkpoint_path=None
):

    if checkpoint_path is None:
        checkpoint_path = CONFIG["checkpoint"]

    model = PVTv2COD(
        backbone_name=CONFIG["backbone"],
        pretrained=True
    ).to(DEVICE)

    criterion = WeightedBCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    best_loss = float("inf")
    history = []

    os.makedirs(
        os.path.dirname(checkpoint_path) or ".",
        exist_ok=True
    )

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(images)

                loss = total_loss(
                    outputs,
                    masks,
                    criterion
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():

            for images, masks, _ in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                loss = total_loss(
                    outputs,
                    masks,
                    criterion
                )

                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train: {train_loss:.5f} | "
            f"Validation: {val_loss:.5f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_loss,
                    "backbone": CONFIG["backbone"]
                },
                checkpoint_path
            )

            print("Best checkpoint saved.")

    history_path = (
        Path(checkpoint_path).parent
        / "training_history.csv"
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    print("Training complete.")
    print("Best validation loss:", best_loss)

    return checkpoint_path


# ================================================================
# 7. EVALUATION METRICS
# ================================================================

def calculate_mae(prediction, gt):

    return float(
        np.mean(np.abs(prediction - gt))
    )


def calculate_max_fmeasure(
    prediction,
    gt,
    beta_squared=0.3
):

    scores = []

    for threshold in np.linspace(0, 1, 256):

        pred = (
            prediction >= threshold
        ).astype(np.uint8)

        tp = np.sum(pred * gt)

        fp = np.sum(pred * (1 - gt))

        fn = np.sum((1 - pred) * gt)

        precision = tp / (tp + fp + 1e-8)

        recall = tp / (tp + fn + 1e-8)

        f_measure = (
            (1 + beta_squared) * precision * recall
        ) / (
            beta_squared * precision + recall + 1e-8
        )

        scores.append(f_measure)

    return float(np.max(scores))


def calculate_max_emeasure(prediction, gt):

    scores = []

    gt = gt.astype(np.float64)

    for threshold in np.linspace(0, 1, 256):

        pred = (
            prediction >= threshold
        ).astype(np.float64)

        gt_centered = gt - np.mean(gt)

        pred_centered = pred - np.mean(pred)

        alignment = (
            2 * gt_centered * pred_centered
        ) / (
            gt_centered ** 2
            + pred_centered ** 2
            + 1e-8
        )

        enhanced_alignment = (
            alignment + 1
        ) ** 2 / 4

        scores.append(
            np.mean(enhanced_alignment)
        )

    return float(np.max(scores))


def calculate_smeasure(prediction, gt):

    gt_mean = np.mean(gt)

    if gt_mean == 0:
        return float(1 - np.mean(prediction))

    if gt_mean == 1:
        return float(np.mean(prediction))

    foreground = prediction[gt == 1]

    background = prediction[gt == 0]

    object_fg = (
        2 * np.mean(foreground)
    ) / (
        np.mean(foreground) ** 2
        + 1
        + np.std(foreground)
        + 1e-8
    )

    object_bg = (
        2 * np.mean(1 - background)
    ) / (
        np.mean(1 - background) ** 2
        + 1
        + np.std(1 - background)
        + 1e-8
    )

    object_score = (
        gt_mean * object_fg
        + (1 - gt_mean) * object_bg
    )

    return float(
        np.clip(object_score, 0, 1)
    )


# ================================================================
# 8. TEST EVALUATION
# ================================================================

@torch.no_grad()
def evaluate_model(
    checkpoint_path,
    test_image_dir,
    test_mask_dir,
    dataset_name,
    output_dir="results"
):

    test_dataset = CODDataset(
        test_image_dir,
        test_mask_dir,
        training=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    model = PVTv2COD(
        backbone_name=CONFIG["backbone"],
        pretrained=False
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    rows = []

    prediction_dir = (
        Path(output_dir)
        / "predictions"
        / dataset_name
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for images, masks, names in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        prediction = outputs[0]

        prediction = F.interpolate(
            prediction,
            size=(CONFIG["image_size"], CONFIG["image_size"]),
            mode="bilinear",
            align_corners=False
        )

        prediction = (
            prediction.squeeze().cpu().numpy()
        )

        gt = masks.squeeze().numpy().astype(
            np.uint8
        )

        prediction = np.clip(
            prediction,
            0,
            1
        )

        metrics = {
            "Dataset": dataset_name,
            "Image": names[0],
            "MAE": calculate_mae(prediction, gt),
            "Max_F": calculate_max_fmeasure(prediction, gt),
            "Max_E": calculate_max_emeasure(prediction, gt),
            "S_measure": calculate_smeasure(prediction, gt)
        }

        rows.append(metrics)

        cv2.imwrite(
            str(prediction_dir / f"{names[0]}.png"),
            (prediction * 255).astype(np.uint8)
        )

    result_df = pd.DataFrame(rows)

    os.makedirs(output_dir, exist_ok=True)

    result_path = (
        Path(output_dir)
        / f"{dataset_name}_metrics.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    print("\n", dataset_name, "RESULTS")

    print(
        result_df[
            ["MAE", "Max_F", "Max_E", "S_measure"]
        ].mean()
    )

    return result_df


# ================================================================
# 9. CONSOLIDATED RESULT GENERATION
# ================================================================

def generate_results(output_dir="results"):

    output_dir = Path(output_dir)

    files = list(
        output_dir.glob("*_metrics.csv")
    )

    if not files:
        print("No evaluation files available.")
        return

    all_results = []

    for file in files:

        df = pd.read_csv(file)

        if "Dataset" not in df.columns:
            df["Dataset"] = file.stem

        all_results.append(df)

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    metrics = [
        "MAE",
        "Max_F",
        "Max_E",
        "S_measure"
    ]

    summary = (
        combined.groupby("Dataset")[metrics]
        .agg(["mean", "std"])
    )

    combined.to_csv(
        output_dir / "all_image_results.csv",
        index=False
    )

    summary.to_csv(
        output_dir / "dataset_summary.csv"
    )

    with pd.ExcelWriter(
        output_dir / "COD_experimental_results.xlsx"
    ) as writer:

        combined.to_excel(
            writer,
            sheet_name="Image_Results",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="Dataset_Summary"
        )

    print("\nConsolidated results saved.")
    print(summary)


# ================================================================
# 10. MAIN EXECUTION
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description="COD Annotation-Efficiency Pipeline"
    )

    parser.add_argument(
        "--mode",
        choices=["split", "train", "evaluate", "results"],
        required=True
    )

    parser.add_argument(
        "--image_dir",
        default="datasets/original/images"
    )

    parser.add_argument(
        "--mask_dir",
        default="datasets/original/masks"
    )

    parser.add_argument(
        "--checkpoint",
        default=CONFIG["checkpoint"]
    )

    parser.add_argument(
        "--dataset_name",
        default="CHAMELEON"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"]
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=CONFIG["epochs"]
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # ---------------- SPLIT ----------------

    if args.mode == "split":

        split_dataset(
            args.image_dir,
            args.mask_dir,
            "datasets/split",
            train_ratio=CONFIG["train_ratio"],
            seed=args.seed
        )

    # ---------------- TRAIN ----------------

    elif args.mode == "train":

        train_dataset = CODDataset(
            "datasets/split/train/images",
            "datasets/split/train/masks",
            training=True
        )

        val_dataset = CODDataset(
            "datasets/split/val/images",
            "datasets/split/val/masks",
            training=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["workers"],
            pin_memory=(DEVICE.type == "cuda")
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["workers"],
            pin_memory=(DEVICE.type == "cuda")
        )

        train_model(
            train_loader,
            val_loader,
            epochs=args.epochs,
            checkpoint_path=args.checkpoint
        )

    # ---------------- EVALUATE ----------------

    elif args.mode == "evaluate":

        evaluate_model(
            checkpoint_path=args.checkpoint,
            test_image_dir=args.image_dir,
            test_mask_dir=args.mask_dir,
            dataset_name=args.dataset_name,
            output_dir=CONFIG["results_dir"]
        )

    # ---------------- RESULTS ----------------

    elif args.mode == "results":

        generate_results(
            CONFIG["results_dir"]
        )


if __name__ == "__main__":

    main()
