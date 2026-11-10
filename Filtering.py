# ============================================================
# filtering.py
# Three-Criterion Mask-Consistency Filtering for COD
#
# Criteria:
# 1. Spatial Fidelity       - SAM mask IoU
# 2. Semantic Fidelity      - CLIP cosine similarity
# 3. Camouflage Validity    - Intra-image Frechet distance
# ============================================================

import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

from segment_anything import sam_model_registry, SamPredictor

from transformers import CLIPModel, CLIPProcessor

from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Configuration
# ============================================================

class FilterConfig:

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # SAM configuration
    SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE = "vit_b"

    # CLIP configuration
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

    IMAGE_SIZE = 512

    # Filtering thresholds: set using validation calibration.
    SAM_IOU_THRESHOLD = 0.80
    CLIP_SIMILARITY_THRESHOLD = 0.80
    MIN_CAMOUFLAGE_SCORE = 0.50

    # Output directories
    ACCEPTED_IMAGE_DIR = "filtered_data/images"
    ACCEPTED_MASK_DIR = "filtered_data/masks"

    REJECTED_DIR = "filtered_data/rejected"

    REPORT_PATH = "filtering_report.csv"


# ============================================================
# 2. Three-Criterion Filtering Class
# ============================================================

class CODThreeCriterionFilter:

    def __init__(self, config=FilterConfig):

        self.config = config

        os.makedirs(config.ACCEPTED_IMAGE_DIR, exist_ok=True)
        os.makedirs(config.ACCEPTED_MASK_DIR, exist_ok=True)
        os.makedirs(config.REJECTED_DIR, exist_ok=True)

        # ----------------------------------------------------
        # Load SAM
        # ----------------------------------------------------

        if not os.path.exists(config.SAM_CHECKPOINT):
            raise FileNotFoundError(
                "SAM checkpoint not found: "
                + config.SAM_CHECKPOINT
            )

        sam = sam_model_registry[
            config.SAM_MODEL_TYPE
        ](
            checkpoint=config.SAM_CHECKPOINT
        )

        sam.to(device=config.DEVICE)

        self.sam_predictor = SamPredictor(sam)

        # ----------------------------------------------------
        # Load CLIP
        # ----------------------------------------------------

        self.clip_model = CLIPModel.from_pretrained(
            config.CLIP_MODEL_NAME
        ).to(config.DEVICE)

        self.clip_processor = CLIPProcessor.from_pretrained(
            config.CLIP_MODEL_NAME
        )

        self.clip_model.eval()

        print("SAM and CLIP models loaded successfully.")


    # ========================================================
    # 3. Image and Mask Loading
    # ========================================================

    def load_pair(self, image_path, mask_path):

        image = Image.open(image_path).convert("RGB")

        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        mask = mask.resize(
            (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE),
            Image.Resampling.NEAREST
        )

        image_array = np.array(image)

        mask_array = np.array(mask)

        mask_array = (
            mask_array > 127
        ).astype(np.uint8)

        return image, mask, image_array, mask_array


    # ========================================================
    # 4. Generate SAM Prompts
    # Centroid + Two Principal-Axis Extrema
    # ========================================================

    def generate_sam_prompts(self, mask):

        coordinates = np.column_stack(
            np.where(mask > 0)
        )

        if len(coordinates) < 3:
            raise ValueError(
                "Mask contains insufficient foreground pixels."
            )

        # Convert row-column coordinates to x-y
        points_xy = coordinates[:, ::-1].astype(np.float32)

        centroid = np.mean(points_xy, axis=0)

        centered = points_xy - centroid

        covariance = np.cov(
            centered.T
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        principal_axis = eigenvectors[:, -1]

        projections = centered @ principal_axis

        point_min = points_xy[
            np.argmin(projections)
        ]

        point_max = points_xy[
            np.argmax(projections)
        ]

        prompts = np.vstack([
            centroid,
            point_min,
            point_max
        ])

        labels = np.array([1, 1, 1])

        return prompts, labels


    # ========================================================
    # 5. Spatial Fidelity Using SAM
    # ========================================================

    def calculate_sam_fidelity(
        self,
        image,
        inherited_mask
    ):

        image_array = np.array(image)

        self.sam_predictor.set_image(
            image_array
        )

        point_coords, point_labels = (
            self.generate_sam_prompts(
                inherited_mask
            )
        )

        predicted_masks, scores, _ = (
            self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
        )

        best_index = np.argmax(scores)

        predicted_mask = predicted_masks[
            best_index
        ].astype(np.uint8)

        intersection = np.logical_and(
            predicted_mask,
            inherited_mask
        ).sum()

        union = np.logical_or(
            predicted_mask,
            inherited_mask
        ).sum()

        sam_iou = (
            intersection / union
            if union > 0
            else 0.0
        )

        return float(sam_iou), predicted_mask


    # ========================================================
    # 6. CLIP Foreground Feature Extraction
    # ========================================================

    def extract_clip_embedding(
        self,
        image,
        mask
    ):

        image_array = np.array(image)

        # Retain foreground and suppress background
        foreground = image_array.copy()

        foreground[mask == 0] = 0

        foreground_image = Image.fromarray(
            foreground
        )

        inputs = self.clip_processor(
            images=foreground_image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.config.DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            embedding = self.clip_model.get_image_features(
                **inputs
            )

            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=-1
            )

        return embedding.cpu().numpy()


    # ========================================================
    # 7. Semantic Fidelity Using CLIP
    # ========================================================

    def calculate_clip_fidelity(
        self,
        source_image,
        synthetic_image,
        mask
    ):

        source_embedding = self.extract_clip_embedding(
            source_image,
            mask
        )

        synthetic_embedding = self.extract_clip_embedding(
            synthetic_image,
            mask
        )

        similarity = cosine_similarity(
            source_embedding,
            synthetic_embedding
        )[0, 0]

        return float(similarity)


    # ========================================================
    # 8. Extract Spatial CLIP Patch Features
    # ========================================================

    def extract_patch_features(self, image):

        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.config.DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            vision_output = (
                self.clip_model.vision_model(
                    pixel_values=inputs["pixel_values"],
                    output_hidden_states=False
                )
            )

            # Remove CLS token; retain spatial patch tokens
            features = vision_output.last_hidden_state[
                0, 1:, :
            ]

        return features.cpu().numpy()


    # ========================================================
    # 9. Intra-Image Frechet Distance
    # ========================================================

    def calculate_frechet_distance(
        self,
        foreground_features,
        background_features
    ):

        if (
            len(foreground_features) < 2
            or len(background_features) < 2
        ):
            return float("inf")

        mean_fg = np.mean(
            foreground_features,
            axis=0
        )

        mean_bg = np.mean(
            background_features,
            axis=0
        )

        covariance_fg = np.cov(
            foreground_features,
            rowvar=False
        )

        covariance_bg = np.cov(
            background_features,
            rowvar=False
        )

        covariance_mean = sqrtm(
            covariance_fg @ covariance_bg
        )

        if np.iscomplexobj(covariance_mean):
            covariance_mean = covariance_mean.real

        distance = (
            np.sum((mean_fg - mean_bg) ** 2)
            +
            np.trace(
                covariance_fg
                + covariance_bg
                - 2 * covariance_mean
            )
        )

        return float(max(distance, 0.0))


    # ========================================================
    # 10. Calculate Camouflage Distance
    # ========================================================

    def calculate_camouflage_distance(
        self,
        image,
        mask
    ):

        patch_features = self.extract_patch_features(
            image
        )

        # CLIP ViT-B/32 has a 7 x 7 patch grid
        grid_size = int(
            np.sqrt(len(patch_features))
        )

        if grid_size * grid_size != len(patch_features):
            raise ValueError(
                "Unexpected CLIP spatial feature dimensions."
            )

        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (grid_size, grid_size),
            interpolation=cv2.INTER_NEAREST
        )

        patch_labels = resized_mask.flatten()

        foreground_features = patch_features[
            patch_labels == 1
        ]

        background_features = patch_features[
            patch_labels == 0
        ]

        distance = self.calculate_frechet_distance(
            foreground_features,
            background_features
        )

        return distance


    # ========================================================
    # 11. Normalize Camouflage Scores
    # Reference distances must come from training data only.
    # Smaller Frechet distance = stronger camouflage.
    # ========================================================

    @staticmethod
    def normalize_camouflage_score(
        distance,
        reference_distances
    ):

        reference_distances = np.asarray(
            reference_distances,
            dtype=float
        )

        reference_distances = reference_distances[
            np.isfinite(reference_distances)
        ]

        if len(reference_distances) == 0:
            raise ValueError(
                "Reference concealment distances are empty."
            )

        # Ascending rank: lower distance receives lower rank.
        # Convert to camouflage score: lower distance = higher score.
        rank = (
            np.sum(reference_distances < distance)
            + 0.5 * np.sum(reference_distances == distance)
        )

        percentile = rank / len(reference_distances)

        camouflage_score = 1.0 - percentile

        return float(camouflage_score)


    # ========================================================
    # 12. Evaluate One Synthetic Image-Mask Pair
    # ========================================================

    def evaluate_candidate(
        self,
        source_image_path,
        synthetic_image_path,
        synthetic_mask_path,
        reference_distances
    ):

        source_image, _, _, _ = self.load_pair(
            source_image_path,
            synthetic_mask_path
        )

        synthetic_image, mask, _, mask_array = (
            self.load_pair(
                synthetic_image_path,
                synthetic_mask_path
            )
        )

        # Criterion 1: Spatial fidelity
        sam_iou, predicted_mask = (
            self.calculate_sam_fidelity(
                synthetic_image,
                mask_array
            )
        )

        # Criterion 2: Semantic fidelity
        clip_similarity = (
            self.calculate_clip_fidelity(
                source_image,
                synthetic_image,
                mask_array
            )
        )

        # Criterion 3: Camouflage validity
        camouflage_distance = (
            self.calculate_camouflage_distance(
                synthetic_image,
                mask_array
            )
        )

        camouflage_score = (
            self.normalize_camouflage_score(
                camouflage_distance,
                reference_distances
            )
        )

        # Conjunctive acceptance rule
        spatial_pass = (
            sam_iou >= self.config.SAM_IOU_THRESHOLD
        )

        semantic_pass = (
            clip_similarity
            >= self.config.CLIP_SIMILARITY_THRESHOLD
        )

        camouflage_pass = (
            camouflage_score
            >= self.config.MIN_CAMOUFLAGE_SCORE
        )

        accepted = (
            spatial_pass
            and semantic_pass
            and camouflage_pass
        )

        return {
            "sam_iou": sam_iou,
            "clip_similarity": clip_similarity,
            "camouflage_distance": camouflage_distance,
            "camouflage_score": camouflage_score,
            "spatial_pass": spatial_pass,
            "semantic_pass": semantic_pass,
            "camouflage_pass": camouflage_pass,
            "accepted": accepted,
            "predicted_mask": predicted_mask
        }


    # ========================================================
    # 13. Filter Complete Synthetic Dataset
    # ========================================================

    def filter_dataset(
        self,
        source_image_paths,
        synthetic_image_paths,
        synthetic_mask_paths,
        reference_distances
    ):

        if not (
            len(source_image_paths)
            == len(synthetic_image_paths)
            == len(synthetic_mask_paths)
        ):
            raise ValueError(
                "Source images, synthetic images and masks "
                "must have equal lengths."
            )

        report = []

        accepted_count = 0
        rejected_count = 0

        for index in range(len(source_image_paths)):

            source_path = source_image_paths[index]

            synthetic_path = synthetic_image_paths[index]

            mask_path = synthetic_mask_paths[index]

            result = self.evaluate_candidate(
                source_image_path=source_path,
                synthetic_image_path=synthetic_path,
                synthetic_mask_path=mask_path,
                reference_distances=reference_distances
            )

            candidate_name = (
                Path(synthetic_path).stem
            )

            if result["accepted"]:

                output_image = os.path.join(
                    self.config.ACCEPTED_IMAGE_DIR,
                    candidate_name + ".png"
                )

                output_mask = os.path.join(
                    self.config.ACCEPTED_MASK_DIR,
                    candidate_name + ".png"
                )

                Image.open(
                    synthetic_path
                ).convert("RGB").save(
                    output_image
                )

                # Retain the inherited mask only after
                # the candidate passes all three criteria.
                Image.open(
                    mask_path
                ).convert("L").save(
                    output_mask
                )

                accepted_count += 1

            else:

                rejected_count += 1

            report.append({
                "source_image": source_path,
                "synthetic_image": synthetic_path,
                "sam_iou": result["sam_iou"],
                "clip_similarity": result["clip_similarity"],
                "camouflage_distance": result[
                    "camouflage_distance"
                ],
                "camouflage_score": result[
                    "camouflage_score"
                ],
                "spatial_pass": result["spatial_pass"],
                "semantic_pass": result["semantic_pass"],
                "camouflage_pass": result["camouflage_pass"],
                "accepted": result["accepted"]
            })

            print(
                f"{candidate_name} | "
                f"SAM: {result['sam_iou']:.4f} | "
                f"CLIP: {result['clip_similarity']:.4f} | "
                f"Camouflage: {result['camouflage_score']:.4f} | "
                f"Accepted: {result['accepted']}"
            )

        # Save filtering report
        dataframe = pd.DataFrame(report)

        dataframe.to_csv(
            self.config.REPORT_PATH,
            index=False
        )

        total = accepted_count + rejected_count

        acceptance_rate = (
            accepted_count / total * 100
            if total > 0
            else 0
        )

        print("\n========== FILTERING SUMMARY ==========")
        print("Total candidates:", total)
        print("Accepted candidates:", accepted_count)
        print("Rejected candidates:", rejected_count)
        print(f"Acceptance rate: {acceptance_rate:.2f}%")
        print("Report saved:", self.config.REPORT_PATH)

        return dataframe


# ============================================================
# 14. Example Execution
# ============================================================

if __name__ == "__main__":

    SOURCE_DIR = "datasets/train/images"
    SYNTHETIC_DIR = "synthetic_data/images"
    MASK_DIR = "synthetic_data/masks"

    source_paths = []
    synthetic_paths = []
    mask_paths = []

    for filename in sorted(
        os.listdir(SYNTHETIC_DIR)
    ):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        synthetic_path = os.path.join(
            SYNTHETIC_DIR,
            filename
        )

        mask_path = os.path.join(
            MASK_DIR,
            filename
        )

        # Generated names: source_synthetic_0.png
        source_name = filename.rsplit(
            "_synthetic_", 1
        )[0]

        source_path = os.path.join(
            SOURCE_DIR,
            source_name + ".jpg"
        )

        if not os.path.exists(source_path):
            source_path = os.path.join(
                SOURCE_DIR,
                source_name + ".png"
            )

        if (
            os.path.exists(source_path)
            and os.path.exists(mask_path)
        ):

            source_paths.append(source_path)
            synthetic_paths.append(synthetic_path)
            mask_paths.append(mask_path)

    # --------------------------------------------------------
    # IMPORTANT:
    # Replace this example with actual Fréchet distances
    # calculated for the original training images.
    # Do not use validation or test images here.
    # --------------------------------------------------------

    if not os.path.exists(
        "reference_camouflage_distances.npy"
    ):
        raise FileNotFoundError(
            "Create reference_camouflage_distances.npy "
            "from original training images before filtering."
        )

    reference_distances = np.load(
        "reference_camouflage_distances.npy"
    )

    filtering_model = CODThreeCriterionFilter()

    filtering_model.filter_dataset(
        source_image_paths=source_paths,
        synthetic_image_paths=synthetic_paths,
        synthetic_mask_paths=mask_paths,
        reference_distances=reference_distances
    )
