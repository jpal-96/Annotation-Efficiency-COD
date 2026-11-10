# ============================================================
# diffusion.py
# Latent Diffusion-Based Synthetic Image Generation for COD
# ============================================================

import os
import cv2
import torch
import numpy as np

from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionInpaintPipeline


# ============================================================
# 1. Configuration
# ============================================================

class DiffusionConfig:

    MODEL_ID = "runwayml/stable-diffusion-inpainting"

    IMAGE_SIZE = 512
    NUM_INFERENCE_STEPS = 40
    GUIDANCE_SCALE = 7.5

    STRENGTH = 0.65
    NUM_IMAGES_PER_SOURCE = 2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_IMAGE_DIR = "synthetic_data/images"
    OUTPUT_MASK_DIR = "synthetic_data/masks"


# ============================================================
# 2. Load Pretrained Latent Diffusion Model
# ============================================================

class CODDiffusionGenerator:

    def __init__(self, config=DiffusionConfig):

        self.config = config

        os.makedirs(config.OUTPUT_IMAGE_DIR, exist_ok=True)
        os.makedirs(config.OUTPUT_MASK_DIR, exist_ok=True)

        dtype = (
            torch.float16
            if config.DEVICE == "cuda"
            else torch.float32
        )

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config.MODEL_ID,
            torch_dtype=dtype
        )

        self.pipeline = self.pipeline.to(config.DEVICE)

        if config.DEVICE == "cuda":
            self.pipeline.enable_attention_slicing()

        print("Latent diffusion model loaded successfully.")


    # ========================================================
    # 3. Load and Resize Image and Mask
    # ========================================================

    def load_image_mask(self, image_path, mask_path):

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

        # Ensure binary mask
        mask_array = np.array(mask)
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)

        mask = Image.fromarray(mask_array)

        return image, mask


    # ========================================================
    # 4. Generate Synthetic Camouflaged Image
    # ========================================================

    def generate_image(
        self,
        image,
        mask,
        prompt=None,
        negative_prompt=None,
        seed=42
    ):

        if prompt is None:
            prompt = (
                "A realistic natural environment, "
                "a naturally camouflaged object blending "
                "with the surrounding background, "
                "realistic texture, natural lighting, "
                "high visual detail"
            )

        if negative_prompt is None:
            negative_prompt = (
                "cartoon, painting, artificial texture, "
                "blurry, distorted object, unrealistic colors, "
                "low quality, watermark"
            )

        generator = torch.Generator(
            device=self.config.DEVICE
        ).manual_seed(seed)

        with torch.inference_mode():

            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=self.config.NUM_INFERENCE_STEPS,
                guidance_scale=self.config.GUIDANCE_SCALE,
                strength=self.config.STRENGTH,
                generator=generator
            )

        synthetic_image = result.images[0]

        return synthetic_image


    # ========================================================
    # 5. Generate Multiple Synthetic Samples
    # ========================================================

    def generate_dataset(
        self,
        image_paths,
        mask_paths,
        num_images=None
    ):

        if len(image_paths) != len(mask_paths):
            raise ValueError(
                "The number of images and masks must be equal."
            )

        if num_images is None:
            num_images = self.config.NUM_IMAGES_PER_SOURCE

        generated_count = 0

        for index, (image_path, mask_path) in enumerate(
            zip(image_paths, mask_paths)
        ):

            image, mask = self.load_image_mask(
                image_path,
                mask_path
            )

            image_name = Path(image_path).stem

            for sample_id in range(num_images):

                seed = 42 + index * num_images + sample_id

                synthetic_image = self.generate_image(
                    image=image,
                    mask=mask,
                    seed=seed
                )

                # Save synthetic image
                image_output_path = os.path.join(
                    self.config.OUTPUT_IMAGE_DIR,
                    f"{image_name}_synthetic_{sample_id}.png"
                )

                # Save inherited mask separately.
                # This mask is provisional and must be validated.
                mask_output_path = os.path.join(
                    self.config.OUTPUT_MASK_DIR,
                    f"{image_name}_synthetic_{sample_id}.png"
                )

                synthetic_image.save(image_output_path)
                mask.save(mask_output_path)

                generated_count += 1

                print(
                    f"Generated: {generated_count} | "
                    f"Source: {image_name} | "
                    f"Sample: {sample_id}"
                )

        print("\nSynthetic generation completed.")
        print("Total generated candidates:", generated_count)

        return generated_count


# ============================================================
# 6. Example Execution
# ============================================================

if __name__ == "__main__":

    IMAGE_DIR = "datasets/train/images"
    MASK_DIR = "datasets/train/masks"

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    image_paths = sorted([
        os.path.join(IMAGE_DIR, filename)
        for filename in os.listdir(IMAGE_DIR)
        if filename.lower().endswith(image_extensions)
    ])

    mask_paths = [
        os.path.join(
            MASK_DIR,
            Path(image_path).stem + ".png"
        )
        for image_path in image_paths
    ]

    # Check whether all corresponding masks exist
    valid_pairs = [
        (image_path, mask_path)
        for image_path, mask_path in zip(image_paths, mask_paths)
        if os.path.exists(mask_path)
    ]

    if not valid_pairs:
        raise FileNotFoundError(
            "No matching image-mask pairs were found."
        )

    image_paths, mask_paths = zip(*valid_pairs)

    generator = CODDiffusionGenerator()

    generator.generate_dataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        num_images=2
    )
