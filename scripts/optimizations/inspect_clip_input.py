from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the exact image representation sent to CLIP."
    )

    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to the original document image.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runtime/clip_model_input.png"),
        help="Path for the reconstructed CLIP input.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {args.image_path}"
        )

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with Image.open(args.image_path) as source:
        image = source.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"][0]

    image_processor = processor.image_processor

    mean = torch.tensor(
        image_processor.image_mean,
        dtype=pixel_values.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        image_processor.image_std,
        dtype=pixel_values.dtype,
    ).view(3, 1, 1)

    reconstructed = (
        pixel_values * std + mean
    ).clamp(0.0, 1.0)

    image_array = (
        reconstructed
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).round().astype(np.uint8)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(image_array).save(args.output)

    print(f"Original size: {image.size}")
    print(
        "CLIP tensor shape:",
        tuple(inputs["pixel_values"].shape),
    )
    print(f"Saved CLIP input to: {args.output}")


if __name__ == "__main__":
    main()