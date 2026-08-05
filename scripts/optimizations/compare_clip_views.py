from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.similarity import cosine_similarity
from app.strategies.clip_embedding import ClipEmbeddingStrategy


QUERIES = [
    "invoice",
    "business invoice",
    "toilet paper",
    "a very fancy chair",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CLIP similarity for different "
            "document preprocessing views."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/runtime/clip-view-comparison"),
    )

    return parser.parse_args()


def trim_white_space(
    image: Image.Image,
    threshold: int = 245,
    margin_ratio: float = 0.03,
) -> Image.Image:
    rgb_image = image.convert("RGB")
    grayscale = np.asarray(rgb_image.convert("L"))

    non_white = grayscale < threshold

    if not non_white.any():
        return rgb_image

    rows, columns = np.where(non_white)

    left = int(columns.min())
    top = int(rows.min())
    right = int(columns.max()) + 1
    bottom = int(rows.max()) + 1

    margin = round(
        max(rgb_image.size) * margin_ratio
    )

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(rgb_image.width, right + margin)
    bottom = min(rgb_image.height, bottom + margin)

    return rgb_image.crop(
        (left, top, right, bottom)
    )


def pad_to_square(
    image: Image.Image,
) -> Image.Image:
    rgb_image = image.convert("RGB")

    side_length = max(rgb_image.size)

    canvas = Image.new(
        "RGB",
        (side_length, side_length),
        "white",
    )

    offset = (
        (side_length - rgb_image.width) // 2,
        (side_length - rgb_image.height) // 2,
    )

    canvas.paste(rgb_image, offset)

    return canvas


def create_square_tiles(
    image: Image.Image,
) -> dict[str, Image.Image]:
    rgb_image = image.convert("RGB")

    if rgb_image.height <= rgb_image.width:
        return {
            "tile_full": pad_to_square(rgb_image),
        }

    tile_size = rgb_image.width
    maximum_start = rgb_image.height - tile_size

    positions = {
        "tile_top": 0,
        "tile_middle": maximum_start // 2,
        "tile_bottom": maximum_start,
    }

    return {
        name: rgb_image.crop(
            (
                0,
                start,
                tile_size,
                start + tile_size,
            )
        )
        for name, start in positions.items()
    }


def main() -> None:
    args = parse_args()

    if not args.image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {args.image_path}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with Image.open(args.image_path) as source:
        original = source.convert("RGB")

    trimmed = trim_white_space(original)

    views: dict[str, Image.Image] = {
        "current": original,
        "full_page_padded": pad_to_square(original),
        "trimmed_padded": pad_to_square(trimmed),
        **create_square_tiles(original),
    }

    for name, view in views.items():
        view.save(
            args.output_dir / f"{name}.png"
        )

    strategy = ClipEmbeddingStrategy()

    image_embeddings = {
        name: strategy.embed_image(view)
        for name, view in views.items()
    }

    print()
    print(f"Original size: {original.size}")
    print(f"Trimmed size:  {trimmed.size}")
    print()

    header = f"{'Query':<24}"

    for name in views:
        header += f"{name:>20}"

    print(header)
    print("-" * len(header))

    for query in QUERIES:
        text_embedding = strategy.embed_text(query)

        row = f"{query:<24}"

        for name in views:
            score = cosine_similarity(
                text_embedding,
                image_embeddings[name],
            )

            row += f"{score:>20.4f}"

        print(row)


if __name__ == "__main__":
    main()