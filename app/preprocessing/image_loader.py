from pathlib import Path
from dataclasses import dataclass

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageLoadResult:
    image: Image.Image
    original_size: tuple[int, int]
    transforms: list[str]


def load_image(image_path: Path) -> ImageLoadResult:
    with Image.open(image_path) as source:
        original_size = source.size
        transforms: list[str] = []

        orientation = source.getexif().get(274)

        corrected = ImageOps.exif_transpose(source)

        if orientation not in (None, 1):
            transforms.append("exif_transpose")

        if corrected.mode != "RGB":
            transforms.append("convert_rgb")

        image = corrected.convert("RGB")

    return ImageLoadResult(
        image=image,
        original_size=original_size,
        transforms=transforms,
    )