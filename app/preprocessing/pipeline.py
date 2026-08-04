from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PIL import Image

from app.preprocessing.image_loader import load_image
from app.preprocessing.image_validator import validate_image
from app.preprocessing.transforms import resize_image


@dataclass(frozen=True)
class PreprocessingResult:
    image: Image.Image
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    transforms: list[str]
    duration_ms: float


def preprocess_pil_image(
    image: Image.Image,
    max_width: int = 1600,
    max_height: int = 1600,
) -> PreprocessingResult:
    started_at = perf_counter()

    prepared_image = image.convert("RGB")

    transforms: list[str] = []

    if image.mode != "RGB":
        transforms.append("convert_rgb")

    return _resize(
        image=prepared_image,
        original_size=image.size,
        transforms=transforms,
        max_width=max_width,
        max_height=max_height,
        started_at=started_at,
    )


def preprocess_image(
    image_path: Path,
    max_width: int = 1600,
    max_height: int = 1600,
) -> PreprocessingResult:
    started_at = perf_counter()

    validate_image(image_path)
    load_result = load_image(image_path)

    return _resize(
        image=load_result.image,
        original_size=load_result.original_size,
        transforms=load_result.transforms,
        max_width=max_width,
        max_height=max_height,
        started_at=started_at,
    )


def _resize(
    image: Image.Image,
    original_size: tuple[int, int],
    transforms: list[str],
    max_width: int,
    max_height: int,
    started_at: float,
) -> PreprocessingResult:
    processed_image, resize_transforms = resize_image(
        image,
        max_width=max_width,
        max_height=max_height,
    )

    return PreprocessingResult(
        image=processed_image,
        original_size=original_size,
        processed_size=processed_image.size,
        transforms=transforms + resize_transforms,
        duration_ms=(perf_counter() - started_at) * 1000,
    )


def save_processed_image(
    result: PreprocessingResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output_path, format="PNG")