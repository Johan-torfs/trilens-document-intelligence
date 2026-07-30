from pathlib import Path
from dataclasses import dataclass
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


def preprocess_image(
    image_path: Path,
    max_width: int = 1600,
    max_height: int = 1600,
) -> PreprocessingResult:
    started_at = perf_counter()

    validate_image(image_path)

    load_result = load_image(image_path)

    processed_image, resize_transforms = resize_image(
        load_result.image,
        max_width=max_width,
        max_height=max_height,
    )

    return PreprocessingResult(
        image=processed_image,
        original_size=load_result.original_size,
        processed_size=processed_image.size,
        transforms=load_result.transforms + resize_transforms,
        duration_ms=(perf_counter() - started_at) * 1000,
    )


def save_processed_image(
    result: PreprocessingResult,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.image.save(output_path, format="PNG")