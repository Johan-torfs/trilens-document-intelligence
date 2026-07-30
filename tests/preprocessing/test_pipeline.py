from pathlib import Path

import pytest
from PIL import Image

from app.preprocessing.image_validator import (
    ImageValidationError,
    validate_image,
)
from app.preprocessing.pipeline import preprocess_image


def test_preprocesses_image_to_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    Image.new("L", (1200, 800)).save(image_path)

    result = preprocess_image(
        image_path,
        max_width=600,
        max_height=600,
    )

    assert result.image.mode == "RGB"
    assert result.processed_size[0] <= 600
    assert result.processed_size[1] <= 600
    assert "convert_rgb" in result.transforms
    assert "resize" in result.transforms


def test_rejects_missing_image(tmp_path: Path) -> None:
    with pytest.raises(ImageValidationError):
        validate_image(tmp_path / "missing.png")


def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("not an image")

    with pytest.raises(ImageValidationError):
        validate_image(file_path)


def test_rejects_corrupted_image(tmp_path: Path) -> None:
    image_path = tmp_path / "corrupted.png"
    image_path.write_bytes(b"not-a-real-image")

    with pytest.raises(ImageValidationError):
        validate_image(image_path)


def test_converts_transparent_png_to_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "transparent.png"

    Image.new(
        mode="RGBA",
        size=(100, 100),
        color=(255, 255, 255, 0),
    ).save(image_path)

    result = preprocess_image(image_path)

    assert result.image.mode == "RGB"
    assert "convert_rgb" in result.transforms


def test_resize_preserves_aspect_ratio(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    Image.new("RGB", (1200, 600)).save(image_path)

    result = preprocess_image(
        image_path,
        max_width=600,
        max_height=600,
    )

    assert result.processed_size == (600, 300)
    assert "resize" in result.transforms

def test_small_image_is_not_resized(tmp_path: Path) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (400, 300)).save(image_path)

    result = preprocess_image(
        image_path,
        max_width=800,
        max_height=800,
    )

    assert result.processed_size == (400, 300)
    assert "resize" not in result.transforms