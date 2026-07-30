from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ImageValidationError(ValueError):
    pass


def validate_image(image_path: Path) -> None:
    if not image_path.exists():
        raise ImageValidationError(
            f"Afbeelding bestaat niet: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Niet-ondersteund bestandstype: {image_path.suffix}"
        )

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError(
            f"Ongeldige of beschadigde afbeelding: {image_path}"
        ) from error