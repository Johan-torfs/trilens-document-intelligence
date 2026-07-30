from pathlib import Path
from random import Random

from PIL import Image, ImageEnhance, ImageFilter


def augment_image(
    source_path: Path,
    output_path: Path,
    seed: int,
) -> list[dict]:
    rng = Random(seed)

    with Image.open(source_path) as source:
        image = source.convert("RGB")

    transforms: list[dict] = []

    angle = rng.uniform(-6.0, 6.0)
    image = image.rotate(angle, expand=True, fillcolor="white")
    transforms.append({"type": "rotation", "angle": round(angle, 2)})

    if rng.random() < 0.5:
        radius = rng.uniform(0.4, 1.2)
        image = image.filter(ImageFilter.GaussianBlur(radius))
        transforms.append({"type": "blur", "radius": round(radius, 2)})

    if rng.random() < 0.5:
        factor = rng.uniform(0.7, 1.3)
        image = ImageEnhance.Contrast(image).enhance(factor)
        transforms.append({"type": "contrast", "factor": round(factor, 2)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=85)

    return transforms