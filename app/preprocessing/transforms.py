from PIL import Image


def resize_image(
    image: Image.Image,
    max_width: int = 1600,
    max_height: int = 1600,
) -> tuple[Image.Image, list[str]]:
    resized = image.copy()

    resized.thumbnail(
        (max_width, max_height),
        Image.Resampling.LANCZOS,
    )

    transforms = []

    if resized.size != image.size:
        transforms.append("resize")

    return resized, transforms