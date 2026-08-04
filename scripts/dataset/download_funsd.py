from io import BytesIO
from pathlib import Path
import shutil
from zipfile import ZipFile

from PIL import Image, UnidentifiedImageError


OUTPUT_DIR = Path("data/external/funsd")
COUNT_PER_SPLIT = 5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def evenly_spaced(
    paths: list[str],
    count: int,
) -> list[str]:
    if len(paths) < count:
        raise RuntimeError(
            f"Need {count} images, found {len(paths)}."
        )

    if count == 1:
        return [paths[0]]

    return [
        paths[
            round(index * (len(paths) - 1) / (count - 1))
        ]
        for index in range(count)
    ]


def is_valid_image_member(
    archive: ZipFile,
    name: str,
) -> bool:
    if "__MACOSX/" in name:
        return False

    if Path(name).name.startswith("._"):
        return False

    try:
        with archive.open(name) as source:
            with Image.open(BytesIO(source.read())) as image:
                image.verify()

        return True

    except (UnidentifiedImageError, OSError):
        return False


def download_funsd(archive_path: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"FUNSD archive not found: {archive_path}"
        )

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)

    with ZipFile(archive_path) as archive:
        candidate_paths = sorted(
            name
            for name in archive.namelist()
            if (
                Path(name).suffix.lower() in IMAGE_EXTENSIONS
                and "/images/" in name
            )
        )

        image_paths = [
            name
            for name in candidate_paths
            if is_valid_image_member(archive, name)
        ]

        training = [
            path
            for path in image_paths
            if "training_data/images" in path
        ]

        testing = [
            path
            for path in image_paths
            if "testing_data/images" in path
        ]

        selections = {
            "train": evenly_spaced(
                training,
                COUNT_PER_SPLIT,
            ),
            "test": evenly_spaced(
                testing,
                COUNT_PER_SPLIT,
            ),
        }

        for split, paths in selections.items():
            for index, source_path in enumerate(
                paths,
                start=1,
            ):
                destination = (
                    OUTPUT_DIR
                    / f"funsd_{split}_{index:02d}.png"
                )

                with archive.open(source_path) as source:
                    with Image.open(
                        BytesIO(source.read())
                    ) as image:
                        image.convert("RGB").save(destination)

                print(destination)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive_path",
        type=Path,
    )

    arguments = parser.parse_args()
    download_funsd(arguments.archive_path)