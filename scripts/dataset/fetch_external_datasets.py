from argparse import ArgumentParser
from pathlib import Path

from scripts.dataset.download_cord import download_cord
from scripts.dataset.download_doclaynet import (
    download_doclaynet,
)
from scripts.dataset.download_funsd import download_funsd


def count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
        }
    )


def main() -> None:
    parser = ArgumentParser(
        description="Fetch the TriLens external dataset."
    )

    parser.add_argument(
        "--funsd-archive",
        required=True,
        type=Path,
    )

    arguments = parser.parse_args()

    if not arguments.funsd_archive.is_file():
        raise FileNotFoundError(
            f"FUNSD archive not found: "
            f"{arguments.funsd_archive}"
        )

    download_cord()
    download_funsd(arguments.funsd_archive)
    download_doclaynet()

    expected = {
        Path("data/external/cord"): 10,
        Path("data/external/funsd"): 10,
        Path("data/external/doclaynet"): 30,
    }

    for directory, expected_count in expected.items():
        actual_count = count_images(directory)

        if actual_count != expected_count:
            raise RuntimeError(
                f"{directory}: expected {expected_count}, "
                f"found {actual_count}."
            )

        print(f"{directory}: {actual_count} images")

    print("External dataset ready: 50 images.")


if __name__ == "__main__":
    import os
    import sys

    main()

    # Avoid a PyArrow shutdown crash after all files
    # have been written and validated.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)