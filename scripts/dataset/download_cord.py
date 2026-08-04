from pathlib import Path
import shutil

from datasets import load_dataset


DATASET_ID = "naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
OUTPUT_DIR = Path("data/external/cord")
COUNT = 10


def download_cord() -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)

    dataset = load_dataset(
        DATASET_ID,
        split="test",
        streaming=True,
        revision=REVISION,
    )

    saved = 0

    for row in dataset:
        saved += 1
        path = OUTPUT_DIR / f"cord_{saved:03d}.png"

        row["image"].convert("RGB").save(path)
        print(path)

        if saved == COUNT:
            break

    if saved != COUNT:
        raise RuntimeError(
            f"Expected {COUNT} CORD images, got {saved}."
        )


if __name__ == "__main__":
    download_cord()