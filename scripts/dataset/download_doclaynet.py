from collections import Counter
from pathlib import Path
import shutil

from datasets import load_dataset


DATASET_ID = "docling-project/DocLayNet-v1.2"
REVISION = "529fab2988b260b3c4c7b7a757d2a5ae63b38735"
OUTPUT_DIR = Path("data/external/doclaynet")
COUNT_PER_CATEGORY = 5

CATEGORIES = {
    "financial_reports",
    "scientific_articles",
    "laws_and_regulations",
    "government_tenders",
    "manuals",
    "patents",
}


def download_doclaynet() -> None:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)

    counts: Counter[str] = Counter()

    seen_documents: dict[str, set[str]] = {
        category: set()
        for category in CATEGORIES
    }

    dataset = load_dataset(
        DATASET_ID,
        split="test",
        streaming=True,
        revision=REVISION,
    )

    for row in dataset:
        metadata = row["metadata"]
        category = metadata["doc_category"]

        if category not in CATEGORIES:
            continue

        if counts[category] >= COUNT_PER_CATEGORY:
            continue

        document_name = metadata["original_filename"]

        # Use only one page from each source document.
        if document_name in seen_documents[category]:
            continue

        seen_documents[category].add(document_name)
        counts[category] += 1

        category_dir = OUTPUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = (
            category_dir
            / f"{category}_{counts[category]:02d}.png"
        )

        row["image"].convert("RGB").save(destination)
        print(destination)

        if all(
            counts[category] == COUNT_PER_CATEGORY
            for category in CATEGORIES
        ):
            break

    incomplete = {
        category: counts[category]
        for category in CATEGORIES
        if counts[category] != COUNT_PER_CATEGORY
    }

    if incomplete:
        raise RuntimeError(
            f"Incomplete DocLayNet sample: {incomplete}"
        )


if __name__ == "__main__":
    download_doclaynet()