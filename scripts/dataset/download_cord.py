from pathlib import Path

from datasets import load_dataset


def collect_cord_samples(
    output_dir: Path,
    count: int = 10,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "naver-clova-ix/cord-v2",
        split="train",
        streaming=True,
    )

    records = []

    for index, example in enumerate(dataset):
        if index >= count:
            break

        document_id = f"cord_{index + 1:03d}"
        output_path = output_dir / f"{document_id}.png"

        image = example["image"].convert("RGB")
        image.save(output_path)

        records.append(
            {
                "id": document_id,
                "path": output_path.as_posix(),
                "source": "cord-v2",
                "document_type": "receipt",
                "contains_table": True,
                "safe_for_public_repo": False,
                "license": "CC-BY-4.0",
            }
        )

    return records