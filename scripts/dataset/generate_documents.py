import json
from pathlib import Path
from collections import Counter

from .templates.invoice import generate_invoice
from .templates.purchase_order import generate_purchase_order
from .templates.delivery_note import generate_delivery_note
from .templates.receipt import generate_receipt
from .templates.application_form import generate_application_form
from .templates.identity_card import generate_identity_card

from .augment_images import augment_image
from .download_cord import collect_cord_samples

DOCUMENT_COUNTS = {
    "invoice": 8,
    "purchase_order": 7,
    "receipt": 4,
    "delivery_note": 4,
    "application_form": 4,
    "identity_card": 3,
}


def main() -> None:
    output_dir = Path("data/generated/invoices")
    manifest_path = Path("data/manifest.jsonl")

    records = []

    for index in range(1, DOCUMENT_COUNTS["invoice"] + 1):
        record = generate_invoice(output_dir, index)
        records.append(record)
        print(f"Generated: {record['path']}")

    purchase_order_dir = Path("data/generated/purchase_orders")
    for index in range(1, DOCUMENT_COUNTS["purchase_order"] + 1):
        record = generate_purchase_order(purchase_order_dir, index)
        records.append(record)
        print(f"Generated: {record['path']}")

    delivery_note_dir = Path("data/generated/delivery_notes")
    for index in range(1, DOCUMENT_COUNTS["delivery_note"] + 1):
        record = generate_delivery_note(delivery_note_dir, index)
        records.append(record)
        print(f"Generated: {record['path']}")

    application_form_dir = Path("data/generated/application_forms")
    for index in range(1, DOCUMENT_COUNTS["application_form"] + 1):
        record = generate_application_form(application_form_dir, index)
        records.append(record)
        print(f"Generated: {record['path']}")

    receipt_dir = Path("data/generated/receipts")
    for index in range(1, DOCUMENT_COUNTS["receipt"] + 1):
        record = generate_receipt(receipt_dir, index)
        records.append(record)
        print(f"Generated: {record['path']}")

    identity_card_dir = Path("data/generated/identity_cards")
    for index in range(1, DOCUMENT_COUNTS["identity_card"] + 1):
        record = generate_identity_card(identity_card_dir, index)
        records.append(record)
        print(f"Generated: {record['path']}")

    cord_records = collect_cord_samples(
        output_dir=Path("data/external/cord"),
        count=16,
    )

    records.extend(cord_records)

    augmented_dir = Path("data/augmented")
    augmented_records = []

    for index, record in enumerate(records[:10], start=1):
        source_path = Path(record["path"])
        output_path = augmented_dir / f"{record['id']}_aug.jpg"

        transforms = augment_image(
            source_path=source_path,
            output_path=output_path,
            seed=1000 + index,
        )

        augmented_records.append(
            {
                **record,
                "id": f"{record['id']}_aug",
                "path": output_path.as_posix(),
                "source": "synthetic_augmentation",
                "source_document_id": record["id"],
                "augmentations": transforms,
                "seed": 1000 + index,
            }
        )

    records.extend(augmented_records)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:
            manifest.write(json.dumps(record) + "\n")

    print(f"Manifest written: {manifest_path}")

    summary_path = Path("data/dataset_summary.json")

    summary = {
        "total_documents": len(records),
        "document_type_counts": dict(
            Counter(record["document_type"] for record in records)
        ),
        "source_counts": dict(
            Counter(record["source"] for record in records)
        ),
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()