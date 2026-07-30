import json
from pathlib import Path

from PIL import Image


def validate_manifest(manifest_path: Path) -> None:
    ids: set[str] = set()
    errors: list[str] = []

    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        record = json.loads(line)

        document_id = record["id"]
        path = Path(record["path"])

        if document_id in ids:
            errors.append(f"Regel {line_number}: dubbel ID '{document_id}'")

        ids.add(document_id)

        if not path.exists():
            errors.append(f"Regel {line_number}: bestand ontbreekt: {path}")
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"Regel {line_number}: ongeldige afbeelding: {exc}")

    if errors:
        raise ValueError("\n".join(errors))

    print(f"Dataset geldig: {len(ids)} documenten")


if __name__ == "__main__":
    validate_manifest(Path("data/manifest.jsonl"))