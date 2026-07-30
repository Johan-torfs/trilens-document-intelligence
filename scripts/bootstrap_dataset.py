from dataset.generate_documents import main
from dataset.validate_dataset import validate_manifest
from pathlib import Path


if __name__ == "__main__":
    main()
    validate_manifest(Path("data/manifest.jsonl"))