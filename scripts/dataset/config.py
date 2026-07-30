from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    generated_dir: Path = Path("data/generated")
    external_dir: Path = Path("data/external")
    augmented_dir: Path = Path("data/augmented")
    manifest_path: Path = Path("data/manifest.jsonl")
    summary_path: Path = Path("data/dataset_summary.json")

    synthetic_count: int = 30
    cord_count: int = 10
    augmentation_count: int = 10
    seed: int = 42