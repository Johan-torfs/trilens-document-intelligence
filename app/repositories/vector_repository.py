from pathlib import Path

import numpy as np


class VectorRepository:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, artifact_id: str, vector: np.ndarray) -> Path:
        path = self.storage_dir / f"{artifact_id}.npy"
        np.save(path, vector)
        return path

    def load(self, artifact_id: str) -> np.ndarray:
        path = self.storage_dir / f"{artifact_id}.npy"
        return np.load(path)