from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class RetrievalStrategy(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def calibrate_score(raw: float) -> float:
        ...