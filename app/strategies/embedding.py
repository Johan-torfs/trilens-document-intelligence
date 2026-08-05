from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from PIL import Image


class EmbeddingStrategy(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_version(self) -> str | None:
        ...

    @abstractmethod
    def embed_images(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        """Return an embedding matrix shaped (image_count, dimensions)."""
        ...

    def embed_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        embeddings = self.embed_images([image])

        if embeddings.ndim != 2 or embeddings.shape[0] != 1:
            raise ValueError(
                "Image strategy returned an invalid embedding batch."
            )

        return embeddings[0]

    @abstractmethod
    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        ...