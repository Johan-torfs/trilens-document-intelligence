from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np

from app.domain.vector import (
    PayloadValue,
    VectorPoint,
    VectorSearchResult,
)


class VectorRepository(ABC):
    @abstractmethod
    def save_batch(
        self,
        points: Sequence[VectorPoint],
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        vector_name: str,
        query_vector: np.ndarray,
        limit: int,
        filters: Mapping[str, PayloadValue] | None = None,
        exact: bool | None = None,
    ) -> list[VectorSearchResult]:
        ...

    @abstractmethod
    def count(
        self,
        filters: Mapping[str, PayloadValue] | None = None,
    ) -> int:
        ...

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
    ) -> None:
        ...