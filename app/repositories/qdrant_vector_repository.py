from collections.abc import Mapping, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.domain.vector import (
    PayloadValue,
    VectorPoint,
    VectorSearchResult,
)
from app.repositories.vector_repository import VectorRepository


class VectorRepositoryConfigurationError(RuntimeError):
    pass


class QdrantVectorRepository(VectorRepository):
    INDEXED_PAYLOAD_FIELDS = (
        "document_id",
        "checksum",
        "unit_type",
        "model_name",
        "model_version",
    )

    def __init__(
        self,
        url: str,
        collection_name: str,
        timeout_seconds: float = 10.0,
        exact_search: bool = True,
    ) -> None:
        self._collection_name = collection_name
        self._exact_search = exact_search
        self._client = QdrantClient(
            url=url,
            timeout=timeout_seconds,
        )

    def save_batch(
        self,
        points: Sequence[VectorPoint],
    ) -> None:
        if not points:
            return

        vector_dimensions: dict[str, int] = {}
        qdrant_points: list[models.PointStruct] = []

        for point in points:
            values = np.asarray(
                point.values,
                dtype=np.float32,
            )

            if values.ndim != 1 or values.size == 0:
                raise ValueError(
                    f"Vector '{point.id}' must be one-dimensional "
                    "and non-empty."
                )

            dimensions = int(values.size)
            existing_dimensions = vector_dimensions.get(
                point.vector_name
            )

            if (
                existing_dimensions is not None
                and existing_dimensions != dimensions
            ):
                raise ValueError(
                    f"Vector space '{point.vector_name}' contains "
                    "inconsistent dimensions."
                )

            vector_dimensions[point.vector_name] = dimensions

            qdrant_points.append(
                models.PointStruct(
                    id=point.id,
                    vector={
                        point.vector_name: values.tolist(),
                    },
                    payload=dict(point.payload),
                )
            )

        self._ensure_collection(vector_dimensions)

        self._client.upsert(
            collection_name=self._collection_name,
            points=qdrant_points,
            wait=True,
        )

    def search(
        self,
        vector_name: str,
        query_vector: np.ndarray,
        limit: int,
        filters: Mapping[str, PayloadValue] | None = None,
        exact: bool | None = None,
    ) -> list[VectorSearchResult]:
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")

        values = np.asarray(
            query_vector,
            dtype=np.float32,
        )

        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "Query vector must be one-dimensional and non-empty."
            )

        if not self._client.collection_exists(
            self._collection_name
        ):
            return []

        if not self._vector_name_exists(vector_name):
            return []

        response = self._client.query_points(
            collection_name=self._collection_name,
            query=values.tolist(),
            using=vector_name,
            query_filter=self._build_filter(filters),
            search_params=models.SearchParams(
                exact=(
                    self._exact_search
                    if exact is None
                    else exact
                )
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            VectorSearchResult(
                point_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in response.points
        ]

    def count(
        self,
        filters: Mapping[str, PayloadValue] | None = None,
    ) -> int:
        if not self._client.collection_exists(
            self._collection_name
        ):
            return 0

        result = self._client.count(
            collection_name=self._collection_name,
            count_filter=self._build_filter(filters),
            exact=True,
        )

        return int(result.count)

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        if not self._client.collection_exists(
            self._collection_name
        ):
            return

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=self._build_filter(
                    {"document_id": document_id}
                )
            ),
            wait=True,
        )

    def _ensure_collection(
        self,
        vector_dimensions: Mapping[str, int],
    ) -> None:
        if not self._client.collection_exists(
            self._collection_name
        ):
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config={
                    name: models.VectorParams(
                        size=dimensions,
                        distance=models.Distance.COSINE,
                    )
                    for name, dimensions
                    in vector_dimensions.items()
                },
            )

            self._ensure_payload_indexes()
            return

        collection = self._client.get_collection(
            self._collection_name
        )

        vectors_config = collection.config.params.vectors

        if not isinstance(vectors_config, dict):
            raise VectorRepositoryConfigurationError(
                "The Qdrant collection must use named vectors."
            )

        for name, dimensions in vector_dimensions.items():
            existing = vectors_config.get(name)

            if existing is None:
                self._client.create_vector_name(
                    collection_name=self._collection_name,
                    vector_name=name,
                    vector_name_config=(
                        models.DenseVectorNameConfig(
                            dense=models.DenseVectorConfig(
                                size=dimensions,
                                distance=models.Distance.COSINE,
                            )
                        )
                    ),
                )
                continue

            if int(existing.size) != dimensions:
                raise VectorRepositoryConfigurationError(
                    f"Vector space '{name}' expects "
                    f"{existing.size} dimensions, got {dimensions}."
                )

            if existing.distance != models.Distance.COSINE:
                raise VectorRepositoryConfigurationError(
                    f"Vector space '{name}' must use cosine distance."
                )

        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        collection = self._client.get_collection(
            self._collection_name
        )

        existing_indexes = set(collection.payload_schema)

        for field_name in self.INDEXED_PAYLOAD_FIELDS:
            if field_name in existing_indexes:
                continue

            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    def _vector_name_exists(
        self,
        vector_name: str,
    ) -> bool:
        collection = self._client.get_collection(
            self._collection_name
        )

        vectors_config = collection.config.params.vectors

        return (
            isinstance(vectors_config, dict)
            and vector_name in vectors_config
        )

    @staticmethod
    def _build_filter(
        filters: Mapping[str, PayloadValue] | None,
    ) -> models.Filter | None:
        if not filters:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in filters.items()
            ]
        )