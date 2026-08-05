from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from qdrant_client import models

from app.domain.vector import VectorPoint
from app.repositories.qdrant_vector_repository import (
    QdrantVectorRepository,
)


def test_saves_batch_and_creates_collection(
    vector_repository: QdrantVectorRepository,
    qdrant_client_mock: MagicMock,
) -> None:
    qdrant_client_mock.collection_exists.return_value = False
    qdrant_client_mock.get_collection.return_value = (
        SimpleNamespace(payload_schema={})
    )

    vector_repository.save_batch(
        [
            VectorPoint(
                id="00000000-0000-0000-0000-000000000001",
                vector_name="visual",
                values=np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                payload={
                    "document_id": "document-1",
                    "checksum": "checksum-1",
                    "unit_type": "page",
                    "page_number": 1,
                    "model_name": "siglip",
                    "model_version": "1",
                },
            )
        ]
    )

    create_call = (
        qdrant_client_mock.create_collection.call_args
    )

    vectors_config = (
        create_call.kwargs["vectors_config"]
    )

    assert vectors_config["visual"].size == 3
    assert (
        vectors_config["visual"].distance
        is models.Distance.COSINE
    )

    upsert_call = qdrant_client_mock.upsert.call_args
    stored_point = upsert_call.kwargs["points"][0]

    assert stored_point.id == (
        "00000000-0000-0000-0000-000000000001"
    )
    assert stored_point.vector["visual"] == [
        1.0,
        0.0,
        0.0,
    ]
    assert stored_point.payload["document_id"] == (
        "document-1"
    )


def test_rejects_inconsistent_vector_dimensions(
    vector_repository: QdrantVectorRepository,
) -> None:
    points = [
        VectorPoint(
            id="00000000-0000-0000-0000-000000000001",
            vector_name="visual",
            values=np.array([1.0, 0.0]),
            payload={},
        ),
        VectorPoint(
            id="00000000-0000-0000-0000-000000000002",
            vector_name="visual",
            values=np.array([1.0, 0.0, 0.0]),
            payload={},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="inconsistent dimensions",
    ):
        vector_repository.save_batch(points)


def test_searches_named_vector(
    vector_repository: QdrantVectorRepository,
    qdrant_client_mock: MagicMock,
) -> None:
    qdrant_client_mock.collection_exists.return_value = True
    qdrant_client_mock.get_collection.return_value = (
        _collection_info()
    )
    qdrant_client_mock.query_points.return_value = (
        SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=(
                        "00000000-0000-0000-0000-"
                        "000000000001"
                    ),
                    score=0.91,
                    payload={
                        "document_id": "document-1",
                        "page_number": 2,
                    },
                )
            ]
        )
    )

    results = vector_repository.search(
        vector_name="visual",
        query_vector=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        limit=5,
        filters={"model_name": "siglip"},
    )

    assert len(results) == 1
    assert results[0].score == pytest.approx(0.91)
    assert (
        results[0].payload["document_id"]
        == "document-1"
    )
    assert results[0].payload["page_number"] == 2

    query_call = (
        qdrant_client_mock.query_points.call_args
    )

    assert query_call.kwargs["using"] == "visual"
    assert query_call.kwargs["limit"] == 5
    assert (
        query_call.kwargs["search_params"].exact
        is True
    )


def test_search_returns_empty_when_collection_is_missing(
    vector_repository: QdrantVectorRepository,
    qdrant_client_mock: MagicMock,
) -> None:
    qdrant_client_mock.collection_exists.return_value = False

    results = vector_repository.search(
        vector_name="visual",
        query_vector=np.array([1.0, 0.0, 0.0]),
        limit=5,
    )

    assert results == []
    qdrant_client_mock.query_points.assert_not_called()


def test_counts_matching_points(
    vector_repository: QdrantVectorRepository,
    qdrant_client_mock: MagicMock,
) -> None:
    qdrant_client_mock.collection_exists.return_value = True
    qdrant_client_mock.count.return_value = (
        SimpleNamespace(count=3)
    )

    count = vector_repository.count(
        filters={"document_id": "document-1"}
    )

    assert count == 3
    qdrant_client_mock.count.assert_called_once()


def test_delete_document_removes_matching_points(
    vector_repository: QdrantVectorRepository,
    qdrant_client_mock: MagicMock,
) -> None:
    qdrant_client_mock.collection_exists.return_value = True

    vector_repository.delete_document("document-1")

    delete_call = qdrant_client_mock.delete.call_args

    assert (
        delete_call.kwargs["collection_name"]
        == "trilens_test_vectors"
    )
    assert delete_call.kwargs["wait"] is True

    selector = delete_call.kwargs["points_selector"]
    condition = selector.filter.must[0]

    assert condition.key == "document_id"
    assert condition.match.value == "document-1"


def _collection_info() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    "visual": SimpleNamespace(
                        size=3,
                        distance=models.Distance.COSINE,
                    )
                }
            )
        ),
        payload_schema={},
    )