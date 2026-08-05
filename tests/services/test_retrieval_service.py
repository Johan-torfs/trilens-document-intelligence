from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from app.repositories.vector_repository import VectorRepository
from app.services.retrieval_service import RetrievalService
from app.strategies.embedding import EmbeddingStrategy


class FakeRetrievalStrategy(EmbeddingStrategy):
    @property
    def model_name(self) -> str:
        return "fake-retrieval-model"

    @property
    def model_version(self) -> str | None:
        return "1.0"

    def embed_images(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        return np.tile(
            np.array([1.0, 0.0], dtype=np.float32),
            (len(images), 1),
        )

    def embed_text(self, text: str) -> np.ndarray:
        if "invoice" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)


def test_search_returns_document_matches(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.search.return_value = [
        SimpleNamespace(
            point_id="point-1",
            score=0.95,
            payload={
                "document_id": "invoice-doc",
                "page_number": 1,
            },
        ),
    ]

    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    results = service.search(query="an invoice", top_k=5)

    assert len(results) == 1
    assert results[0].document_id == "invoice-doc"
    assert results[0].score == pytest.approx(0.95)
    assert results[0].best_page_number == 1


def test_search_groups_pages_by_document(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.search.return_value = [
        SimpleNamespace(
            point_id="p1",
            score=0.90,
            payload={"document_id": "doc-1", "page_number": 1},
        ),
        SimpleNamespace(
            point_id="p2",
            score=0.80,
            payload={"document_id": "doc-1", "page_number": 2},
        ),
        SimpleNamespace(
            point_id="p3",
            score=0.70,
            payload={"document_id": "doc-2", "page_number": 1},
        ),
    ]

    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    results = service.search(query="an invoice", top_k=10)

    assert len(results) == 2
    assert results[0].document_id == "doc-1"
    assert results[0].score == pytest.approx(0.90)
    assert results[1].document_id == "doc-2"


def test_search_rejects_empty_query(
    mock_vector_repository: MagicMock,
) -> None:
    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    with pytest.raises(
        ValueError,
        match="zoekopdracht mag niet leeg zijn",
    ):
        service.search(query="   ", top_k=5)


def test_service_exposes_strategy_model_name(
    mock_vector_repository: MagicMock,
) -> None:
    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    assert service.model_name == "fake-retrieval-model"


def test_text_similarity_compares_clip_text_embeddings(
    mock_vector_repository: MagicMock,
) -> None:
    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    same_topic_score = service.text_similarity(
        "invoice with product rows",
        "an invoice document",
    )

    different_topic_score = service.text_similarity(
        "invoice with product rows",
        "a landscape photograph",
    )

    assert same_topic_score == 1.0
    assert different_topic_score == 0.0


def test_text_similarity_returns_zero_for_empty_text(
    mock_vector_repository: MagicMock,
) -> None:
    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    assert service.text_similarity("", "an invoice") == 0.0
    assert service.text_similarity("invoice", "   ") == 0.0


def test_search_text_groups_chunks_by_document(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.search.return_value = [
        SimpleNamespace(
            point_id="chunk-1",
            score=0.88,
            payload={
                "document_id": "doc-1",
                "page_number": 1,
            },
        ),
        SimpleNamespace(
            point_id="chunk-2",
            score=0.75,
            payload={
                "document_id": "doc-1",
                "page_number": 2,
            },
        ),
        SimpleNamespace(
            point_id="chunk-3",
            score=0.60,
            payload={
                "document_id": "doc-2",
                "page_number": 1,
            },
        ),
    ]

    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    results = service.search_text(query="invoice", top_k=5)

    assert len(results) == 2
    assert results[0].document_id == "doc-1"
    assert results[0].score == pytest.approx(0.88)
    assert results[1].document_id == "doc-2"

    query_call = mock_vector_repository.search.call_args
    assert query_call.kwargs["vector_name"] == "text"


def test_search_text_uses_text_unit_type_filter(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.search.return_value = []

    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=mock_vector_repository,
    )

    service.search_text(query="invoice", top_k=5)

    filters = mock_vector_repository.search.call_args.kwargs["filters"]
    assert filters["unit_type"] == "chunk"
    assert filters["vector_type"] == "text"