from collections.abc import Sequence
from unittest.mock import MagicMock, create_autospec

import numpy as np
import pytest
from PIL import Image

from app.domain.document import DocumentRecord
from app.domain.prepared_document import DocumentPage
from app.repositories.vector_repository import VectorRepository
from app.services.indexing_service import (
    DocumentIndexingError,
    IndexingResult,
    IndexingService,
)
from app.strategies.embedding import EmbeddingStrategy


class FakeEmbeddingStrategy(EmbeddingStrategy):
    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def model_version(self) -> str | None:
        return "1.0"

    def embed_images(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        return np.tile(
            np.array([0.6, 0.8], dtype=np.float32),
            (len(images), 1),
        )

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def make_document(
    document_id: str = "document-001",
    page_count: int = 1,
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        original_filename="invoice.png",
        stored_path="data/invoice.png",
        checksum="checksum-1",
        width=100,
        height=100,
        page_count=page_count,
        mime_type="image/png",
        document_type="invoice",
    )


def make_pages(
    count: int = 1,
) -> list[DocumentPage]:
    return [
        DocumentPage(
            page_number=i,
            image=Image.new("RGB", (100, 100), "white"),
        )
        for i in range(1, count + 1)
    ]


def test_index_pages_returns_indexing_result(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.side_effect = [0, 1]

    service = IndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    document = make_document()
    pages = make_pages(1)

    result = service.index_pages(
        document=document,
        pages=pages,
    )

    assert isinstance(result, IndexingResult)
    assert result.document_id == "document-001"
    assert result.page_count == 1
    assert result.dimensions == 2
    assert result.model_name == "fake-model"
    assert result.reused_existing is False

    mock_vector_repository.save_batch.assert_called_once()


def test_index_pages_multi_page(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.side_effect = [0, 3]

    service = IndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    document = make_document(page_count=3)
    pages = make_pages(3)

    result = service.index_pages(
        document=document,
        pages=pages,
    )

    assert result.page_count == 3
    saved_points = mock_vector_repository.save_batch.call_args.args[0]
    assert len(saved_points) == 3
    assert [p.payload["page_number"] for p in saved_points] == [1, 2, 3]


def test_index_pages_returns_cached_when_already_indexed(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.return_value = 1

    service = IndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    document = make_document()
    pages = make_pages(1)

    result = service.index_pages(
        document=document,
        pages=pages,
    )

    assert result.reused_existing is True
    mock_vector_repository.save_batch.assert_not_called()


def test_index_pages_rejects_empty_pages(
    mock_vector_repository: MagicMock,
) -> None:
    service = IndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    with pytest.raises(ValueError, match="At least one"):
        service.index_pages(
            document=make_document(),
            pages=[],
        )


def test_index_pages_rejects_mismatched_page_count(
    mock_vector_repository: MagicMock,
) -> None:
    service = IndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    with pytest.raises(ValueError, match="expects 2 pages"):
        service.index_pages(
            document=make_document(page_count=2),
            pages=make_pages(1),
        )