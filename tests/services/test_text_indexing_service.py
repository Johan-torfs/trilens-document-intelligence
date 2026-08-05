from unittest.mock import MagicMock

import numpy as np
import pytest

from app.domain.document import DocumentRecord
from app.domain.ocr import OCRPageResult, OCRResult
from app.services.text_indexing_service import (
    TextIndexingError,
    TextIndexingResult,
    TextIndexingService,
)


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


def make_ocr_result(text: str = "Invoice number 1234\n\nTotal amount 99.95") -> OCRResult:
    return OCRResult(
        text=text,
        pages=[
            OCRPageResult(
                page_number=1,
                text=text,
                words=[],
                mean_confidence=0.9,
            )
        ],
        mean_confidence=0.9,
        model_name="doctr",
        model_version="1",
    )


class FakeEmbeddingStrategy:
    model_name = "fake-siglip"
    model_version = "1.0"

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.6, 0.8], dtype=np.float32)


def test_index_document_text_stores_one_point_per_chunk(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.return_value = 0

    service = TextIndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    document = make_document()
    ocr = make_ocr_result()

    result = service.index_document_text(document, ocr)

    assert isinstance(result, TextIndexingResult)
    assert result.document_id == "document-001"
    assert result.chunk_count == 2
    assert result.model_name == "fake-siglip"
    assert result.reused_existing is False

    saved_points = mock_vector_repository.save_batch.call_args.args[0]
    assert len(saved_points) == 2
    assert all(p.vector_name == "text" for p in saved_points)
    assert saved_points[0].payload["page_number"] == 1
    assert saved_points[0].payload["chunk_number"] == 0
    assert saved_points[1].payload["chunk_number"] == 1


def test_index_document_text_returns_cached_when_already_indexed(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.return_value = 3

    service = TextIndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    result = service.index_document_text(
        make_document(),
        make_ocr_result(),
    )

    assert result.reused_existing is True
    assert result.chunk_count == 3
    mock_vector_repository.save_batch.assert_not_called()


def test_index_document_text_raises_when_no_chunks_produced(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.return_value = 0

    service = TextIndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    empty_ocr = make_ocr_result("")

    with pytest.raises(TextIndexingError, match="No text chunks"):
        service.index_document_text(make_document(), empty_ocr)


def test_point_ids_are_deterministic(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.return_value = 0

    service = TextIndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    service.index_document_text(make_document(), make_ocr_result())
    first_call_ids = [
        p.id
        for p in mock_vector_repository.save_batch.call_args.args[0]
    ]

    mock_vector_repository.reset_mock()
    mock_vector_repository.count.return_value = 0

    service.index_document_text(make_document(), make_ocr_result())
    second_call_ids = [
        p.id
        for p in mock_vector_repository.save_batch.call_args.args[0]
    ]

    assert first_call_ids == second_call_ids
