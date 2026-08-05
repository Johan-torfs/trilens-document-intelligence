from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from app.domain.document import DocumentRecord
from app.domain.prepared_document import DocumentPage
from app.repositories.vector_repository import VectorRepository
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
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
        results = []

        for image in images:
            first_pixel = image.getpixel((0, 0))

            if first_pixel == (255, 255, 255):
                results.append([1.0, 0.0])
            else:
                results.append([0.0, 1.0])

        return np.array(results, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        if "invoice" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)


def make_document(
    document_id: str,
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        original_filename=f"{document_id}.png",
        stored_path=f"data/{document_id}.png",
        checksum=document_id,
        width=100,
        height=100,
        page_count=1,
        mime_type="image/png",
        document_type="invoice",
    )


def test_indexed_documents_produce_correct_vector_payload(
    mock_vector_repository: MagicMock,
) -> None:
    mock_vector_repository.count.side_effect = [0, 1, 0, 1]

    indexing_service = IndexingService(
        strategy=FakeEmbeddingStrategy(),
        vector_repository=mock_vector_repository,
    )

    invoice_doc = make_document("invoice-document")
    landscape_doc = make_document("landscape-document")

    invoice_page = DocumentPage(
        page_number=1,
        image=Image.new("RGB", (100, 100), "white"),
    )
    landscape_page = DocumentPage(
        page_number=1,
        image=Image.new("RGB", (100, 100), "black"),
    )

    indexing_service.index_pages(
        document=invoice_doc,
        pages=[invoice_page],
    )
    indexing_service.index_pages(
        document=landscape_doc,
        pages=[landscape_page],
    )

    calls = mock_vector_repository.save_batch.call_args_list
    assert len(calls) == 2

    invoice_points = calls[0].args[0]
    assert invoice_points[0].payload["document_id"] == "invoice-document"
    assert invoice_points[0].payload["page_number"] == 1

    landscape_points = calls[1].args[0]
    assert landscape_points[0].payload["document_id"] == "landscape-document"