from unittest.mock import create_autospec

import numpy as np
from PIL import Image

from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
from app.strategies.retrieval import RetrievalStrategy


class FakeRetrievalStrategy(RetrievalStrategy):
    @property
    def model_name(self) -> str:
        return "fake-model"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        first_pixel = image.getpixel((0, 0))

        if first_pixel == (255, 255, 255):
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)

    @staticmethod
    def calibrate_score(raw: float) -> float:
        return raw

    def embed_text(self, text: str) -> np.ndarray:
        if "invoice" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)


def test_indexed_documents_can_be_retrieved(
    vector_repository: VectorRepository,
) -> None:
    strategy = FakeRetrievalStrategy()

    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    indexing_service = IndexingService(
        strategy=strategy,
        vector_repository=vector_repository,
        document_repository=document_repository,
    )

    retrieval_service = RetrievalService(
        strategy=strategy,
        vector_repository=vector_repository,
    )

    invoice_artifact = indexing_service.index_image(
        document_id="invoice-document",
        image=Image.new("RGB", (100, 100), "white"),
    )

    landscape_artifact = indexing_service.index_image(
        document_id="landscape-document",
        image=Image.new("RGB", (100, 100), "black"),
    )

    results = retrieval_service.search(
        query="an invoice",
        artifact_ids=[
            landscape_artifact.id,
            invoice_artifact.id,
        ],
    )

    assert results[0][0] == invoice_artifact.id
    assert results[0][1] == 1.0

    assert document_repository.save_artifact.call_count == 2