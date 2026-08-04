from unittest.mock import create_autospec

import numpy as np
from PIL import Image

from app.domain.document import (
    ArtifactType,
    DocumentMetadata,
    DocumentRecord,
    ModelArtifact,
)
from app.domain.search import SearchQuery
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.document_search_service import DocumentSearchService
from app.services.retrieval_service import RetrievalService
from app.strategies.retrieval import RetrievalStrategy


class FakeRetrievalStrategy(RetrievalStrategy):
    @property
    def model_name(self) -> str:
        return "fake-clip-model"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    @staticmethod
    def calibrate_score(raw: float) -> float:
        return raw


def test_search_returns_enriched_document_result(
    vector_repository: VectorRepository,
) -> None:
    embedding_artifact = ModelArtifact(
        id="invoice-embedding",
        document_id="document-001",
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="fake-clip-model",
        storage_path="invoice-embedding.npy",
        dimensions=2,
    )

    caption_artifact = ModelArtifact(
        id="invoice-caption",
        document_id="document-001",
        artifact_type=ArtifactType.CAPTION,
        model_name="fake-caption-model",
        content="an automatically generated invoice caption",
    )

    document = DocumentRecord(
        id="document-001",
        original_filename="invoice.png",
        stored_path="data/documents/invoice.png",
        checksum="test-checksum",
        width=100,
        height=100,
        mime_type="image/png",
        document_type="invoice",
        metadata=DocumentMetadata(
            document_type="invoice",
        ),
    )

    vector_repository.save(
        artifact_id=embedding_artifact.id,
        vector=np.array([1.0, 0.0], dtype=np.float32),
    )

    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    document_repository.find_artifacts.return_value = [
        embedding_artifact
    ]
    document_repository.get_document.return_value = document
    document_repository.get_artifacts.return_value = [
        embedding_artifact,
        caption_artifact,
    ]

    retrieval_service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=vector_repository,
    )

    search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    results = search_service.search(
        SearchQuery(
            text="an invoice",
            top_k=5,
        )
    )

    assert len(results) == 1

    result = results[0]

    assert result.document_id == "document-001"
    assert result.score == 1.0
    assert result.rank == 1
    assert (
        result.caption
        == "an automatically generated invoice caption"
    )
    assert result.stored_path == "data/documents/invoice.png"
    assert result.document_type == "invoice"

    document_repository.find_artifacts.assert_called_once_with(
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="fake-clip-model",
        document_type=None,
    )


def test_search_passes_document_type_filter_to_repository(
    vector_repository: VectorRepository,
) -> None:
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    document_repository.find_artifacts.return_value = []

    retrieval_service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=vector_repository,
    )

    search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    results = search_service.search(
        SearchQuery(
            text="a receipt",
            top_k=5,
            document_type="receipt",
        )
    )

    assert results == []

    document_repository.find_artifacts.assert_called_once_with(
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="fake-clip-model",
        document_type="receipt",
    )