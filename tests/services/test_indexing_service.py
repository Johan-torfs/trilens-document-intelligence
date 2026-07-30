from pathlib import Path
from unittest.mock import create_autospec
from uuid import UUID

import numpy as np
from PIL import Image

from app.domain.document import ArtifactType, ModelArtifact
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.indexing_service import IndexingService
from app.strategies.retrieval import RetrievalStrategy


class FakeRetrievalStrategy(RetrievalStrategy):
    @property
    def model_name(self) -> str:
        return "fake-model"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return np.array([0.6, 0.8], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_index_image_stores_vector_and_returns_artifact(
    vector_repository: VectorRepository,
) -> None:
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    service = IndexingService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=vector_repository,
        document_repository=document_repository,
    )

    artifact = service.index_image(
        document_id="document-001",
        image=Image.new("RGB", (100, 100), "white"),
    )

    stored_embedding = vector_repository.load(
        artifact_id=artifact.id,
    )

    assert isinstance(artifact, ModelArtifact)
    assert artifact.document_id == "document-001"
    assert artifact.artifact_type == ArtifactType.IMAGE_EMBEDDING
    assert artifact.model_name == "fake-model"
    assert artifact.dimensions == 2
    assert artifact.storage_path is not None
    assert Path(artifact.storage_path).exists()

    np.testing.assert_array_equal(
        stored_embedding,
        np.array([0.6, 0.8], dtype=np.float32),
    )

    document_repository.save_artifact.assert_called_once_with(
        artifact
    )


def test_index_image_generates_valid_artifact_id(
    vector_repository: VectorRepository,
) -> None:
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    service = IndexingService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=vector_repository,
        document_repository=document_repository,
    )

    artifact = service.index_image(
        document_id="document-001",
        image=Image.new("RGB", (100, 100), "white"),
    )

    parsed_id = UUID(artifact.id)

    assert str(parsed_id) == artifact.id