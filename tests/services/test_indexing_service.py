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

    @staticmethod
    def calibrate_score(raw: float) -> float:
        return raw


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


def test_index_pages_single_image_delegates_to_index_image(
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

    images = [Image.new("RGB", (100, 100), "white")]
    artifact = service.index_pages(
        document_id="document-001",
        images=images,
    )

    stored_embedding = vector_repository.load(artifact.id)

    assert artifact.artifact_type == ArtifactType.IMAGE_EMBEDDING
    np.testing.assert_array_equal(
        stored_embedding,
        np.array([0.6, 0.8], dtype=np.float32),
    )


def test_index_pages_multi_page_combines_normalized_embeddings(
    vector_repository: VectorRepository,
) -> None:
    call_count = 0

    class TwoVectorStrategy(RetrievalStrategy):
        """Returns different embeddings per call to simulate two pages."""

        @property
        def model_name(self) -> str:
            return "fake-model"

        def embed_image(
            self, image: Image.Image
        ) -> np.ndarray:
            nonlocal call_count
            call_count += 1

            # page 1: [3, 4] -> normalized [0.6, 0.8]
            # page 2: [0, 1] -> normalized [0, 1]
            if call_count == 1:
                return np.array([3.0, 4.0], dtype=np.float32)
            return np.array([0.0, 1.0], dtype=np.float32)

        def embed_text(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

        @staticmethod
        def calibrate_score(raw: float) -> float:
            return raw

    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    service = IndexingService(
        strategy=TwoVectorStrategy(),
        vector_repository=vector_repository,
        document_repository=document_repository,
    )

    images = [
        Image.new("RGB", (100, 100), "white"),
        Image.new("RGB", (100, 100), "gray"),
    ]

    artifact = service.index_pages(
        document_id="document-001",
        images=images,
    )

    stored = vector_repository.load(artifact.id)

    # Mean of [0.6, 0.8] and [0, 1] -> [0.3, 0.9], then normalized
    expected_raw = np.array([0.3, 0.9], dtype=np.float32)
    expected_norm = expected_raw / np.linalg.norm(expected_raw)

    assert artifact.dimensions == 2
    np.testing.assert_array_almost_equal(stored, expected_norm)