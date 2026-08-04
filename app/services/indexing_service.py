from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from app.domain.document import ArtifactType, ModelArtifact
from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.strategies.retrieval import RetrievalStrategy


class IndexingService:
    def __init__(
        self,
        strategy: RetrievalStrategy,
        vector_repository: VectorRepository,
        document_repository: DocumentRepository,
    ) -> None:
        self._strategy = strategy
        self._vector_repository = vector_repository
        self._document_repository = document_repository

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    def index_image(
        self,
        document_id: str,
        image: Image.Image,
    ) -> ModelArtifact:
        embedding = self._strategy.embed_image(image)

        artifact_id = str(uuid4())

        storage_path = self._vector_repository.save(
            artifact_id=artifact_id,
            vector=embedding,
        )

        artifact = ModelArtifact(
            id=artifact_id,
            document_id=document_id,
            artifact_type=ArtifactType.IMAGE_EMBEDDING,
            model_name=self._strategy.model_name,
            storage_path=str(storage_path),
            dimensions=len(embedding),
        )

        self._document_repository.save_artifact(artifact)

        return artifact

    def index_pages(
        self,
        document_id: str,
        images: list[Image.Image],
    ) -> ModelArtifact:
        if len(images) == 1:
            return self.index_image(document_id, images[0])

        embeddings = [
            self._strategy.embed_image(img) for img in images
        ]

        normalized = []
        for e in embeddings:
            norm = np.linalg.norm(e)
            normalized.append(e / norm if norm > 0.0 else e)

        combined = np.mean(normalized, axis=0)
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 0.0:
            combined = combined / combined_norm

        artifact_id = str(uuid4())

        storage_path = self._vector_repository.save(
            artifact_id=artifact_id,
            vector=combined,
        )

        artifact = ModelArtifact(
            id=artifact_id,
            document_id=document_id,
            artifact_type=ArtifactType.IMAGE_EMBEDDING,
            model_name=self._strategy.model_name,
            storage_path=str(storage_path),
            dimensions=len(combined),
        )

        self._document_repository.save_artifact(artifact)

        return artifact
