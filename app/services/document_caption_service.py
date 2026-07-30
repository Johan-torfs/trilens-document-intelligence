import logging
from uuid import uuid4

from PIL import Image

from app.domain.document import ArtifactType, ModelArtifact
from app.repositories.document_repository import DocumentRepository
from app.services.caption_service import CaptionService

logger = logging.getLogger(__name__)


class DocumentCaptionService:
    def __init__(
        self,
        caption_service: CaptionService,
        document_repository: DocumentRepository,
    ) -> None:
        self._caption_service = caption_service
        self._document_repository = document_repository

    @property
    def model_name(self) -> str:
        return self._caption_service.model_name

    @property
    def model_version(self) -> str | None:
        return self._caption_service.model_version

    def caption_document(
        self,
        document_id: str,
        image: Image.Image,
    ) -> ModelArtifact:
        cached_artifact = self._find_cached_caption(document_id)

        if cached_artifact is not None:
            return cached_artifact

        result = self._caption_service.generate(image)

        artifact = ModelArtifact(
            id=str(uuid4()),
            document_id=document_id,
            artifact_type=ArtifactType.CAPTION,
            model_name=result.model_name,
            model_version=result.model_version,
            content=result.caption,
        )

        self._document_repository.save_artifact(artifact)

        logger.info(
            "Caption generated for document %s with model %s "
            "in %.2f ms",
            document_id,
            result.model_name,
            result.duration_ms,
        )

        return artifact

    def _find_cached_caption(
        self,
        document_id: str,
    ) -> ModelArtifact | None:
        artifacts = self._document_repository.get_artifacts(
            document_id
        )

        for artifact in artifacts:
            if (
                artifact.artifact_type == ArtifactType.CAPTION
                and artifact.model_name
                == self._caption_service.model_name
                and artifact.model_version
                == self._caption_service.model_version
                and artifact.content
            ):
                return artifact

        return None