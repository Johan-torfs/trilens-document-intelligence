from uuid import uuid4

from app.domain.document import (
    ArtifactType,
    ModelArtifact,
)
from app.domain.ocr import OCRResult
from app.domain.prepared_document import DocumentPage
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.strategies.ocr import OCRStrategy


class OCRProcessingError(RuntimeError):
    pass


class DocumentOCRService:
    def __init__(
        self,
        strategy: OCRStrategy,
        document_repository: DocumentRepository,
    ) -> None:
        self._strategy = strategy
        self._document_repository = document_repository

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    @property
    def model_version(self) -> str | None:
        return self._strategy.model_version

    def process_document(
        self,
        document_id: str,
        pages: list[DocumentPage],
    ) -> ModelArtifact:
        cached_artifact = self._find_cached_artifact(
            document_id
        )

        if cached_artifact is not None:
            return cached_artifact

        try:
            result = self._strategy.extract(pages)
        except Exception as error:
            raise OCRProcessingError(
                f"OCR failed for document {document_id}: "
                f"{error}"
            ) from error

        artifact = self._create_artifact(
            document_id=document_id,
            result=result,
        )

        self._document_repository.save_artifact(artifact)

        return artifact

    def _find_cached_artifact(
        self,
        document_id: str,
    ) -> ModelArtifact | None:
        artifacts = self._document_repository.get_artifacts(
            document_id
        )

        for artifact in reversed(artifacts):
            if (
                artifact.artifact_type == ArtifactType.OCR
                and artifact.model_name == self.model_name
                and artifact.model_version
                == self.model_version
            ):
                return artifact

        return None

    def _create_artifact(
        self,
        document_id: str,
        result: OCRResult,
    ) -> ModelArtifact:
        return ModelArtifact(
            id=str(uuid4()),
            document_id=document_id,
            artifact_type=ArtifactType.OCR,
            model_name=self.model_name,
            model_version=self.model_version,
            content=result.model_dump_json(),
        )