from abc import ABC, abstractmethod

from app.domain.document import ArtifactType, DocumentRecord, ModelArtifact, ProcessingStatus


class DocumentRepository(ABC):
    @abstractmethod
    def save_document(self, document: DocumentRecord) -> None:
        ...

    @abstractmethod
    def get_document(self, document_id: str) -> DocumentRecord | None:
        ...

    @abstractmethod
    def get_document_by_checksum(
        self,
        checksum: str,
    ) -> DocumentRecord | None:
        ...

    @abstractmethod
    def save_artifact(self, artifact: ModelArtifact) -> None:
        ...

    @abstractmethod
    def get_artifacts(
        self,
        document_id: str,
    ) -> list[ModelArtifact]:
        ...

    @abstractmethod
    def update_processing_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> None:
        ...

    @abstractmethod
    def find_artifacts(
        self,
        artifact_type: ArtifactType,
        model_name: str,
        document_type: str | None = None,
    ) -> list[ModelArtifact]:
        ...