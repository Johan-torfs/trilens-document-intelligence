from abc import ABC, abstractmethod

from app.domain.document import (
    DocumentRecord,
    ProcessingStatus,
)
from app.domain.ocr import OCRResult


class DocumentRepository(ABC):
    @abstractmethod
    def save_document(
        self,
        document: DocumentRecord,
    ) -> None:
        ...

    @abstractmethod
    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        ...

    @abstractmethod
    def get_document_by_checksum(
        self,
        checksum: str,
    ) -> DocumentRecord | None:
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
    def update_ocr(
        self,
        document_id: str,
        result: OCRResult | None,
    ) -> None:
        ...

    @abstractmethod
    def update_document_type(
        self,
        document_id: str,
        document_type: str,
    ) -> None:
        ...

    @abstractmethod
    def lexical_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        ...