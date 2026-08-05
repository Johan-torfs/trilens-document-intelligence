from collections.abc import Sequence

from app.domain.document import DocumentRecord
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

    def current_result(
        self,
        document: DocumentRecord,
    ) -> OCRResult | None:
        result = document.ocr

        if result is None:
            return None

        if result.model_name != self.model_name:
            return None

        if result.model_version != self.model_version:
            return None

        return result

    def process_document(
        self,
        document: DocumentRecord,
        pages: Sequence[DocumentPage],
    ) -> OCRResult:
        cached_result = self.current_result(document)

        if cached_result is not None:
            return cached_result

        if not pages:
            raise ValueError(
                "At least one page is required for OCR."
            )

        try:
            result = self._strategy.extract(pages)

            if result.model_name != self.model_name:
                result = result.model_copy(
                    update={"model_name": self.model_name}
                )

            if result.model_version != self.model_version:
                result = result.model_copy(
                    update={
                        "model_version": self.model_version
                    }
                )

            self._document_repository.update_ocr(
                document_id=document.id,
                result=result,
            )

            return result

        except Exception as error:
            raise OCRProcessingError(
                f"OCR failed for document '{document.id}': "
                f"{error}"
            ) from error