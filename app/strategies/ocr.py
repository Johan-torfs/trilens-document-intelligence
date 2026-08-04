from abc import ABC, abstractmethod

from app.domain.ocr import OCRResult
from app.domain.prepared_document import DocumentPage


class OCRStrategy(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the OCR model."""
        ...

    @property
    def model_version(self) -> str | None:
        """Return the version of the OCR model, if applicable."""
        return None

    @abstractmethod
    def extract(self, pages: list[DocumentPage]) -> OCRResult:
        """Extract text from ordered document pages and return an OCRResult."""
        ...