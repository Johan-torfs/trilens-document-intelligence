from abc import ABC, abstractmethod

from app.domain.prepared_document import (
    DocumentPage,
    DocumentSource,
)


class UnsupportedDocumentFormatError(ValueError):
    pass


class DocumentFormatStrategy(ABC):
    @abstractmethod
    def supports(self, source: DocumentSource) -> bool:
        """Return whether this strategy can read the source."""
        ...

    @abstractmethod
    def extract_pages(
        self,
        source: DocumentSource,
    ) -> list[DocumentPage]:
        """Convert the source into ordered RGB page images."""
        ...