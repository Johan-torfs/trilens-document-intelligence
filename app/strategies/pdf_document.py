from pathlib import Path

import pymupdf
from PIL import Image

from app.domain.prepared_document import (
    DocumentPage,
    DocumentSource,
)
from app.strategies.document_format import (
    DocumentFormatStrategy,
)


class InvalidPDFDocumentError(ValueError):
    pass


class PDFDocumentStrategy(DocumentFormatStrategy):
    MIME_TYPES = {"application/pdf"}
    EXTENSIONS = {".pdf"}

    def __init__(self, dpi: int = 200) -> None:
        if dpi <= 0:
            raise ValueError("DPI must be greater than zero.")

        self._dpi = dpi

    def supports(self, source: DocumentSource) -> bool:
        extension = Path(source.filename).suffix.lower()

        return (
            source.mime_type.lower() in self.MIME_TYPES
            or extension in self.EXTENSIONS
        )

    def extract_pages(
        self,
        source: DocumentSource,
    ) -> list[DocumentPage]:
        try:
            with pymupdf.open(
                stream=source.content,
                filetype="pdf",
            ) as document:
                if document.needs_pass:
                    raise InvalidPDFDocumentError(
                        "Password-protected PDFs are not supported."
                    )

                if document.page_count == 0:
                    raise InvalidPDFDocumentError(
                        "PDF contains no pages."
                    )

                pages: list[DocumentPage] = []

                for index, page in enumerate(document, start=1):
                    pixmap = page.get_pixmap(
                        dpi=self._dpi,
                        colorspace=pymupdf.csRGB,
                        alpha=False,
                    )

                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )

                    pages.append(
                        DocumentPage(
                            page_number=index,
                            image=image,
                        )
                    )

                return pages

        except InvalidPDFDocumentError:
            raise

        except (
            pymupdf.FileDataError,
            RuntimeError,
            ValueError,
        ) as error:
            raise InvalidPDFDocumentError(
                f"Cannot read PDF document: {source.filename}"
            ) from error