from io import BytesIO

import pymupdf

from app.domain.prepared_document import (
    DocumentSource,
    PreparedDocument,
)
from app.preprocessing.pipeline import (
    PreprocessingResult,
    preprocess_pil_image,
)
from app.strategies.document_format import (
    DocumentFormatStrategy,
    UnsupportedDocumentFormatError,
)


class DocumentPreparationService:
    def __init__(
        self,
        strategies: list[DocumentFormatStrategy],
    ) -> None:
        if not strategies:
            raise ValueError(
                "At least one document format strategy is required."
            )

        self._strategies = strategies

    def prepare(
        self,
        source: DocumentSource,
    ) -> PreparedDocument:
        strategy = self._find_strategy(source)
        pages = strategy.extract_pages(source)

        if not pages:
            raise ValueError(
                f"Document contains no readable pages: "
                f"{source.filename}"
            )

        return PreparedDocument(
            source=source,
            pages=pages,
        )

    def preprocess_pages(
        self,
        prepared: PreparedDocument,
    ) -> list[PreprocessingResult]:
        return [
            preprocess_pil_image(page.image)
            for page in prepared.pages
        ]

    def to_pdf(
        self,
        source: DocumentSource,
    ) -> bytes:
        # Preserve an existing PDF instead of rasterizing it again.
        if (
            source.mime_type.lower() == "application/pdf"
            or source.filename.lower().endswith(".pdf")
        ):
            return source.content

        prepared = self.prepare(source)
        document = pymupdf.open()

        try:
            for page in prepared.pages:
                image_buffer = BytesIO()
                page.image.convert("RGB").save(
                    image_buffer,
                    format="PNG",
                )

                pdf_page = document.new_page(
                    width=page.width,
                    height=page.height,
                )

                pdf_page.insert_image(
                    pdf_page.rect,
                    stream=image_buffer.getvalue(),
                )

            return document.tobytes()

        finally:
            document.close()

    def _find_strategy(
        self,
        source: DocumentSource,
    ) -> DocumentFormatStrategy:
        for strategy in self._strategies:
            if strategy.supports(source):
                return strategy

        raise UnsupportedDocumentFormatError(
            f"Unsupported document format: "
            f"{source.filename} ({source.mime_type})"
        )