from io import BytesIO
from pathlib import Path

from PIL import Image, ImageSequence, UnidentifiedImageError

from app.domain.prepared_document import (
    DocumentPage,
    DocumentSource,
)
from app.strategies.document_format import (
    DocumentFormatStrategy,
)


class InvalidImageDocumentError(ValueError):
    pass


class ImageDocumentStrategy(DocumentFormatStrategy):
    MIME_TYPES = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/tiff",
    }

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff",
    }

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
            with Image.open(BytesIO(source.content)) as image:
                pages = [
                    DocumentPage(
                        page_number=index,
                        image=frame.convert("RGB").copy(),
                    )
                    for index, frame in enumerate(
                        ImageSequence.Iterator(image),
                        start=1,
                    )
                ]

        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageDocumentError(
                f"Cannot read image document: {source.filename}"
            ) from error

        if not pages:
            raise InvalidImageDocumentError(
                f"Image document contains no pages: "
                f"{source.filename}"
            )

        return pages