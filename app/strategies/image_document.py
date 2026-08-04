from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

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
                prepared_image = image.convert("RGB").copy()

        except (UnidentifiedImageError, OSError) as error:
            raise InvalidImageDocumentError(
                f"Cannot read image document: {source.filename}"
            ) from error

        return [
            DocumentPage(
                page_number=1,
                image=prepared_image,
            )
        ]