from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class DocumentSource:
    filename: str
    mime_type: str
    content: bytes


@dataclass(frozen=True)
class DocumentPage:
    page_number: int
    image: Image.Image

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass(frozen=True)
class PreparedDocument:
    source: DocumentSource
    pages: list[DocumentPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)