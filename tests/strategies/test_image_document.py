from io import BytesIO

import pytest
from PIL import Image

from app.domain.prepared_document import DocumentSource
from app.strategies.image_document import (
    ImageDocumentStrategy,
    InvalidImageDocumentError,
)


def create_png_bytes() -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (120, 80),
    ).save(buffer, format="PNG")

    return buffer.getvalue()


def create_tiff_bytes() -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (60, 40),
    ).save(buffer, format="TIFF")

    return buffer.getvalue()


def create_webp_bytes() -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (50, 30),
    ).save(buffer, format="WEBP")

    return buffer.getvalue()


def test_supports_png_image() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="document.png",
        mime_type="image/png",
        content=create_png_bytes(),
    )

    assert strategy.supports(source) is True


def test_supports_tiff_image() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="document.tiff",
        mime_type="image/tiff",
        content=create_tiff_bytes(),
    )

    assert strategy.supports(source) is True


def test_supports_webp_image() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="document.webp",
        mime_type="image/webp",
        content=create_webp_bytes(),
    )

    assert strategy.supports(source) is True


def test_extracts_single_rgb_page() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="document.png",
        mime_type="image/png",
        content=create_png_bytes(),
    )

    pages = strategy.extract_pages(source)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image.mode == "RGB"
    assert pages[0].width == 120
    assert pages[0].height == 80


def test_extracts_tiff_as_single_page() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="document.tiff",
        mime_type="image/tiff",
        content=create_tiff_bytes(),
    )

    pages = strategy.extract_pages(source)

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].image.mode == "RGB"


def test_rejects_invalid_image_bytes() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="corrupt.png",
        mime_type="image/png",
        content=b"this-is-not-an-image",
    )

    with pytest.raises(InvalidImageDocumentError):
        strategy.extract_pages(source)


def test_does_not_support_pdf() -> None:
    strategy = ImageDocumentStrategy()

    source = DocumentSource(
        filename="document.pdf",
        mime_type="application/pdf",
        content=b"%PDF-1.4",
    )

    assert strategy.supports(source) is False