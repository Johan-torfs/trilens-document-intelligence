from io import BytesIO

import pymupdf
import pytest
from PIL import Image

from app.domain.prepared_document import DocumentSource
from app.services.document_preparation_service import (
    DocumentPreparationService,
)
from app.strategies.document_format import (
    UnsupportedDocumentFormatError,
)
from app.strategies.image_document import (
    ImageDocumentStrategy,
)
from app.strategies.pdf_document import PDFDocumentStrategy


def create_png_bytes(width: int = 120, height: int = 80) -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (width, height),
    ).save(buffer, format="PNG")

    return buffer.getvalue()


def create_pdf_bytes(page_count: int = 2) -> bytes:
    document = pymupdf.open()

    try:
        for index in range(page_count):
            page = document.new_page(width=200, height=300)
            page.insert_text((30, 50), f"Page {index + 1}")

        return document.tobytes()

    finally:
        document.close()


def create_service() -> DocumentPreparationService:
    return DocumentPreparationService(
        strategies=[
            ImageDocumentStrategy(),
            PDFDocumentStrategy(dpi=100),
        ]
    )


def test_prepares_image_as_single_page() -> None:
    source = DocumentSource(
        filename="document.png",
        mime_type="image/png",
        content=create_png_bytes(),
    )

    prepared = create_service().prepare(source)

    assert prepared.page_count == 1
    assert prepared.pages[0].page_number == 1
    assert prepared.pages[0].width == 120
    assert prepared.pages[0].height == 80


def test_prepares_multi_page_pdf() -> None:
    source = DocumentSource(
        filename="document.pdf",
        mime_type="application/pdf",
        content=create_pdf_bytes(page_count=3),
    )

    prepared = create_service().prepare(source)

    assert prepared.page_count == 3
    assert [p.page_number for p in prepared.pages] == [1, 2, 3]

    for page in prepared.pages:
        assert page.image.mode == "RGB"
        assert page.width > 0
        assert page.height > 0


def test_converts_image_to_pdf() -> None:
    source = DocumentSource(
        filename="document.png",
        mime_type="image/png",
        content=create_png_bytes(),
    )

    pdf_bytes = create_service().to_pdf(source)

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as document:
        assert document.page_count == 1


def test_pdf_passthrough_returns_original_bytes() -> None:
    original_bytes = create_pdf_bytes(page_count=2)

    source = DocumentSource(
        filename="document.pdf",
        mime_type="application/pdf",
        content=original_bytes,
    )

    result = create_service().to_pdf(source)

    assert result == original_bytes


def test_rejects_unsupported_format() -> None:
    source = DocumentSource(
        filename="document.docx",
        mime_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        content=b"not-a-docx",
    )

    with pytest.raises(
        UnsupportedDocumentFormatError,
        match="Unsupported document format",
    ):
        create_service().prepare(source)


def test_preprocess_pages_returns_one_result_per_page() -> None:
    source = DocumentSource(
        filename="document.pdf",
        mime_type="application/pdf",
        content=create_pdf_bytes(page_count=3),
    )

    service = create_service()
    prepared = service.prepare(source)
    results = service.preprocess_pages(prepared)

    assert len(results) == 3

    for result in results:
        assert result.image.mode == "RGB"
        assert result.image.width > 0
        assert result.image.height > 0


def test_preprocess_pages_works_with_in_memory_image() -> None:
    source = DocumentSource(
        filename="document.png",
        mime_type="image/png",
        content=create_png_bytes(width=300, height=200),
    )

    service = create_service()
    prepared = service.prepare(source)
    results = service.preprocess_pages(prepared)

    assert len(results) == 1
    assert results[0].image.mode == "RGB"
    assert results[0].original_size == (300, 200)