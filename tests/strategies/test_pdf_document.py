import pymupdf
import pytest

from app.domain.prepared_document import DocumentSource
from app.strategies.pdf_document import (
    InvalidPDFDocumentError,
    PDFDocumentStrategy,
)


def create_pdf_bytes(page_count: int = 2) -> bytes:
    document = pymupdf.open()

    try:
        for index in range(page_count):
            page = document.new_page(
                width=200,
                height=300,
            )
            page.insert_text(
                (30, 50),
                f"Page {index + 1}",
            )

        return document.tobytes()

    finally:
        document.close()


def test_supports_pdf() -> None:
    strategy = PDFDocumentStrategy()

    source = DocumentSource(
        filename="document.pdf",
        mime_type="application/pdf",
        content=create_pdf_bytes(),
    )

    assert strategy.supports(source) is True


def test_extracts_all_pdf_pages_in_order() -> None:
    strategy = PDFDocumentStrategy(dpi=100)

    source = DocumentSource(
        filename="document.pdf",
        mime_type="application/pdf",
        content=create_pdf_bytes(page_count=2),
    )

    pages = strategy.extract_pages(source)

    assert len(pages) == 2
    assert [page.page_number for page in pages] == [1, 2]

    for page in pages:
        assert page.image.mode == "RGB"
        assert page.width > 0
        assert page.height > 0


def test_no_page_is_silently_ignored() -> None:
    strategy = PDFDocumentStrategy(dpi=100)

    for page_count in [1, 3, 5]:
        source = DocumentSource(
            filename="document.pdf",
            mime_type="application/pdf",
            content=create_pdf_bytes(page_count=page_count),
        )

        pages = strategy.extract_pages(source)

        assert len(pages) == page_count
        assert [p.page_number for p in pages] == list(
            range(1, page_count + 1)
        )


def test_rejects_invalid_pdf_bytes() -> None:
    strategy = PDFDocumentStrategy()

    source = DocumentSource(
        filename="corrupt.pdf",
        mime_type="application/pdf",
        content=b"this-is-not-a-pdf",
    )

    with pytest.raises(InvalidPDFDocumentError):
        strategy.extract_pages(source)


def test_rejects_password_protected_pdf() -> None:
    strategy = PDFDocumentStrategy()

    protected_document = pymupdf.open()
    try:
        protected_document.new_page()
        protected_bytes = protected_document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="secret",
        )
    finally:
        protected_document.close()

    source = DocumentSource(
        filename="protected.pdf",
        mime_type="application/pdf",
        content=protected_bytes,
    )

    with pytest.raises(
        InvalidPDFDocumentError,
        match="Password-protected",
    ):
        strategy.extract_pages(source)


def test_does_not_support_image() -> None:
    strategy = PDFDocumentStrategy()

    source = DocumentSource(
        filename="photo.png",
        mime_type="image/png",
        content=b"\x89PNG",
    )

    assert strategy.supports(source) is False