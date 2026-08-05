from unittest.mock import create_autospec

import pytest
from PIL import Image

from app.domain.document import DocumentRecord
from app.domain.ocr import (
    OCRBoundingBox,
    OCRPageResult,
    OCRResult,
    OCRWord,
)
from app.domain.prepared_document import DocumentPage
from app.repositories.document_repository import DocumentRepository
from app.services.document_ocr_service import (
    DocumentOCRService,
    OCRProcessingError,
)
from app.strategies.ocr import OCRStrategy


def make_document(
    document_id: str = "document-1",
    ocr: OCRResult | None = None,
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        original_filename="invoice.png",
        stored_path="data/invoice.png",
        checksum="checksum-1",
        width=100,
        height=100,
        page_count=1,
        mime_type="image/png",
        document_type="invoice",
        ocr=ocr,
    )


def make_page(
    page_number: int = 1,
    text: str = "Invoice total 42.50",
) -> DocumentPage:
    return DocumentPage(
        page_number=page_number,
        image=Image.new("RGB", (100, 100)),
    )


def make_ocr_result(
    pages: list[tuple[int, str]] | None = None,
) -> OCRResult:
    if pages is None:
        pages = [(1, "Invoice total 42.50")]

    page_results = []

    for page_number, text in pages:
        word = OCRWord(
            text=text.split()[0],
            confidence=0.95,
            bounding_box=OCRBoundingBox(
                left=0.1, top=0.1, right=0.3, bottom=0.2
            ),
        )
        page_results.append(
            OCRPageResult(
                page_number=page_number,
                text=text,
                words=[word],
                mean_confidence=0.95,
            )
        )

    all_words = [w for p in page_results for w in p.words]
    mean_confidence = (
        sum(w.confidence for w in all_words) / len(all_words)
        if all_words
        else 0.0
    )

    return OCRResult(
        text="\n\n".join(p.text for p in page_results),
        pages=page_results,
        mean_confidence=mean_confidence,
        model_name="fake-ocr",
        model_version="1.0",
    )


class FakeOCRStrategy(OCRStrategy):
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-ocr"

    @property
    def model_version(self) -> str:
        return "1.0"

    def extract(self, pages: list[DocumentPage]) -> OCRResult:
        self.calls += 1

        if self.should_fail:
            raise RuntimeError("Model failure")

        return make_ocr_result()


def test_process_document_calls_strategy_and_updates_ocr() -> None:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    strategy = FakeOCRStrategy()
    service = DocumentOCRService(strategy, repository)

    document = make_document()
    result = service.process_document(
        document=document,
        pages=[make_page()],
    )

    assert isinstance(result, OCRResult)
    assert result.model_name == "fake-ocr"
    assert result.model_version == "1.0"
    assert result.text == "Invoice total 42.50"
    assert result.pages[0].page_number == 1
    assert result.pages[0].words[0].text == "Invoice"
    assert result.mean_confidence == 0.95

    repository.update_ocr.assert_called_once_with(
        document_id=document.id,
        result=result,
    )
    assert strategy.calls == 1


def test_process_document_multi_page_stores_all_pages() -> None:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    class MultiPageOCRStrategy(OCRStrategy):
        @property
        def model_name(self) -> str:
            return "fake-ocr"

        @property
        def model_version(self) -> str:
            return "1.0"

        def extract(
            self, pages
        ) -> OCRResult:
            return make_ocr_result(
                [
                    (p.page_number, f"Page {p.page_number} text")
                    for p in pages
                ]
            )

    service = DocumentOCRService(
        MultiPageOCRStrategy(), repository
    )

    document = DocumentRecord(
        id="document-1",
        original_filename="invoice.pdf",
        stored_path="data/invoice.pdf",
        checksum="checksum-1",
        width=100,
        height=100,
        page_count=3,
        mime_type="application/pdf",
        document_type="invoice",
    )
    pages = [make_page(1), make_page(2), make_page(3)]
    result = service.process_document(
        document=document,
        pages=pages,
    )

    assert len(result.pages) == 3
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.pages[2].page_number == 3
    assert "Page 1 text" in result.text
    assert "Page 2 text" in result.text
    assert "Page 3 text" in result.text


def test_process_document_uses_cached_result() -> None:
    cached_ocr = make_ocr_result()

    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    strategy = FakeOCRStrategy()
    service = DocumentOCRService(strategy, repository)

    document = make_document(ocr=cached_ocr)
    result = service.process_document(
        document=document,
        pages=[make_page()],
    )

    assert result == cached_ocr
    assert strategy.calls == 0
    repository.update_ocr.assert_not_called()


def test_process_document_wraps_model_errors() -> None:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    service = DocumentOCRService(
        FakeOCRStrategy(should_fail=True),
        repository,
    )

    with pytest.raises(
        OCRProcessingError,
        match="OCR failed for document 'document-1'",
    ):
        service.process_document(
            document=make_document(),
            pages=[make_page()],
        )

    repository.update_ocr.assert_not_called()