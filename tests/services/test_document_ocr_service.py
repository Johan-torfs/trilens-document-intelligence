from unittest.mock import create_autospec

import pytest
from PIL import Image

from app.domain.document import ArtifactType, ModelArtifact
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


def test_process_document_stores_ocr_artifact() -> None:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    repository.get_artifacts.return_value = []

    strategy = FakeOCRStrategy()
    service = DocumentOCRService(strategy, repository)

    artifact = service.process_document(
        document_id="document-1",
        pages=[make_page()],
    )

    assert artifact.artifact_type == ArtifactType.OCR
    assert artifact.model_name == "fake-ocr"
    assert artifact.model_version == "1.0"

    result = OCRResult.model_validate_json(artifact.content)

    assert result.text == "Invoice total 42.50"
    assert result.pages[0].page_number == 1
    assert result.pages[0].words[0].text == "Invoice"
    assert result.mean_confidence == 0.95

    repository.save_artifact.assert_called_once_with(artifact)
    assert strategy.calls == 1


def test_process_document_multi_page_stores_all_pages() -> None:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    repository.get_artifacts.return_value = []

    class MultiPageOCRStrategy(OCRStrategy):
        @property
        def model_name(self) -> str:
            return "fake-ocr"

        @property
        def model_version(self) -> str:
            return "1.0"

        def extract(
            self, pages: list[DocumentPage]
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

    pages = [make_page(1), make_page(2), make_page(3)]
    artifact = service.process_document(
        document_id="document-1",
        pages=pages,
    )

    result = OCRResult.model_validate_json(artifact.content)

    assert len(result.pages) == 3
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2
    assert result.pages[2].page_number == 3
    assert "Page 1 text" in result.text
    assert "Page 2 text" in result.text
    assert "Page 3 text" in result.text


def test_process_document_uses_cached_artifact() -> None:
    cached = ModelArtifact(
        id="ocr-1",
        document_id="document-1",
        artifact_type=ArtifactType.OCR,
        model_name="fake-ocr",
        model_version="1.0",
        content=make_ocr_result().model_dump_json(),
    )

    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    repository.get_artifacts.return_value = [cached]

    strategy = FakeOCRStrategy()
    service = DocumentOCRService(strategy, repository)

    result = service.process_document(
        document_id="document-1",
        pages=[make_page()],
    )

    assert result == cached
    assert strategy.calls == 0
    repository.save_artifact.assert_not_called()


def test_process_document_wraps_model_errors() -> None:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    repository.get_artifacts.return_value = []

    service = DocumentOCRService(
        FakeOCRStrategy(should_fail=True),
        repository,
    )

    with pytest.raises(
        OCRProcessingError,
        match="OCR failed for document document-1",
    ):
        service.process_document(
            document_id="document-1",
            pages=[make_page()],
        )

    repository.save_artifact.assert_not_called()