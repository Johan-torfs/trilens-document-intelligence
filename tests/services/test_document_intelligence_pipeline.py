from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

import pymupdf
import pytest
from PIL import Image

from app.domain.document import (
    DocumentRecord,
    ProcessingStatus,
)
from app.domain.ocr import OCRPageResult, OCRResult
from app.domain.prepared_document import (
    DocumentPage,
    DocumentSource,
    PreparedDocument,
)
from app.preprocessing.pipeline import PreprocessingResult
from app.repositories.document_repository import DocumentRepository
from app.services.analysis_service import (
    AnalysisResult,
    AnalysisService,
)
from app.services.document_intelligence_pipeline import (
    DocumentIntelligencePipeline,
)
from app.services.document_classification_service import (
    DocumentClassificationService,
)
from app.domain.classification import ClassificationResult
from app.services.document_ocr_service import DocumentOCRService
from app.services.document_preparation_service import (
    DocumentPreparationService,
)
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.indexing_service import IndexingResult, IndexingService
from app.services.retrieval_service import RetrievalService
from app.services.score_calibration import LinearScoreCalibrator
from app.domain.search import SearchQuery, SearchResult


def make_png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (800, 1000), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def make_pdf_bytes(page_count: int = 2) -> bytes:
    document = pymupdf.open()
    try:
        for i in range(page_count):
            page = document.new_page(width=200, height=300)
            page.insert_text((30, 50), f"Page {i + 1}")
        return document.tobytes()
    finally:
        document.close()


def make_image_source() -> DocumentSource:
    return DocumentSource(
        filename="invoice.png",
        mime_type="image/png",
        content=make_png_bytes(),
    )


def make_pdf_source(page_count: int = 2) -> DocumentSource:
    return DocumentSource(
        filename="invoice.pdf",
        mime_type="application/pdf",
        content=make_pdf_bytes(page_count),
    )


def make_page(page_number: int = 1) -> DocumentPage:
    return DocumentPage(
        page_number=page_number,
        image=Image.new("RGB", (200, 300), "white"),
    )


def make_preprocessing_result(
    page: DocumentPage,
) -> PreprocessingResult:
    return PreprocessingResult(
        image=page.image,
        original_size=(page.width, page.height),
        processed_size=(page.width, page.height),
        transforms=[],
        duration_ms=1.0,
    )


def make_indexing_result(
    document_id: str = "document-1",
) -> IndexingResult:
    return IndexingResult(
        document_id=document_id,
        page_count=1,
        dimensions=512,
        model_name="fake-clip",
        model_version=None,
        reused_existing=False,
    )


def make_ocr_result(
    document_id: str = "document-1",
) -> OCRResult:
    return OCRResult(
        text="Invoice text",
        pages=[
            OCRPageResult(
                page_number=1,
                text="Invoice text",
                words=[],
                mean_confidence=0.9,
            )
        ],
        mean_confidence=0.9,
        model_name="fake-doctr",
        model_version="version-1",
    )


def make_pipeline(
    tmp_path: Path | None = None,
) -> tuple[
    DocumentIntelligencePipeline,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    upload_dir = (
        tmp_path / "uploads"
        if tmp_path
        else Path("/tmp/test_uploads")
    )

    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    preparation_service = create_autospec(
        DocumentPreparationService,
        instance=True,
    )
    indexing_service = create_autospec(
        IndexingService,
        instance=True,
    )
    document_search_service = create_autospec(
        DocumentSearchService,
        instance=True,
    )
    retrieval_service = create_autospec(
        RetrievalService,
        instance=True,
    )
    analysis_service = create_autospec(
        AnalysisService,
        instance=True,
    )
    ocr_service = create_autospec(
        DocumentOCRService,
        instance=True,
    )

    indexing_service.model_name = "fake-clip"
    ocr_service.model_name = "fake-doctr"
    ocr_service.model_version = "version-1"

    pipeline = DocumentIntelligencePipeline(
        document_repository=repository,
        preparation_service=preparation_service,
        upload_dir=upload_dir,
        indexing_service=indexing_service,
        document_search_service=document_search_service,
        retrieval_service=retrieval_service,
        analysis_service=analysis_service,
        ocr_service=ocr_service,
        score_calibrator=LinearScoreCalibrator(
            noise_floor=0.04,
            ceiling=0.28,
        ),
    )

    return (
        pipeline,
        repository,
        preparation_service,
        indexing_service,
        document_search_service,
        retrieval_service,
        analysis_service,
        ocr_service,
    )


def test_index_document_image_runs_complete_pipeline(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        indexing_service,
        _,
        _,
        _,
        ocr_service,
    ) = make_pipeline(tmp_path)

    source = make_image_source()
    page = make_page(1)
    indexing_result = make_indexing_result()
    ocr_result = make_ocr_result()

    repository.get_document_by_checksum.return_value = None
    repository.get_document.return_value = None
    indexing_service.index_pages.return_value = indexing_result
    ocr_service.process_document.return_value = ocr_result

    prepared = PreparedDocument(source=source, pages=[page])
    preprocessing = [make_preprocessing_result(page)]

    preparation_service.prepare.return_value = prepared
    preparation_service.preprocess_pages.return_value = preprocessing

    outcome = pipeline.index_document(
        source=source,
        document_type="invoice",
    )

    preparation_service.prepare.assert_called_once_with(source)
    preparation_service.preprocess_pages.assert_called_once_with(
        prepared
    )

    indexing_service.index_pages.assert_called_once()
    ocr_service.process_document.assert_called_once()

    called_pages = ocr_service.process_document.call_args.kwargs[
        "pages"
    ]
    assert len(called_pages) == 1
    assert called_pages[0].page_number == 1

    repository.save_document.assert_called_once()
    saved_doc = repository.save_document.call_args.args[0]
    assert saved_doc.original_filename == "invoice.png"
    assert saved_doc.mime_type == "image/png"
    assert saved_doc.page_count == 1

    assert outcome.indexing_result == indexing_result
    assert outcome.ocr_result == ocr_result
    assert outcome.is_searchable is True
    assert outcome.has_ocr is True
    assert outcome.fully_succeeded is True
    assert outcome.reused_document is False
    assert outcome.duration_ms >= 0

    status_calls = (
        repository.update_processing_status.call_args_list
    )
    assert status_calls[0].kwargs["status"] == ProcessingStatus.PROCESSING
    assert status_calls[1].kwargs["status"] == ProcessingStatus.COMPLETED
    assert status_calls[1].kwargs.get("error") is None


def test_index_document_pdf_processes_all_pages(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        indexing_service,
        _,
        _,
        _,
        ocr_service,
    ) = make_pipeline(tmp_path)

    source = make_pdf_source(page_count=3)
    pages = [make_page(1), make_page(2), make_page(3)]
    indexing_result = make_indexing_result()
    ocr_result = make_ocr_result()

    repository.get_document_by_checksum.return_value = None
    repository.get_document.return_value = None
    indexing_service.index_pages.return_value = indexing_result
    ocr_service.process_document.return_value = ocr_result

    prepared = PreparedDocument(source=source, pages=pages)
    preprocessing = [make_preprocessing_result(p) for p in pages]

    preparation_service.prepare.return_value = prepared
    preparation_service.preprocess_pages.return_value = preprocessing

    outcome = pipeline.index_document(
        source=source,
        document_type="invoice",
    )

    saved_doc = repository.save_document.call_args.args[0]
    assert saved_doc.page_count == 3
    assert saved_doc.mime_type == "application/pdf"

    index_pages_call = indexing_service.index_pages.call_args.kwargs
    assert len(index_pages_call["pages"]) == 3

    ocr_pages = ocr_service.process_document.call_args.kwargs[
        "pages"
    ]
    assert len(ocr_pages) == 3
    assert [p.page_number for p in ocr_pages] == [1, 2, 3]

    assert outcome.is_searchable is True
    assert outcome.has_ocr is True

def test_index_document_reuses_existing_artifacts(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        indexing_service,
        _,
        _,
        _,
        ocr_service,
    ) = make_pipeline(tmp_path)

    existing_document = DocumentRecord(
        id="document-1",
        original_filename="invoice.png",
        stored_path=str(tmp_path / "invoice.png"),
        checksum="any-checksum",
        width=800,
        height=1000,
        mime_type="image/png",
        document_type="invoice",
        page_count=1,
    )
    indexing_result = make_indexing_result()
    ocr_result = make_ocr_result()

    repository.get_document_by_checksum.return_value = (
        existing_document
    )
    indexing_service.current_result.return_value = indexing_result
    ocr_service.current_result.return_value = ocr_result

    outcome = pipeline.index_document(
        source=make_image_source(),
        document_type="invoice",
    )

    preparation_service.prepare.assert_not_called()
    repository.save_document.assert_not_called()
    repository.update_processing_status.assert_not_called()
    indexing_service.index_pages.assert_not_called()
    ocr_service.process_document.assert_not_called()

    assert outcome.document == existing_document
    assert outcome.indexing_result == indexing_result
    assert outcome.ocr_result == ocr_result
    assert outcome.reused_document is True


def test_hybrid_search_reranks_larger_candidate_pool() -> None:
    (
        pipeline,
        _,
        _,
        _,
        document_search_service,
        retrieval_service,
        _,
        _,
    ) = make_pipeline()

    query = SearchQuery(
        text="invoice",
        top_k=2,
    )

    # document-b has a lower visual score but a strong text match;
    # hybrid ranking should rank it above document-a.
    document_search_service.search.return_value = [
        SearchResult(
            document_id="document-a",
            score=0.45,
            rank=1,
            stored_path="data/documents/a.png",
            document_type="receipt",
        ),
        SearchResult(
            document_id="document-b",
            score=0.40,
            rank=2,
            stored_path="data/documents/b.png",
            document_type="invoice",
            text_score=0.80,
        ),
        SearchResult(
            document_id="document-c",
            score=0.30,
            rank=3,
            stored_path="data/documents/c.png",
            document_type="purchase_order",
        ),
    ]

    outcome = pipeline.search(
        query=query,
    )

    requested_query = (
        document_search_service.search.call_args.args[0]
    )

    assert requested_query.text == "invoice"
    assert requested_query.top_k == 6

    assert outcome.ranking_mode == "hybrid"
    assert len(outcome.results) == 2

    first_result = outcome.results[0]
    second_result = outcome.results[1]

    # document-b: 0.60*0.40 + 0.30*0.80 + 0.10*0 = 0.24 + 0.24 = 0.48
    assert first_result.document_id == "document-b"
    assert first_result.rank == 1
    assert first_result.visual_score == 0.40
    assert first_result.text_score == pytest.approx(0.80)
    assert first_result.final_score == pytest.approx(0.48)

    # document-a: no text or fts score, final = visual = 0.45
    assert second_result.document_id == "document-a"
    assert second_result.rank == 2
    assert second_result.final_score == pytest.approx(0.45)


def test_analyze_document_uses_preparation_service(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        _,
        _,
        _,
        analysis_service,
        _,
    ) = make_pipeline(tmp_path)

    stored_file = tmp_path / "invoice.png"
    Image.new("RGB", (800, 1000), "white").save(stored_file)

    document = DocumentRecord(
        id="document-1",
        original_filename="invoice.png",
        stored_path=str(stored_file),
        checksum="checksum-1",
        width=800,
        height=1000,
        mime_type="image/png",
        document_type="invoice",
    )

    repository.get_document.return_value = document

    first_page_image = Image.new("RGB", (600, 800), "white")
    page = DocumentPage(
        page_number=1, image=first_page_image
    )
    preprocessing = PreprocessingResult(
        image=first_page_image,
        original_size=(600, 800),
        processed_size=(600, 800),
        transforms=[],
        duration_ms=1.0,
    )

    prepared = PreparedDocument(
        source=DocumentSource(
            filename="invoice.png",
            mime_type="image/png",
            content=stored_file.read_bytes(),
        ),
        pages=[page],
    )

    preparation_service.prepare.return_value = prepared
    preparation_service.preprocess_pages.return_value = [
        preprocessing
    ]

    analysis_result = AnalysisResult(
        text="A signature may be visible at the bottom.",
        model_name="fake-open-flamingo",
        model_version="version-1",
        duration_ms=1200.0,
    )

    analysis_service.analyze.return_value = analysis_result

    outcome = pipeline.analyze_document(
        document_id=" document-1 ",
        question=" Is there a signature? ",
    )

    repository.get_document.assert_called_once_with(
        "document-1"
    )
    preparation_service.prepare.assert_called_once()
    preparation_service.preprocess_pages.assert_called_once()

    analysis_service.analyze.assert_called_once_with(
        image=first_page_image,
        question="Is there a signature?",
    )

    assert outcome.document == document
    assert outcome.question == "Is there a signature?"
    assert outcome.analysis == analysis_result
    assert outcome.duration_ms >= 0


def test_analyze_document_rejects_unknown_document(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        _,
        _,
        _,
        analysis_service,
        _,
    ) = make_pipeline(tmp_path)

    repository.get_document.return_value = None

    with pytest.raises(
        ValueError,
        match="werd niet gevonden",
    ):
        pipeline.analyze_document(
            document_id="missing-document",
            question="Describe this document.",
        )

    preparation_service.prepare.assert_not_called()
    analysis_service.analyze.assert_not_called()


# ---------------------------------------------------------------------------
# Auto-classification tests
# ---------------------------------------------------------------------------

def make_pipeline_with_classifier(
    tmp_path: Path,
) -> tuple[
    DocumentIntelligencePipeline,
    MagicMock,  # repository
    MagicMock,  # preparation_service
    MagicMock,  # indexing_service
    MagicMock,  # ocr_service
    MagicMock,  # classification_service
]:
    upload_dir = tmp_path / "uploads"

    repository = create_autospec(DocumentRepository, instance=True)
    preparation_service = create_autospec(DocumentPreparationService, instance=True)
    indexing_service = create_autospec(IndexingService, instance=True)
    document_search_service = create_autospec(DocumentSearchService, instance=True)
    retrieval_service = create_autospec(RetrievalService, instance=True)
    analysis_service = create_autospec(AnalysisService, instance=True)
    ocr_service = create_autospec(DocumentOCRService, instance=True)
    classification_service = create_autospec(
        DocumentClassificationService, instance=True
    )

    indexing_service.model_name = "fake-clip"
    ocr_service.model_name = "fake-doctr"
    ocr_service.model_version = "version-1"

    pipeline = DocumentIntelligencePipeline(
        document_repository=repository,
        preparation_service=preparation_service,
        upload_dir=upload_dir,
        indexing_service=indexing_service,
        document_search_service=document_search_service,
        retrieval_service=retrieval_service,
        analysis_service=analysis_service,
        ocr_service=ocr_service,
        score_calibrator=LinearScoreCalibrator(noise_floor=0.04, ceiling=0.28),
        classification_service=classification_service,
    )

    return pipeline, repository, preparation_service, indexing_service, ocr_service, classification_service


def test_auto_classify_calls_classification_service_and_updates_type(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        indexing_service,
        ocr_service,
        classification_service,
    ) = make_pipeline_with_classifier(tmp_path)

    source = make_image_source()
    page = make_page(1)
    indexing_result = make_indexing_result()
    ocr_result = make_ocr_result()

    classification_result = ClassificationResult(
        document_type="invoice",
        confidence=0.82,
        visual_score=0.75,
        lexical_score=0.65,
        is_confident=True,
    )

    repository.get_document_by_checksum.return_value = None
    repository.get_document.return_value = None
    indexing_service.index_pages.return_value = indexing_result
    ocr_service.process_document.return_value = ocr_result
    classification_service.classify.return_value = classification_result

    prepared = PreparedDocument(source=source, pages=[page])
    preparation_service.prepare.return_value = prepared
    preparation_service.preprocess_pages.return_value = [make_preprocessing_result(page)]

    outcome = pipeline.index_document(source=source, document_type=None)

    classification_service.classify.assert_called_once()
    repository.update_document_type.assert_called_once_with(
        document_id=repository.save_document.call_args.args[0].id,
        document_type="invoice",
    )
    assert outcome.classification_confidence == pytest.approx(0.82)


def test_auto_classify_raises_without_classification_service(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = make_pipeline(tmp_path)

    with pytest.raises(ValueError, match="classificationservice"):
        pipeline.index_document(source=make_image_source(), document_type=None)


def test_explicit_document_type_skips_classification(
    tmp_path: Path,
) -> None:
    (
        pipeline,
        repository,
        preparation_service,
        indexing_service,
        ocr_service,
        classification_service,
    ) = make_pipeline_with_classifier(tmp_path)

    source = make_image_source()
    page = make_page(1)
    indexing_result = make_indexing_result()
    ocr_result = make_ocr_result()

    repository.get_document_by_checksum.return_value = None
    repository.get_document.return_value = None
    indexing_service.index_pages.return_value = indexing_result
    ocr_service.process_document.return_value = ocr_result

    prepared = PreparedDocument(source=source, pages=[page])
    preparation_service.prepare.return_value = prepared
    preparation_service.preprocess_pages.return_value = [make_preprocessing_result(page)]

    outcome = pipeline.index_document(source=source, document_type="receipt")

    classification_service.classify.assert_not_called()
    repository.update_document_type.assert_not_called()
    assert outcome.classification_confidence is None