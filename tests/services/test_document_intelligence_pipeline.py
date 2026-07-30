from pathlib import Path
from unittest.mock import MagicMock, create_autospec, patch

from PIL import Image
import pytest

from app.domain.document import (
    ArtifactType,
    DocumentMetadata,
    DocumentRecord,
    ModelArtifact,
    ProcessingStatus,
)
from app.repositories.document_repository import DocumentRepository
from app.services.document_caption_service import (
    DocumentCaptionService,
)
from app.services.document_intelligence_pipeline import (
    DocumentIntelligencePipeline,
)
from app.services.indexing_service import IndexingService
from app.domain.search import SearchQuery, SearchResult
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.retrieval_service import RetrievalService
from app.services.analysis_service import (
    AnalysisResult,
    AnalysisService,
)


def make_document() -> DocumentRecord:
    return DocumentRecord(
        id="document-1",
        original_filename="invoice.png",
        stored_path="data/documents/invoice.png",
        checksum="checksum-1",
        width=800,
        height=1000,
        mime_type="image/png",
        document_type="invoice",
        metadata=DocumentMetadata(
            document_type="invoice",
        ),
    )


def make_embedding_artifact() -> ModelArtifact:
    return ModelArtifact(
        id="embedding-1",
        document_id="document-1",
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="fake-clip",
        storage_path="data/vectors/embedding-1.npy",
        dimensions=512,
    )


def make_caption_artifact() -> ModelArtifact:
    return ModelArtifact(
        id="caption-1",
        document_id="document-1",
        artifact_type=ArtifactType.CAPTION,
        model_name="fake-blip",
        model_version="version-1",
        content="an invoice with several product rows",
    )


def make_pipeline() -> tuple[
    DocumentIntelligencePipeline,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    indexing_service = create_autospec(
        IndexingService,
        instance=True,
    )
    caption_service = create_autospec(
        DocumentCaptionService,
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

    indexing_service.model_name = "fake-clip"
    caption_service.model_name = "fake-blip"
    caption_service.model_version = "version-1"

    pipeline = DocumentIntelligencePipeline(
        document_repository=repository,
        indexing_service=indexing_service,
        caption_service=caption_service,
        document_search_service=document_search_service,
        retrieval_service=retrieval_service,
        analysis_service=analysis_service,
    )

    return (
        pipeline,
        repository,
        indexing_service,
        caption_service,
        document_search_service,
        retrieval_service,
        analysis_service,
    )


def test_index_document_runs_complete_pipeline() -> None:
    (
        pipeline,
        repository,
        indexing_service,
        caption_service,
        _,
        _,
        _,
    ) = make_pipeline()

    document = make_document()
    embedding = make_embedding_artifact()
    caption = make_caption_artifact()

    repository.get_document_by_checksum.return_value = None
    indexing_service.index_image.return_value = embedding
    caption_service.caption_document.return_value = caption

    processed_image = Image.new(
        "RGB",
        (600, 800),
        "white",
    )
    preprocessing_result = MagicMock(
        image=processed_image
    )

    with patch(
        "app.services.document_intelligence_pipeline"
        ".preprocess_image",
        return_value=preprocessing_result,
    ) as preprocess:
        outcome = pipeline.index_document(
            document=document,
            image_path=Path("invoice.png"),
        )

    preprocess.assert_called_once_with(
        Path("invoice.png")
    )

    repository.save_document.assert_called_once_with(
        document
    )

    indexing_service.index_image.assert_called_once_with(
        document_id=document.id,
        image=processed_image,
    )

    caption_service.caption_document.assert_called_once_with(
        document_id=document.id,
        image=processed_image,
    )

    assert outcome.embedding_artifact == embedding
    assert outcome.caption_artifact == caption
    assert outcome.is_searchable is True
    assert outcome.has_caption is True
    assert outcome.fully_succeeded is True
    assert outcome.reused_document is False
    assert outcome.duration_ms >= 0

    assert repository.update_processing_status.call_args_list[
        0
    ].args == (
        document.id,
        ProcessingStatus.PROCESSING,
    )

    assert repository.update_processing_status.call_args_list[
        1
    ].args == (
        document.id,
        ProcessingStatus.COMPLETED,
        None,
    )

def test_index_document_reuses_existing_artifacts() -> None:
    (
        pipeline,
        repository,
        indexing_service,
        caption_service,
        _,
        _,
        _,
    ) = make_pipeline()

    existing_document = make_document()
    embedding = make_embedding_artifact()
    caption = make_caption_artifact()

    repository.get_document_by_checksum.return_value = (
        existing_document
    )
    repository.get_artifacts.return_value = [
        embedding,
        caption,
    ]

    with patch(
        "app.services.document_intelligence_pipeline"
        ".preprocess_image",
    ) as preprocess:
        outcome = pipeline.index_document(
            document=make_document(),
            image_path=Path("invoice.png"),
        )

    preprocess.assert_not_called()
    repository.save_document.assert_not_called()
    repository.update_processing_status.assert_not_called()
    indexing_service.index_image.assert_not_called()
    caption_service.caption_document.assert_not_called()

    assert outcome.document == existing_document
    assert outcome.embedding_artifact == embedding
    assert outcome.caption_artifact == caption
    assert outcome.reused_document is True


def test_caption_failure_does_not_block_indexing() -> None:
    (
        pipeline,
        repository,
        indexing_service,
        caption_service,
        _,
        _,
        _,
    ) = make_pipeline()

    document = make_document()
    embedding = make_embedding_artifact()

    repository.get_document_by_checksum.return_value = None
    indexing_service.index_image.return_value = embedding
    caption_service.caption_document.side_effect = RuntimeError(
        "Captionmodel kon niet worden uitgevoerd."
    )

    preprocessing_result = MagicMock(
        image=Image.new(
            "RGB",
            (600, 800),
            "white",
        )
    )

    with patch(
        "app.services.document_intelligence_pipeline"
        ".preprocess_image",
        return_value=preprocessing_result,
    ):
        outcome = pipeline.index_document(
            document=document,
            image_path=Path("invoice.png"),
        )

    assert outcome.embedding_artifact == embedding
    assert outcome.caption_artifact is None
    assert outcome.embedding_error is None
    assert (
        outcome.caption_error
        == "Captionmodel kon niet worden uitgevoerd."
    )

    assert outcome.is_searchable is True
    assert outcome.has_caption is False
    assert outcome.fully_succeeded is False

    assert repository.update_processing_status.call_args_list[
        -1
    ].args == (
        document.id,
        ProcessingStatus.COMPLETED,
        "Captionmodel kon niet worden uitgevoerd.",
    )


def test_search_returns_clip_baseline_without_reranking() -> None:
    (
        pipeline,
        _,
        _,
        _,
        document_search_service,
        retrieval_service,
        _,
    ) = make_pipeline()

    query = SearchQuery(
        text="invoice with product rows",
        top_k=2,
    )

    document_search_service.search.return_value = [
        SearchResult(
            document_id="document-1",
            score=0.91,
            rank=1,
            caption="an invoice document",
            stored_path="data/documents/invoice.png",
            document_type="invoice",
        ),
        SearchResult(
            document_id="document-2",
            score=0.72,
            rank=2,
            caption="a receipt",
            stored_path="data/documents/receipt.png",
            document_type="receipt",
        ),
    ]

    outcome = pipeline.search(
        query=query,
        use_hybrid_ranking=False,
    )

    document_search_service.search.assert_called_once_with(
        query
    )
    retrieval_service.text_similarity.assert_not_called()

    assert outcome.ranking_mode == "clip"
    assert len(outcome.results) == 2

    assert outcome.results[0].document_id == "document-1"
    assert outcome.results[0].rank == 1
    assert outcome.results[0].final_score == 0.91
    assert outcome.results[0].clip_score == 0.91
    assert outcome.results[0].caption_score == 0.0
    assert outcome.results[0].metadata_score == 0.0

    assert outcome.duration_ms >= 0


def test_hybrid_search_reranks_larger_candidate_pool() -> None:
    (
        pipeline,
        _,
        _,
        _,
        document_search_service,
        retrieval_service,
        _,
    ) = make_pipeline()

    query = SearchQuery(
        text="invoice",
        top_k=2,
    )

    document_search_service.search.return_value = [
        SearchResult(
            document_id="document-a",
            score=0.90,
            rank=1,
            caption="a landscape photograph",
            stored_path="data/documents/a.png",
            document_type="receipt",
        ),
        SearchResult(
            document_id="document-b",
            score=0.80,
            rank=2,
            caption="an invoice with product rows",
            stored_path="data/documents/b.png",
            document_type="invoice",
        ),
        SearchResult(
            document_id="document-c",
            score=0.70,
            rank=3,
            caption=None,
            stored_path="data/documents/c.png",
            document_type="purchase_order",
        ),
    ]

    similarity_scores = {
        ("invoice", "a landscape photograph"): 0.0,
        ("invoice", "receipt"): 0.0,
        ("invoice", "an invoice with product rows"): 1.0,
        ("invoice", "invoice"): 1.0,
        ("invoice", "purchase order"): 0.0,
    }

    retrieval_service.text_similarity.side_effect = (
        lambda first, second: similarity_scores[
            (first, second)
        ]
    )

    outcome = pipeline.search(
        query=query,
        use_hybrid_ranking=True,
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

    assert first_result.document_id == "document-b"
    assert first_result.rank == 1
    assert first_result.clip_score == 0.80
    assert first_result.caption_score == 1.0
    assert first_result.metadata_score == 1.0
    assert first_result.final_score == pytest.approx(0.86)

    assert second_result.document_id == "document-a"
    assert second_result.rank == 2
    assert second_result.final_score == pytest.approx(0.63)


def test_analyze_document_uses_stored_image() -> None:
    (
        pipeline,
        repository,
        _,
        _,
        _,
        _,
        analysis_service,
    ) = make_pipeline()

    document = make_document()
    repository.get_document.return_value = document

    processed_image = Image.new(
        "RGB",
        (600, 800),
        "white",
    )

    preprocessing_result = MagicMock(
        image=processed_image
    )

    analysis_result = AnalysisResult(
        text="A signature may be visible at the bottom.",
        source="open_flamingo",
        model_name="fake-open-flamingo",
        model_version="version-1",
        duration_ms=1200.0,
    )

    analysis_service.analyze.return_value = (
        analysis_result
    )

    with patch(
        "app.services.document_intelligence_pipeline"
        ".preprocess_image",
        return_value=preprocessing_result,
    ) as preprocess:
        outcome = pipeline.analyze_document(
            document_id=" document-1 ",
            question=" Is there a signature? ",
        )

    repository.get_document.assert_called_once_with(
        "document-1"
    )

    preprocess.assert_called_once_with(
        Path(document.stored_path)
    )

    analysis_service.analyze.assert_called_once_with(
        image=processed_image,
        question="Is there a signature?",
    )

    assert outcome.document == document
    assert outcome.question == "Is there a signature?"
    assert outcome.analysis == analysis_result
    assert outcome.used_fallback is False
    assert outcome.duration_ms >= 0


def test_analyze_document_rejects_unknown_document() -> None:
    (
        pipeline,
        repository,
        _,
        _,
        _,
        _,
        analysis_service,
    ) = make_pipeline()

    repository.get_document.return_value = None

    with patch(
        "app.services.document_intelligence_pipeline"
        ".preprocess_image",
    ) as preprocess:
        with pytest.raises(
            ValueError,
            match="werd niet gevonden",
        ):
            pipeline.analyze_document(
                document_id="missing-document",
                question="Describe this document.",
            )

    preprocess.assert_not_called()
    analysis_service.analyze.assert_not_called()