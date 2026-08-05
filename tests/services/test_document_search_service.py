from unittest.mock import create_autospec

from app.domain.document import DocumentRecord
from app.domain.search import SearchQuery
from app.repositories.document_repository import DocumentRepository
from app.services.document_search_service import DocumentSearchService
from app.services.retrieval_service import DocumentSearchMatch, RetrievalService


def make_document(
    document_id: str = "document-001",
    document_type: str = "invoice",
    stored_path: str = "data/documents/invoice.png",
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        original_filename="invoice.png",
        stored_path=stored_path,
        checksum="test-checksum",
        width=100,
        height=100,
        page_count=1,
        mime_type="image/png",
        document_type=document_type,
    )


def make_match(
    document_id: str = "document-001",
    score: float = 0.91,
    best_page_number: int = 1,
) -> DocumentSearchMatch:
    return DocumentSearchMatch(
        document_id=document_id,
        score=score,
        best_page_number=best_page_number,
        pages=(),
    )


def test_search_returns_enriched_document_result() -> None:
    retrieval_service = create_autospec(
        RetrievalService,
        instance=True,
    )
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    document = make_document()
    retrieval_service.search.return_value = [make_match()]
    retrieval_service.search_text.return_value = []
    document_repository.get_document.return_value = document
    document_repository.lexical_search.return_value = []

    search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    results = search_service.search(
        SearchQuery(
            text="an invoice",
            top_k=5,
        )
    )

    assert len(results) == 1
    result = results[0]
    assert result.document_id == "document-001"
    assert result.score == 0.91
    assert result.rank == 1
    assert result.stored_path == "data/documents/invoice.png"
    assert result.document_type == "invoice"

    retrieval_service.search.assert_called_once_with(
        query="an invoice",
        top_k=5,
        document_type=None,
    )


def test_search_passes_document_type_filter() -> None:
    retrieval_service = create_autospec(
        RetrievalService,
        instance=True,
    )
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    retrieval_service.search.return_value = []
    retrieval_service.search_text.return_value = []
    document_repository.lexical_search.return_value = []

    search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    results = search_service.search(
        SearchQuery(
            text="a receipt",
            top_k=5,
            document_type="receipt",
        )
    )

    assert results == []

    retrieval_service.search.assert_called_once_with(
        query="a receipt",
        top_k=5,
        document_type="receipt",
    )


def test_search_skips_missing_documents() -> None:
    retrieval_service = create_autospec(
        RetrievalService,
        instance=True,
    )
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )

    retrieval_service.search.return_value = [
        make_match(document_id="missing")
    ]
    retrieval_service.search_text.return_value = []
    document_repository.get_document.return_value = None
    document_repository.lexical_search.return_value = []

    search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    results = search_service.search(
        SearchQuery(text="invoice", top_k=5)
    )

    assert results == []