from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.dependencies import get_pipeline
from app.api.main import create_app
from app.domain.search import SearchQuery
from app.services.document_intelligence_pipeline import (
    RankedDocument,
    SearchOutcome,
)


def test_search_returns_hybrid_results() -> None:
    pipeline = MagicMock()

    pipeline.search.return_value = SearchOutcome(
        query=SearchQuery(
            text="invoice",
            top_k=2,
            document_type=None,
        ),
        ranking_mode="hybrid",
        results=[
            RankedDocument(
                document_id="document-1",
                rank=1,
                final_score=0.86,
                clip_score=0.80,
                caption_score=1.0,
                metadata_score=1.0,
                calibrated_score=0.75,
                caption="an invoice with product rows",
                stored_path="data/runtime/uploads/invoice.png",
                document_type="invoice",
            ),
            RankedDocument(
                document_id="document-2",
                rank=2,
                final_score=0.63,
                clip_score=0.90,
                caption_score=0.0,
                metadata_score=0.0,
                calibrated_score=0.42,
                caption="a landscape photograph",
                stored_path="data/runtime/uploads/other.png",
                document_type="receipt",
            ),
        ],
        duration_ms=42.5,
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "query": "invoice",
            "top_k": 2,
            "use_hybrid_ranking": True,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "invoice"
    assert body["top_k"] == 2
    assert body["ranking_mode"] == "hybrid"
    assert body["duration_ms"] == 42.5
    assert len(body["results"]) == 2

    first_result = body["results"][0]

    assert first_result == {
        "document_id": "document-1",
        "rank": 1,
        "final_score": 0.86,
        "clip_score": 0.80,
        "caption_score": 1.0,
        "metadata_score": 1.0,
        "calibrated_score": 0.75,
        "caption": "an invoice with product rows",
        "image_url": (
            "/api/documents/document-1/image"
        ),
        "document_type": "invoice",
    }

    pipeline.search.assert_called_once()

    call_arguments = pipeline.search.call_args

    search_query = call_arguments.kwargs["query"]

    assert search_query.text == "invoice"
    assert search_query.top_k == 2
    assert search_query.document_type is None
    assert (
        call_arguments.kwargs["use_hybrid_ranking"]
        is True
    )


def test_search_rejects_invalid_top_k() -> None:
    pipeline = MagicMock()

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.post(
        "/api/search",
        json={
            "query": "invoice",
            "top_k": 0,
        },
    )

    assert response.status_code == 422
    pipeline.search.assert_not_called()