from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.api.dependencies import get_pipeline
from app.api.main import create_app
from app.domain.document import (
    DocumentMetadata,
    DocumentRecord,
)
from app.services.analysis_service import (
    AnalysisDisabledError,
    AnalysisResult,
)
from app.services.document_intelligence_pipeline import (
    AnalyzeDocumentOutcome,
)


def make_document() -> DocumentRecord:
    return DocumentRecord(
        id="document-1",
        original_filename="invoice.png",
        stored_path="data/runtime/uploads/invoice.png",
        checksum="checksum-1",
        width=800,
        height=1000,
        mime_type="image/png",
        document_type="invoice",
        metadata=DocumentMetadata(
            document_type="invoice",
        ),
    )


def test_analysis_returns_model_result() -> None:
    pipeline = MagicMock()

    pipeline.analyze_document.return_value = (
        AnalyzeDocumentOutcome(
            document=make_document(),
            question="Is there a signature?",
            analysis=AnalysisResult(
                text="A signature may be visible.",
                source="open_flamingo",
                model_name="fake-open-flamingo",
                model_version="version-1",
                duration_ms=1200.0,
            ),
            duration_ms=1250.0,
        )
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.post(
        "/api/documents/document-1/analysis",
        json={
            "question": "Is there a signature?",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "document_id": "document-1",
        "question": "Is there a signature?",
        "text": "A signature may be visible.",
        "source": "open_flamingo",
        "used_fallback": False,
        "model_name": "fake-open-flamingo",
        "model_version": "version-1",
        "model_duration_ms": 1200.0,
        "total_duration_ms": 1250.0,
    }

    pipeline.analyze_document.assert_called_once_with(
        document_id="document-1",
        question="Is there a signature?",
    )

def test_analysis_exposes_caption_fallback() -> None:
    pipeline = MagicMock()

    pipeline.analyze_document.return_value = (
        AnalyzeDocumentOutcome(
            document=make_document(),
            question="Describe this document.",
            analysis=AnalysisResult(
                text="an invoice with several rows",
                source="caption_fallback",
                model_name="fake-blip",
                model_version="version-1",
                duration_ms=30.0,
            ),
            duration_ms=50.0,
        )
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.post(
        "/api/documents/document-1/analysis",
        json={
            "question": "Describe this document.",
        },
    )

    assert response.status_code == 200
    assert response.json()["source"] == "caption_fallback"
    assert response.json()["used_fallback"] is True

def test_analysis_returns_503_when_disabled() -> None:
    pipeline = MagicMock()

    pipeline.analyze_document.side_effect = (
        AnalysisDisabledError(
            "OpenFlamingo-analyse is uitgeschakeld."
        )
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.post(
        "/api/documents/document-1/analysis",
        json={
            "question": "Describe this document.",
        },
    )

    assert response.status_code == 503
    assert "uitgeschakeld" in response.json()["detail"]