from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_pipeline
from app.api.main import create_app
from app.domain.document import DocumentRecord
from app.domain.ocr import OCRPageResult, OCRResult
from app.domain.prepared_document import DocumentSource
from app.services.document_intelligence_pipeline import (
    IndexDocumentOutcome,
)
from app.services.indexing_service import IndexingResult


def make_png_bytes() -> bytes:
    buffer = BytesIO()

    Image.new(
        "RGB",
        (200, 300),
        "white",
    ).save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_upload_indexes_document(
    tmp_path,
    monkeypatch,
) -> None:
    pipeline = MagicMock()

    def fake_index_document(
        source: DocumentSource,
        document_type: str,
    ) -> IndexDocumentOutcome:
        document = DocumentRecord(
            id="document-1",
            original_filename=source.filename,
            stored_path=str(tmp_path / source.filename),
            checksum="checksum-1",
            width=200,
            height=300,
            mime_type=source.mime_type,
            document_type=document_type,
            page_count=1,
        )

        indexing_result = IndexingResult(
            document_id=document.id,
            page_count=1,
            dimensions=512,
            model_name="fake-clip",
            model_version=None,
            reused_existing=False,
        )

        ocr_result = OCRResult(
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

        return IndexDocumentOutcome(
            document=document,
            indexing_result=indexing_result,
            ocr_result=ocr_result,
            text_indexing_result=None,
            indexing_error=None,
            ocr_error=None,
            text_indexing_error=None,
            reused_document=False,
            duration_ms=125.0,
        )

    pipeline.index_document.side_effect = (
        fake_index_document
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.post(
        "/api/documents",
        files={
            "file": (
                "invoice.png",
                make_png_bytes(),
                "image/png",
            )
        },
        data={
            "document_type": "invoice",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["document_type"] == "invoice"
    assert body["original_filename"] == "invoice.png"
    assert body["is_searchable"] is True
    assert body["has_ocr"] is True
    assert body["fully_succeeded"] is True
    assert body["reused_document"] is False
    assert body["duration_ms"] == 125.0

    pipeline.index_document.assert_called_once()


def test_get_document_image_returns_file(
    tmp_path,
) -> None:
    image_path = tmp_path / "invoice.png"
    image_bytes = make_png_bytes()
    image_path.write_bytes(image_bytes)

    document = DocumentRecord(
        id="document-1",
        original_filename="invoice.png",
        stored_path=str(image_path),
        checksum="checksum-1",
        width=200,
        height=300,
        mime_type="image/png",
        document_type="invoice",
    )

    pipeline = MagicMock()
    pipeline.get_document_file.return_value = (
        document,
        image_path,
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.get(
        "/api/documents/document-1/image"
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == image_bytes

    pipeline.get_document_file.assert_called_once_with(
        "document-1"
    )


def test_get_document_image_returns_404() -> None:
    pipeline = MagicMock()
    pipeline.get_document_file.side_effect = ValueError(
        "Document 'missing' werd niet gevonden."
    )

    app = create_app()
    app.dependency_overrides[get_pipeline] = (
        lambda: pipeline
    )

    client = TestClient(app)

    response = client.get(
        "/api/documents/missing/image"
    )

    assert response.status_code == 404
    assert "niet gevonden" in response.json()["detail"]