from io import BytesIO
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from PIL import Image

from app.api.dependencies import get_pipeline
from app.api.main import create_app
from app.domain.document import (
    ArtifactType,
    ModelArtifact,
    DocumentMetadata,
    DocumentRecord,
)
from app.domain.prepared_document import DocumentSource
from app.services.document_intelligence_pipeline import (
    IndexDocumentOutcome,
)


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
            metadata=DocumentMetadata(
                document_type=document_type,
            ),
        )

        embedding = ModelArtifact(
            id="embedding-1",
            document_id=document.id,
            artifact_type=ArtifactType.IMAGE_EMBEDDING,
            model_name="fake-clip",
            storage_path=str(
                tmp_path / "embedding-1.npy"
            ),
            dimensions=512,
        )

        caption = ModelArtifact(
            id="caption-1",
            document_id=document.id,
            artifact_type=ArtifactType.CAPTION,
            model_name="fake-blip",
            content="an invoice document",
        )

        ocr = ModelArtifact(
            id="ocr-1",
            document_id=document.id,
            artifact_type=ArtifactType.OCR,
            model_name="fake-doctr",
            content="Invoice text",
        )

        return IndexDocumentOutcome(
            document=document,
            embedding_artifact=embedding,
            caption_artifact=caption,
            embedding_error=None,
            caption_error=None,
            ocr_artifact=ocr,
            ocr_error=None,
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
    assert body["has_caption"] is True
    assert body["fully_succeeded"] is True
    assert body["reused_document"] is False
    assert body["caption"] == "an invoice document"
    assert body["embedding_model"] == "fake-clip"
    assert body["caption_model"] == "fake-blip"
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
        metadata=DocumentMetadata(
            document_type="invoice",
        ),
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