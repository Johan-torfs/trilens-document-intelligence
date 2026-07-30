import pytest

from app.domain.document import (
    ArtifactType,
    DocumentRecord,
    ModelArtifact,
    ProcessingStatus,
)
from app.repositories.sqlite_document_repository import (
    DuplicateDocumentError,
    SQLiteDocumentRepository,
)


def test_saves_and_retrieves_document(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    stored_document = document_repository.get_document(
        sample_document.id
    )

    assert stored_document is not None
    assert stored_document.id == sample_document.id
    assert stored_document.checksum == sample_document.checksum
    assert stored_document.document_type == "invoice"
    assert stored_document.metadata.contains_table is True


def test_retrieves_document_by_checksum(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    stored_document = document_repository.get_document_by_checksum(
        sample_document.checksum
    )

    assert stored_document is not None
    assert stored_document.id == sample_document.id


def test_blocks_document_with_duplicate_checksum(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    duplicate = sample_document.model_copy(
        update={"id": "invoice_001_copy"}
    )

    with pytest.raises(
        DuplicateDocumentError,
        match="invoice_001",
    ):
        document_repository.save_document(duplicate)


def test_updates_processing_status_and_error(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    document_repository.update_processing_status(
        document_id=sample_document.id,
        status=ProcessingStatus.FAILED,
        error="Test error",
    )

    stored_document = document_repository.get_document(
        sample_document.id
    )

    assert stored_document is not None
    assert stored_document.processing_status is ProcessingStatus.FAILED
    assert stored_document.processing_error == "Test error"


def test_saves_and_retrieves_model_artifact(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    artifact = ModelArtifact(
        id="invoice_001_clip",
        document_id=sample_document.id,
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="test-model",
        model_version="1",
        storage_path="vectors/invoice_001_clip.npy",
        dimensions=4,
    )

    document_repository.save_artifact(artifact)

    stored_artifacts = document_repository.get_artifacts(
        sample_document.id
    )

    assert len(stored_artifacts) == 1
    assert stored_artifacts[0].id == artifact.id
    assert stored_artifacts[0].model_name == "test-model"
    assert stored_artifacts[0].dimensions == 4


def test_find_artifacts_returns_only_matching_model(
    document_repository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    matching_artifact = ModelArtifact(
        id="clip-artifact",
        document_id=sample_document.id,
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="clip-model",
        storage_path="clip-artifact.npy",
        dimensions=512,
    )
    other_model_artifact = ModelArtifact(
        id="other-model-artifact",
        document_id=sample_document.id,
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="other-model",
        storage_path="other-model-artifact.npy",
        dimensions=512,
    )
    caption_artifact = ModelArtifact(
        id="caption-artifact",
        document_id=sample_document.id,
        artifact_type=ArtifactType.CAPTION,
        model_name="clip-model",
        content="generated caption",
    )

    document_repository.save_artifact(matching_artifact)
    document_repository.save_artifact(other_model_artifact)
    document_repository.save_artifact(caption_artifact)

    artifacts = document_repository.find_artifacts(
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="clip-model",
    )

    assert artifacts == [matching_artifact]


def test_find_artifacts_filters_by_document_type(
    document_repository,
    sample_document: DocumentRecord,
    sample_receipt_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)
    document_repository.save_document(sample_receipt_document)

    invoice_artifact = ModelArtifact(
        id="invoice-embedding",
        document_id=sample_document.id,
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="clip-model",
        storage_path="invoice-embedding.npy",
        dimensions=512,
    )
    receipt_artifact = ModelArtifact(
        id="receipt-embedding",
        document_id=sample_receipt_document.id,
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="clip-model",
        storage_path="receipt-embedding.npy",
        dimensions=512,
    )

    document_repository.save_artifact(invoice_artifact)
    document_repository.save_artifact(receipt_artifact)

    artifacts = document_repository.find_artifacts(
        artifact_type=ArtifactType.IMAGE_EMBEDDING,
        model_name="clip-model",
        document_type="invoice",
    )

    assert artifacts == [invoice_artifact]