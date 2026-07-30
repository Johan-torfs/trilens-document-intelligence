from unittest.mock import create_autospec

from PIL import Image

from app.domain.document import ArtifactType, ModelArtifact
from app.repositories.document_repository import DocumentRepository
from app.services.caption_service import CaptionService
from app.services.document_caption_service import (
    DocumentCaptionService,
)
from app.strategies.caption import CaptionOptions, CaptionStrategy


class FakeCaptionStrategy(CaptionStrategy):
    def __init__(self) -> None:
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return "fake-caption-model"

    @property
    def model_version(self) -> str | None:
        return "version-1"

    def generate_caption(
        self,
        image: Image.Image,
        options: CaptionOptions,
    ) -> str:
        self.call_count += 1
        return "an automatically generated document caption"


def test_caption_document_stores_caption_artifact() -> None:
    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    document_repository.get_artifacts.return_value = []

    strategy = FakeCaptionStrategy()

    service = DocumentCaptionService(
        caption_service=CaptionService(strategy),
        document_repository=document_repository,
    )

    artifact = service.caption_document(
        document_id="document-001",
        image=Image.new("RGB", (100, 100), "white"),
    )

    assert artifact.document_id == "document-001"
    assert artifact.artifact_type == ArtifactType.CAPTION
    assert artifact.model_name == "fake-caption-model"
    assert artifact.model_version == "version-1"
    assert (
        artifact.content
        == "an automatically generated document caption"
    )
    assert strategy.call_count == 1

    document_repository.save_artifact.assert_called_once_with(
        artifact
    )


def test_caption_document_uses_cached_artifact() -> None:
    cached_artifact = ModelArtifact(
        id="cached-caption",
        document_id="document-001",
        artifact_type=ArtifactType.CAPTION,
        model_name="fake-caption-model",
        model_version="version-1",
        content="an existing generated caption",
    )

    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    document_repository.get_artifacts.return_value = [
        cached_artifact
    ]

    strategy = FakeCaptionStrategy()

    service = DocumentCaptionService(
        caption_service=CaptionService(strategy),
        document_repository=document_repository,
    )

    artifact = service.caption_document(
        document_id="document-001",
        image=Image.new("RGB", (100, 100), "white"),
    )

    assert artifact == cached_artifact
    assert strategy.call_count == 0
    document_repository.save_artifact.assert_not_called()


def test_caption_document_regenerates_for_different_model_version() -> None:
    cached_artifact = ModelArtifact(
        id="old-caption",
        document_id="document-001",
        artifact_type=ArtifactType.CAPTION,
        model_name="fake-caption-model",
        model_version="version-0",
        content="an old generated caption",
    )

    document_repository = create_autospec(
        DocumentRepository,
        instance=True,
    )
    document_repository.get_artifacts.return_value = [
        cached_artifact
    ]

    strategy = FakeCaptionStrategy()

    service = DocumentCaptionService(
        caption_service=CaptionService(strategy),
        document_repository=document_repository,
    )

    artifact = service.caption_document(
        document_id="document-001",
        image=Image.new("RGB", (100, 100), "white"),
    )

    assert artifact.id != cached_artifact.id
    assert artifact.model_version == "version-1"
    assert strategy.call_count == 1

    document_repository.save_artifact.assert_called_once_with(
        artifact
    )