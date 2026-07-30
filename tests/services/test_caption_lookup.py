from app.domain.document import ArtifactType, ModelArtifact
from app.services.caption_lookup import find_caption


def test_find_caption_returns_preferred_model_caption() -> None:
    artifacts = [
        ModelArtifact(
            id="caption-old",
            document_id="document-001",
            artifact_type=ArtifactType.CAPTION,
            model_name="old-model",
            content="old generated caption",
        ),
        ModelArtifact(
            id="caption-blip",
            document_id="document-001",
            artifact_type=ArtifactType.CAPTION,
            model_name="blip-model",
            content="generated BLIP caption",
        ),
    ]

    caption = find_caption(
        artifacts,
        preferred_model_name="blip-model",
    )

    assert caption == "generated BLIP caption"


def test_find_caption_returns_none_when_caption_is_missing() -> None:
    artifacts = [
        ModelArtifact(
            id="image-embedding",
            document_id="document-001",
            artifact_type=ArtifactType.IMAGE_EMBEDDING,
            model_name="clip-model",
            storage_path="vector.npy",
            dimensions=512,
        )
    ]

    assert find_caption(artifacts) is None


def test_find_caption_ignores_empty_caption() -> None:
    artifacts = [
        ModelArtifact(
            id="empty-caption",
            document_id="document-001",
            artifact_type=ArtifactType.CAPTION,
            model_name="blip-model",
            content="",
        )
    ]

    assert find_caption(artifacts) is None