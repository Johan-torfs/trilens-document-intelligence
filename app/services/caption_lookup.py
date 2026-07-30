from app.domain.document import ArtifactType, ModelArtifact


def find_caption(
    artifacts: list[ModelArtifact],
    preferred_model_name: str | None = None,
) -> str | None:
    captions = [
        artifact
        for artifact in artifacts
        if (
            artifact.artifact_type == ArtifactType.CAPTION
            and artifact.content
        )
    ]

    if preferred_model_name is not None:
        for artifact in captions:
            if artifact.model_name == preferred_model_name:
                return artifact.content

    if captions:
        return captions[-1].content

    return None