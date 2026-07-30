import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel

from app.api.dependencies import (
    PipelineDependency,
    RUNTIME_DIR,
)
from app.services.document_factory import (
    create_document_from_bytes,
)
from app.services.document_intelligence_pipeline import (
    IndexDocumentOutcome,
)
from fastapi.responses import FileResponse


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
)

UPLOAD_DIR = RUNTIME_DIR / "uploads"


class IndexDocumentResponse(BaseModel):
    document_id: str
    document_type: str
    original_filename: str

    is_searchable: bool
    has_caption: bool
    fully_succeeded: bool
    reused_document: bool

    caption: str | None
    embedding_model: str | None
    caption_model: str | None

    embedding_error: str | None
    caption_error: str | None

    duration_ms: float


@router.post(
    "",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
)
def index_document(
    pipeline: PipelineDependency,
    file: UploadFile = File(...),
    document_type: str = Form(...),
) -> IndexDocumentResponse:
    try:
        file_bytes = file.file.read()

        document, image_path = create_document_from_bytes(
            filename=file.filename or "document",
            content_type=file.content_type,
            file_bytes=file_bytes,
            document_type=document_type,
            upload_dir=UPLOAD_DIR,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    try:
        outcome = pipeline.index_document(
            document=document,
            image_path=image_path,
        )

    except Exception as error:
        logger.exception(
            "Documentverwerking mislukt voor %s",
            document.id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="De documentverwerking is onverwacht mislukt.",
        ) from error

    return _to_response(outcome)


@router.get(
    "/{document_id}/image",
    response_class=FileResponse,
    responses={
        200: {
            "content": {
                "image/png": {},
                "image/jpeg": {},
            },
            "description": "De opgeslagen documentafbeelding.",
        },
        404: {
            "description": "Document of afbeeldingsbestand niet gevonden."
        },
    },
)
def get_document_image(
    document_id: str,
    pipeline: PipelineDependency,
) -> FileResponse:
    try:
        document, image_path = pipeline.get_document_file(
            document_id
        )

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    return FileResponse(
        path=image_path,
        media_type=document.mime_type,
        filename=document.original_filename,
        content_disposition_type="inline",
    )


def _to_response(
    outcome: IndexDocumentOutcome,
) -> IndexDocumentResponse:
    return IndexDocumentResponse(
        document_id=outcome.document.id,
        document_type=outcome.document.document_type,
        original_filename=(
            outcome.document.original_filename
        ),
        is_searchable=outcome.is_searchable,
        has_caption=outcome.has_caption,
        fully_succeeded=outcome.fully_succeeded,
        reused_document=outcome.reused_document,
        caption=(
            outcome.caption_artifact.content
            if outcome.caption_artifact
            else None
        ),
        embedding_model=(
            outcome.embedding_artifact.model_name
            if outcome.embedding_artifact
            else None
        ),
        caption_model=(
            outcome.caption_artifact.model_name
            if outcome.caption_artifact
            else None
        ),
        embedding_error=outcome.embedding_error,
        caption_error=outcome.caption_error,
        duration_ms=outcome.duration_ms,
    )