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
from app.domain.prepared_document import DocumentSource
from app.services.document_intelligence_pipeline import (
    IndexDocumentOutcome,
)
from fastapi.responses import FileResponse


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"],
)


class IndexDocumentResponse(BaseModel):
    document_id: str
    document_type: str
    original_filename: str

    is_searchable: bool
    has_ocr: bool
    fully_succeeded: bool
    reused_document: bool

    indexing_error: str | None
    ocr_error: str | None

    classification_confidence: float | None
    duration_ms: float


@router.post(
    "",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_200_OK,
)
def index_document(
    pipeline: PipelineDependency,
    file: UploadFile = File(...),
    document_type: str | None = Form(default=None),
) -> IndexDocumentResponse:
    try:
        file_bytes = file.file.read()

        if not file_bytes:
            raise ValueError("Het geüploade bestand is leeg.")

        explicit_type: str | None = None
        if document_type is not None:
            cleaned = document_type.strip()
            if cleaned:
                explicit_type = cleaned

        source = DocumentSource(
            filename=file.filename or "document",
            mime_type=file.content_type
            or "application/octet-stream",
            content=file_bytes,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    try:
        outcome = pipeline.index_document(
            source=source,
            document_type=explicit_type,
        )

    except Exception as error:
        logger.exception(
            "Documentverwerking mislukt voor %s",
            file.filename,
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
        has_ocr=outcome.has_ocr,
        fully_succeeded=outcome.fully_succeeded,
        reused_document=outcome.reused_document,
        indexing_error=outcome.indexing_error,
        ocr_error=outcome.ocr_error,
        classification_confidence=outcome.classification_confidence,
        duration_ms=outcome.duration_ms,
    )