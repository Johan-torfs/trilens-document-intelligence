import logging

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import BaseModel, Field

from app.api.dependencies import PipelineDependency
from app.services.analysis_service import (
    AnalysisDisabledError,
)
from app.services.document_intelligence_pipeline import (
    AnalyzeDocumentOutcome,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["analysis"],
)


class AnalyzeDocumentRequest(BaseModel):
    question: str = Field(min_length=1)


class AnalyzeDocumentResponse(BaseModel):
    document_id: str
    question: str
    text: str

    source: str
    used_fallback: bool

    model_name: str
    model_version: str | None

    model_duration_ms: float
    total_duration_ms: float


@router.post(
    "/{document_id}/analysis",
    response_model=AnalyzeDocumentResponse,
)
def analyze_document(
    document_id: str,
    request: AnalyzeDocumentRequest,
    pipeline: PipelineDependency,
) -> AnalyzeDocumentResponse:
    try:
        outcome = pipeline.analyze_document(
            document_id=document_id,
            question=request.question,
        )

    except AnalysisDisabledError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Analyse mislukt voor document %s",
            document_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="De documentanalyse is onverwacht mislukt.",
        ) from error

    return _to_response(outcome)


def _to_response(
    outcome: AnalyzeDocumentOutcome,
) -> AnalyzeDocumentResponse:
    return AnalyzeDocumentResponse(
        document_id=outcome.document.id,
        question=outcome.question,
        text=outcome.analysis.text,
        source=outcome.analysis.source,
        used_fallback=outcome.used_fallback,
        model_name=outcome.analysis.model_name,
        model_version=outcome.analysis.model_version,
        model_duration_ms=outcome.analysis.duration_ms,
        total_duration_ms=outcome.duration_ms,
    )