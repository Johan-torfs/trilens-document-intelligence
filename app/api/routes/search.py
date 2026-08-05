import logging

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pydantic import BaseModel, Field

from app.api.dependencies import PipelineDependency
from app.domain.search import SearchQuery
from app.services.document_intelligence_pipeline import (
    RankedDocument,
    SearchOutcome,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/search",
    tags=["search"],
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    document_type: str | None = None


class SearchResultResponse(BaseModel):
    document_id: str
    rank: int

    final_score: float
    visual_score: float
    text_score: float
    fts_score: float

    image_url: str
    document_type: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    document_type: str | None
    ranking_mode: str
    results: list[SearchResultResponse]
    duration_ms: float


@router.post(
    "",
    response_model=SearchResponse,
)
def search_documents(
    request: SearchRequest,
    pipeline: PipelineDependency,
) -> SearchResponse:
    search_query = SearchQuery(
        text=request.query.strip(),
        top_k=request.top_k,
        document_type=(
            request.document_type.strip()
            if request.document_type
            else None
        ),
    )

    try:
        outcome = pipeline.search(
            query=search_query,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Zoeken mislukt voor query %r",
            request.query,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="De zoekopdracht is onverwacht mislukt.",
        ) from error

    return _to_response(outcome)


def _to_response(
    outcome: SearchOutcome,
) -> SearchResponse:
    return SearchResponse(
        query=outcome.query.text,
        top_k=outcome.query.top_k,
        document_type=outcome.query.document_type,
        ranking_mode=outcome.ranking_mode,
        results=[
            _to_result_response(result)
            for result in outcome.results
        ],
        duration_ms=outcome.duration_ms,
    )


def _to_result_response(
    result: RankedDocument,
) -> SearchResultResponse:
    return SearchResultResponse(
        document_id=result.document_id,
        rank=result.rank,
        final_score=result.final_score,
        visual_score=result.visual_score,
        text_score=result.text_score,
        fts_score=result.fts_score,
        image_url=(
            f"/api/documents/{result.document_id}/image"
        ),
        document_type=result.document_type,
    )