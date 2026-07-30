from datetime import datetime, timezone

from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    id: str
    query: str = Field(min_length=1)
    relevant_document_ids: list[str]
    top_k: int = Field(default=3, ge=1)


class EvaluationRun(BaseModel):
    id: str
    model_name: str
    model_version: str | None = None

    total_cases: int = Field(ge=0)
    recall_at_1: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    average_query_time_ms: float = Field(ge=0)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )