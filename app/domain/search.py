from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    document_type: str | None = None


class SearchResult(BaseModel):
    document_id: str
    score: float
    rank: int = Field(ge=1)

    stored_path: str
    document_type: str
    page_number: int | None = None
    text_score: float = 0.0
    fts_score: float = 0.0