from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    document_type: str
    language: str | None = None
    contains_table: bool | None = None
    contains_signature: bool | None = None
    contains_logo: bool | None = None
    contains_portrait: bool | None = None
    product_row_count: int | None = None


class DocumentRecord(BaseModel):
    id: str
    original_filename: str
    stored_path: str
    checksum: str

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str
    document_type: str

    caption: str | None = None
    retrieval_model: str | None = None
    caption_model: str | None = None
    embedding_path: str | None = None

    metadata: DocumentMetadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: str | None = None


class ArtifactType(StrEnum):
    IMAGE_EMBEDDING = "image_embedding"
    TEXT_EMBEDDING = "text_embedding"
    CAPTION = "caption"
    ANALYSIS = "analysis"


class ModelArtifact(BaseModel):
    id: str
    document_id: str
    artifact_type: ArtifactType

    model_name: str
    model_version: str | None = None

    storage_path: str | None = None
    content: str | None = None
    dimensions: int | None = Field(default=None, gt=0)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )