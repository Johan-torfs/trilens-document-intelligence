from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field

from app.domain.ocr import OCRResult


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRecord(BaseModel):
    id: str
    original_filename: str
    stored_path: str
    checksum: str

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime_type: str
    document_type: str
    page_count: int = Field(default=1, gt=0)

    language: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: str | None = None

    ocr: OCRResult | None = None