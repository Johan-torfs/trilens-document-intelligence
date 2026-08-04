from pydantic import BaseModel, Field


class OCRBoundingBox(BaseModel):
    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)


class OCRWord(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: OCRBoundingBox


class OCRPageResult(BaseModel):
    page_number: int = Field(gt=0)
    text: str
    words: list[OCRWord]
    mean_confidence: float = Field(ge=0.0, le=1.0)


class OCRResult(BaseModel):
    text: str
    pages: list[OCRPageResult]
    mean_confidence: float = Field(ge=0.0, le=1.0)

    @property
    def words(self) -> list[OCRWord]:
        return [
            word
            for page in self.pages
            for word in page.words
        ]