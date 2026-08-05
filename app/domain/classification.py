from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentTypeCandidate:
    label: str
    prompts: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationResult:
    document_type: str
    confidence: float
    visual_score: float
    lexical_score: float
    is_confident: bool
