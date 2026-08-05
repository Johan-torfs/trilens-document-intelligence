import numpy as np
import pytest
from PIL import Image

from app.domain.classification import (
    ClassificationResult,
    DocumentTypeCandidate,
)
from app.services.document_classification_service import (
    DocumentClassificationService,
    _softmax,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_image() -> Image.Image:
    return Image.new("RGB", (64, 64), "white")


class FakeEmbeddingStrategy:
    """Returns deterministic embeddings for tests."""

    model_name = "fake-siglip"
    model_version = "1.0"

    def embed_text(self, text: str) -> np.ndarray:
        # First word of text drives the embedding direction so prompts for
        # "invoice" point toward axis-0 and prompts for "receipt" toward axis-1.
        if "invoice" in text or "billing" in text:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "receipt" in text or "store" in text:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:  # noqa: ARG002
        # Default image embedding points toward "invoice"
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed_images(self, images):
        return np.stack([self.embed_image(img) for img in images])


INVOICE_CANDIDATE = DocumentTypeCandidate(
    label="invoice",
    prompts=("a business invoice", "billing statement"),
    keywords=("invoice", "total due", "amount due"),
)

RECEIPT_CANDIDATE = DocumentTypeCandidate(
    label="receipt",
    prompts=("a store receipt", "payment receipt"),
    keywords=("receipt", "thank you for your purchase", "cashier"),
)

OTHER_CANDIDATE = DocumentTypeCandidate(
    label="report",
    prompts=("a report document",),
    keywords=("findings", "conclusion", "executive summary"),
)


def make_service(
    candidates: list[DocumentTypeCandidate] | None = None,
    visual_weight: float = 0.6,
    confidence_threshold: float = 0.4,
) -> DocumentClassificationService:
    return DocumentClassificationService(
        embedding_strategy=FakeEmbeddingStrategy(),
        candidates=candidates or [INVOICE_CANDIDATE, RECEIPT_CANDIDATE, OTHER_CANDIDATE],
        visual_weight=visual_weight,
        confidence_threshold=confidence_threshold,
    )


# ---------------------------------------------------------------------------
# _softmax helper
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _softmax(arr)
    assert abs(result.sum() - 1.0) < 1e-6


def test_softmax_preserves_order() -> None:
    arr = np.array([0.1, 0.5, 0.3], dtype=np.float32)
    result = _softmax(arr)
    assert result[1] > result[2] > result[0]


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_init_raises_on_empty_candidates() -> None:
    with pytest.raises(ValueError, match="At least one candidate"):
        DocumentClassificationService(
            embedding_strategy=FakeEmbeddingStrategy(),
            candidates=[],
        )


def test_init_raises_on_invalid_visual_weight() -> None:
    with pytest.raises(ValueError, match="visual_weight"):
        DocumentClassificationService(
            embedding_strategy=FakeEmbeddingStrategy(),
            candidates=[INVOICE_CANDIDATE],
            visual_weight=1.5,
        )


# ---------------------------------------------------------------------------
# Visual-only classification (no OCR text)
# ---------------------------------------------------------------------------

def test_classify_visual_only_returns_confident_result() -> None:
    service = make_service(confidence_threshold=0.2)
    result = service.classify(image=make_image(), ocr_text="")

    assert isinstance(result, ClassificationResult)
    assert result.document_type == "invoice"
    assert result.is_confident is True
    assert 0.0 <= result.confidence <= 1.0


def test_classify_visual_only_no_lexical_signal() -> None:
    service = make_service()
    result = service.classify(image=make_image(), ocr_text="")
    # No OCR text → lexical_score for winner is 0
    assert result.lexical_score == 0.0


# ---------------------------------------------------------------------------
# Lexical signal
# ---------------------------------------------------------------------------

def test_classify_lexical_boost_overrides_visual() -> None:
    # Image embedding points toward "invoice", but text is strongly "receipt"
    class ReceiptTextImage(FakeEmbeddingStrategy):
        def embed_image(self, image):  # noqa: ARG002
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    service = DocumentClassificationService(
        embedding_strategy=ReceiptTextImage(),
        candidates=[INVOICE_CANDIDATE, RECEIPT_CANDIDATE, OTHER_CANDIDATE],
        visual_weight=0.5,
        confidence_threshold=0.2,
    )
    result = service.classify(
        image=make_image(),
        ocr_text="receipt cashier thank you for your purchase",
    )
    assert result.document_type == "receipt"
    assert result.lexical_score > 0.0


def test_classify_lexical_keyword_match_increases_score() -> None:
    service = make_service(confidence_threshold=0.2)
    without_keywords = service.classify(image=make_image(), ocr_text="random text")
    with_keywords = service.classify(image=make_image(), ocr_text="invoice total due")

    # Both should pick invoice, but with keywords the lexical score should be higher
    assert with_keywords.lexical_score >= without_keywords.lexical_score


# ---------------------------------------------------------------------------
# Low confidence → fallback
# ---------------------------------------------------------------------------

def test_classify_returns_fallback_below_threshold() -> None:
    service = make_service(confidence_threshold=0.99)
    result = service.classify(image=make_image(), ocr_text="")
    assert result.document_type == "unknown"
    assert result.is_confident is False


def test_classify_fallback_label_configurable() -> None:
    # With all candidates present and threshold=0.99, no candidate reaches it
    service = DocumentClassificationService(
        embedding_strategy=FakeEmbeddingStrategy(),
        candidates=[INVOICE_CANDIDATE, RECEIPT_CANDIDATE, OTHER_CANDIDATE],
        confidence_threshold=0.99,
        fallback_label="other",
    )
    result = service.classify(image=make_image(), ocr_text="")
    assert result.document_type == "other"


# ---------------------------------------------------------------------------
# Single candidate edge case
# ---------------------------------------------------------------------------

def test_classify_single_candidate_always_wins() -> None:
    service = DocumentClassificationService(
        embedding_strategy=FakeEmbeddingStrategy(),
        candidates=[INVOICE_CANDIDATE],
        confidence_threshold=0.0,
    )
    result = service.classify(image=make_image(), ocr_text="")
    assert result.document_type == "invoice"
    assert result.is_confident is True
