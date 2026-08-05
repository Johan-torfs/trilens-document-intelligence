import numpy as np
from PIL import Image

from app.domain.classification import (
    ClassificationResult,
    DocumentTypeCandidate,
)
from app.strategies.embedding import EmbeddingStrategy


def _softmax(arr: np.ndarray) -> np.ndarray:
    shifted = arr - arr.max()
    exp_arr = np.exp(shifted)
    return exp_arr / exp_arr.sum()


class DocumentClassificationService:
    def __init__(
        self,
        embedding_strategy: EmbeddingStrategy,
        candidates: list[DocumentTypeCandidate],
        visual_weight: float = 0.6,
        confidence_threshold: float = 0.4,
        fallback_label: str = "unknown",
    ) -> None:
        if not candidates:
            raise ValueError("At least one candidate is required.")
        if not 0.0 <= visual_weight <= 1.0:
            raise ValueError("visual_weight must be in [0, 1].")

        self._strategy = embedding_strategy
        self._candidates = candidates
        self._visual_weight = visual_weight
        self._lexical_weight = 1.0 - visual_weight
        self._confidence_threshold = confidence_threshold
        self._fallback_label = fallback_label

        # Precompute mean prompt embedding per type (once at startup)
        self._prompt_embeddings: list[np.ndarray] = [
            self._mean_prompt_embedding(candidate.prompts)
            for candidate in candidates
        ]

    def _mean_prompt_embedding(
        self,
        prompts: tuple[str, ...],
    ) -> np.ndarray:
        embeddings = np.stack([
            self._strategy.embed_text(prompt)
            for prompt in prompts
        ])
        mean = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 1e-8 else mean

    def _visual_scores(self, image: Image.Image) -> np.ndarray:
        image_emb = self._strategy.embed_image(image)
        raw = np.array(
            [float(np.dot(image_emb, pe)) for pe in self._prompt_embeddings],
            dtype=np.float32,
        )
        return _softmax(raw)

    def _lexical_scores(self, ocr_text: str) -> np.ndarray:
        lowered = ocr_text.lower()
        counts = np.array(
            [
                sum(1 for kw in c.keywords if kw in lowered)
                for c in self._candidates
            ],
            dtype=np.float32,
        )
        total = counts.sum()
        if total == 0.0:
            return counts  # no lexical signal → all zeros
        return _softmax(counts)

    def classify(
        self,
        image: Image.Image,
        ocr_text: str,
    ) -> ClassificationResult:
        visual = self._visual_scores(image)
        lexical = self._lexical_scores(ocr_text)

        has_lexical = lexical.sum() > 0.0

        if has_lexical:
            combined = self._visual_weight * visual + self._lexical_weight * lexical
        else:
            combined = visual

        best_idx = int(combined.argmax())
        confidence = float(combined[best_idx])
        is_confident = confidence >= self._confidence_threshold

        return ClassificationResult(
            document_type=(
                self._candidates[best_idx].label
                if is_confident
                else self._fallback_label
            ),
            confidence=confidence,
            visual_score=float(visual[best_idx]),
            lexical_score=float(lexical[best_idx]),
            is_confident=is_confident,
        )
