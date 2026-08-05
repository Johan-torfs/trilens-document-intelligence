from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from app.strategies.embedding import EmbeddingStrategy


SENTENCE_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class SentenceEmbeddingStrategy(EmbeddingStrategy):
    """Text-only embedding strategy for text–text semantic similarity.

    Uses mean-pooled sentence embeddings with up to 512 tokens,
    making it suitable for OCR chunk indexing where SigLIP's
    64-token limit causes failures on longer text blocks.
    """

    def __init__(
        self,
        model_name: str = SENTENCE_MODEL_NAME,
        model_version: str | None = None,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._device = self._resolve_device(device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_version,
        )

        self._model = AutoModel.from_pretrained(
            model_name,
            revision=model_version,
        )

        self._model.to(self._device)
        self._model.eval()

        self._model_version = (
            model_version
            or getattr(
                self._model.config,
                "_commit_hash",
                None,
            )
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def embed_images(
        self,
        images: Sequence[Image.Image],
    ) -> np.ndarray:
        raise NotImplementedError(
            "SentenceEmbeddingStrategy is text-only "
            "and does not support image embedding."
        )

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        cleaned = text.strip()

        if not cleaned:
            raise ValueError("Text may not be empty.")

        inputs = self._tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
            padding=True,
        )

        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        # Mean-pool over non-padding tokens
        mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        sum_embeddings = torch.sum(
            token_embeddings * mask_expanded, dim=1
        )
        sum_mask = torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9
        )
        mean_pooled = sum_embeddings / sum_mask

        normalized = torch.nn.functional.normalize(
            mean_pooled, p=2, dim=-1
        )

        return (
            normalized[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    @staticmethod
    def _resolve_device(
        requested: str | None,
    ) -> torch.device:
        if requested is not None:
            return torch.device(requested)

        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")
