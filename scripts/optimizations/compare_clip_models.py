from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.services.similarity import cosine_similarity
from app.strategies.clip_retrieval import ClipRetrievalStrategy


SIGLIP_MODEL_NAME = "google/siglip-base-patch16-384"

QUERIES = [
    "invoice",
    "business invoice",
    "factuur",
    "zakelijke factuur",
    "toilet paper",
    "a very fancy chair",
]

POSITIVE_QUERIES = {
    "invoice",
    "business invoice",
    "factuur",
    "zakelijke factuur",
}


class RetrievalStrategy(Protocol):
    @property
    def model_name(self) -> str:
        ...

    def embed_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        ...

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        ...


class SiglipRetrievalStrategy:
    def __init__(
        self,
        model_name: str = SIGLIP_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name

        self._device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        dtype = (
            torch.float16
            if self._device.type == "cuda"
            else torch.float32
        )

        self._processor = AutoProcessor.from_pretrained(
            model_name
        )

        self._model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self._model.to(self._device)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        inputs = self._processor(
            images=image.convert("RGB"),
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model.get_image_features(
                **inputs
            )

        features = self._extract_features(outputs)

        return self._normalize(features[0])

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        cleaned_text = text.strip().lower()

        if not cleaned_text:
            raise ValueError("Text may not be empty.")

        inputs = self._processor(
            text=[cleaned_text],
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model.get_text_features(
                **inputs
            )

        features = self._extract_features(outputs)

        return self._normalize(features[0])

    @staticmethod
    def _extract_features(
        outputs: object,
    ) -> torch.Tensor:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output

        if isinstance(outputs, torch.Tensor):
            return outputs

        raise TypeError(
            "Model returned an unsupported feature type: "
            f"{type(outputs).__name__}"
        )

    @staticmethod
    def _normalize(
        features: torch.Tensor,
    ) -> np.ndarray:
        normalized = torch.nn.functional.normalize(
            features.float(),
            p=2,
            dim=-1,
        )

        return (
            normalized
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current CLIP model with SigLIP 2."
        )
    )

    parser.add_argument(
        "image_path",
        type=Path,
    )

    return parser.parse_args()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_strategy(
    strategy: RetrievalStrategy,
    image: Image.Image,
) -> None:
    synchronize()

    image_start = time.perf_counter()
    image_embedding = strategy.embed_image(image)
    synchronize()

    image_duration_ms = (
        time.perf_counter() - image_start
    ) * 1000

    scores: dict[str, float] = {}
    text_durations: list[float] = []

    for query in QUERIES:
        synchronize()
        text_start = time.perf_counter()

        text_embedding = strategy.embed_text(query)

        synchronize()
        text_durations.append(
            (time.perf_counter() - text_start) * 1000
        )

        scores[query] = cosine_similarity(
            text_embedding,
            image_embedding,
        )

    positive_scores = [
        score
        for query, score in scores.items()
        if query in POSITIVE_QUERIES
    ]

    negative_scores = [
        score
        for query, score in scores.items()
        if query not in POSITIVE_QUERIES
    ]

    best_positive_margin = (
        max(positive_scores)
        - max(negative_scores)
    )

    all_positive_margin = (
        min(positive_scores)
        - max(negative_scores)
    )

    mean_positive_margin = (
        float(np.mean(positive_scores))
        - max(negative_scores)
    )

    print()
    print(strategy.model_name)
    print("-" * len(strategy.model_name))

    for query, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        label = (
            "relevant"
            if query in POSITIVE_QUERIES
            else "unrelated"
        )

        print(
            f"{query:<24} "
            f"{score:>8.4f} "
            f"{label}"
        )

    print()
    print(
        f"Best positive margin: "
        f"{best_positive_margin:.4f}"
    )

    print(
        f"Mean positive margin: "
        f"{mean_positive_margin:.4f}"
    )

    print(
        f"All-positive separation: "
        f"{all_positive_margin:.4f}"
    )
    print(
        f"Image embedding latency: "
        f"{image_duration_ms:.1f} ms"
    )
    print(
        f"Average text latency: "
        f"{np.mean(text_durations):.1f} ms"
    )
    print(
        f"Embedding dimensions: "
        f"{image_embedding.shape[0]}"
    )


def load_strategy(
    factory,
) -> tuple[RetrievalStrategy, float]:
    synchronize()
    start = time.perf_counter()

    strategy = factory()

    synchronize()

    duration_ms = (
        time.perf_counter() - start
    ) * 1000

    return strategy, duration_ms


def main() -> None:
    args = parse_args()

    if not args.image_path.is_file():
        raise FileNotFoundError(
            f"Image does not exist: {args.image_path}"
        )

    with Image.open(args.image_path) as source:
        image = source.convert("RGB")

    print(f"Image: {args.image_path}")
    print(f"Image size: {image.size}")
    print(
        "Device:",
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    clip, clip_load_ms = load_strategy(
        ClipRetrievalStrategy
    )

    print(
        f"\nCurrent CLIP load time: "
        f"{clip_load_ms:.1f} ms"
    )

    benchmark_strategy(
        strategy=clip,
        image=image,
    )

    del clip

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    siglip, siglip_load_ms = load_strategy(
        SiglipRetrievalStrategy
    )

    print(
        f"\nSigLIP load time: "
        f"{siglip_load_ms:.1f} ms"
    )

    benchmark_strategy(
        strategy=siglip,
        image=image,
    )


if __name__ == "__main__":
    main()