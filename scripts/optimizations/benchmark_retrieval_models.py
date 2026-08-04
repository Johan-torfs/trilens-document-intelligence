from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.services.similarity import cosine_similarity
from app.strategies.clip_retrieval import ClipRetrievalStrategy
from scripts.optimizations.compare_clip_models import (
    SiglipRetrievalStrategy,
)


DATA_ROOT = Path("data/external")

QUERIES = {
    "receipt": "store receipt",
    "form": "scanned form with input fields",
    "financial_reports": "financial report",
    "scientific_articles": "scientific article",
    "laws_and_regulations": "law or regulation document",
    "government_tenders": "government tender document",
    "manuals": "instruction manual",
    "patents": "patent document",
}


def document_label(path: Path) -> str:
    if "cord" in path.parts:
        return "receipt"

    if "funsd" in path.parts:
        return "form"

    return path.parent.name


def load_documents() -> list[tuple[Path, str]]:
    paths = sorted(DATA_ROOT.rglob("*.png"))

    return [
        (path, document_label(path))
        for path in paths
    ]


def benchmark(name: str, strategy) -> None:
    documents = load_documents()

    image_embeddings = {}
    image_times = []

    for path, _ in documents:
        with Image.open(path) as source:
            image = source.convert("RGB")

        start = time.perf_counter()
        image_embeddings[path] = strategy.embed_image(image)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        image_times.append(
            (time.perf_counter() - start) * 1000
        )

    recall_at_1 = 0
    recall_at_3 = 0

    print(f"\n{name}")
    print("-" * len(name))

    for expected_label, query in QUERIES.items():
        text_embedding = strategy.embed_text(query)

        ranked = sorted(
            documents,
            key=lambda item: cosine_similarity(
                text_embedding,
                image_embeddings[item[0]],
            ),
            reverse=True,
        )

        top_labels = [
            label
            for _, label in ranked[:3]
        ]

        if top_labels[0] == expected_label:
            recall_at_1 += 1

        if expected_label in top_labels:
            recall_at_3 += 1

        print(
            f"{query:<32} "
            f"top-3={top_labels}"
        )

    query_count = len(QUERIES)

    print()
    print(
        f"Recall@1: {recall_at_1 / query_count:.3f}"
    )
    print(
        f"Recall@3: {recall_at_3 / query_count:.3f}"
    )
    print(
        f"Average image latency: "
        f"{np.mean(image_times):.1f} ms"
    )


def main() -> None:
    benchmark(
        "CLIP",
        ClipRetrievalStrategy(),
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    benchmark(
        "SigLIP",
        SiglipRetrievalStrategy(),
    )


if __name__ == "__main__":
    main()