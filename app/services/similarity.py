import numpy as np


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError("Embeddings moeten eendimensionaal zijn.")

    if first.shape != second.shape:
        raise ValueError(
            "Embeddings moeten dezelfde dimensies hebben."
        )

    first_norm = np.linalg.norm(first)
    second_norm = np.linalg.norm(second)

    if first_norm == 0 or second_norm == 0:
        raise ValueError("Een nulvector kan niet worden vergeleken.")

    return float(
        np.dot(first, second) / (first_norm * second_norm)
    )


def rank_by_similarity(
    query_embedding: np.ndarray,
    document_embeddings: dict[str, np.ndarray],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    if top_k <= 0:
        raise ValueError("top_k moet groter zijn dan nul.")

    ranked = [
        (
            document_id,
            cosine_similarity(query_embedding, embedding),
        )
        for document_id, embedding in document_embeddings.items()
    ]

    ranked.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return ranked[:top_k]