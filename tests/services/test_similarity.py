import numpy as np
import pytest

from app.services.similarity import cosine_similarity, rank_by_similarity


def test_identical_vectors_have_similarity_one() -> None:
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    similarity = cosine_similarity(vector, vector)

    assert similarity == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)

    similarity = cosine_similarity(first, second)

    assert similarity == pytest.approx(0.0)


def test_different_dimensions_are_rejected() -> None:
    first = np.array([1.0, 2.0], dtype=np.float32)
    second = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError):
        cosine_similarity(first, second)


def test_rank_by_similarity_returns_best_match_first() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)

    documents = {
        "unrelated": np.array([0.0, 1.0], dtype=np.float32),
        "related": np.array([0.9, 0.1], dtype=np.float32),
        "opposite": np.array([-1.0, 0.0], dtype=np.float32),
    }

    results = rank_by_similarity(
        query_embedding=query,
        document_embeddings=documents,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0][0] == "related"
    assert results[1][0] == "unrelated"


def test_rank_by_similarity_rejects_invalid_top_k() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        rank_by_similarity(
            query_embedding=query,
            document_embeddings={},
            top_k=0,
        )