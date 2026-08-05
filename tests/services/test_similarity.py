import numpy as np
import pytest

from app.services.similarity import cosine_similarity


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