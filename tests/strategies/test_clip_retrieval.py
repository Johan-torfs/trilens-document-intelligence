import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from app.strategies.clip_embedding import ClipEmbeddingStrategy
from app.preprocessing.pipeline import preprocess_image

pytestmark = pytest.mark.model_integration


@pytest.fixture(scope="module")
def strategy() -> ClipEmbeddingStrategy:
    return ClipEmbeddingStrategy()


def test_image_embedding_is_normalized_vector(
    strategy: ClipEmbeddingStrategy,
) -> None:
    image = Image.new("RGB", (400, 300), color="white")

    embedding = strategy.embed_image(image)

    assert embedding.ndim == 1
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(
        1.0,
        abs=1e-5,
    )


def test_text_embedding_is_normalized_vector(
    strategy: ClipEmbeddingStrategy,
) -> None:
    embedding = strategy.embed_text("an invoice")

    assert embedding.ndim == 1
    assert embedding.dtype == np.float32
    assert np.linalg.norm(embedding) == pytest.approx(
        1.0,
        abs=1e-5,
    )


def test_image_and_text_embeddings_have_same_dimensions(
    strategy: ClipEmbeddingStrategy,
) -> None:
    image = Image.new("RGB", (400, 300), color="white")

    image_embedding = strategy.embed_image(image)
    text_embedding = strategy.embed_text("a white document")

    assert image_embedding.shape == text_embedding.shape


def test_identical_texts_have_similarity_one(
    strategy: ClipEmbeddingStrategy,
) -> None:
    first_embedding = strategy.embed_text("an invoice")
    second_embedding = strategy.embed_text("an invoice")

    similarity = float(
        np.dot(first_embedding, second_embedding)
    )

    assert similarity == pytest.approx(1.0, abs=1e-5)


@pytest.mark.skip(reason="Requires generated test image not present in repo")
def test_invoice_matches_invoice_query_better_than_landscape_query(
    strategy: ClipEmbeddingStrategy,
) -> None:
    image_path = Path(
        "data/generated/invoices/invoice_001.png"
    )

    assert image_path.exists(), (
        f"Testafbeelding ontbreekt: {image_path}"
    )

    preprocessing_result = preprocess_image(image_path)

    image_embedding = strategy.embed_image(
        preprocessing_result.image
    )
    invoice_embedding = strategy.embed_text(
        "an invoice document with product rows and totals"
    )
    landscape_embedding = strategy.embed_text(
        "a mountain landscape with trees and a lake"
    )

    invoice_similarity = float(
        np.dot(image_embedding, invoice_embedding)
    )
    landscape_similarity = float(
        np.dot(image_embedding, landscape_embedding)
    )

    assert invoice_similarity > landscape_similarity