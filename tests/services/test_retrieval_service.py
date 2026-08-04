import numpy as np
import pytest
from PIL import Image

from app.repositories.vector_repository import VectorRepository
from app.services.retrieval_service import RetrievalService
from app.strategies.retrieval import RetrievalStrategy


class FakeRetrievalStrategy(RetrievalStrategy):
    @property
    def model_name(self) -> str:
        return "fake-retrieval-model"

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        if "invoice" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)

        return np.array([0.0, 1.0], dtype=np.float32)

    @staticmethod
    def calibrate_score(raw: float) -> float:
        return raw


@pytest.fixture
def service(
    vector_repository: VectorRepository,
) -> RetrievalService:
    return RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=vector_repository,
    )


def test_search_loads_and_ranks_stored_embeddings(
    service: RetrievalService,
    vector_repository: VectorRepository,
) -> None:
    vector_repository.save(
        artifact_id="invoice-embedding",
        vector=np.array([1.0, 0.0], dtype=np.float32),
    )
    vector_repository.save(
        artifact_id="landscape-embedding",
        vector=np.array([0.0, 1.0], dtype=np.float32),
    )

    results = service.search(
        query="an invoice",
        artifact_ids=[
            "landscape-embedding",
            "invoice-embedding",
        ],
    )

    assert results[0][0] == "invoice-embedding"
    assert results[0][1] == pytest.approx(1.0)


def test_search_respects_top_k(
    service: RetrievalService,
    vector_repository: VectorRepository,
) -> None:
    vector_repository.save(
        artifact_id="best-match",
        vector=np.array([1.0, 0.0], dtype=np.float32),
    )
    vector_repository.save(
        artifact_id="second-match",
        vector=np.array([0.8, 0.2], dtype=np.float32),
    )

    results = service.search(
        query="an invoice",
        artifact_ids=[
            "best-match",
            "second-match",
        ],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0][0] == "best-match"


def test_search_rejects_empty_query(
    service: RetrievalService,
) -> None:
    with pytest.raises(
        ValueError,
        match="zoekopdracht mag niet leeg zijn",
    ):
        service.search(
            query="   ",
            artifact_ids=[],
        )


def test_service_exposes_strategy_model_name(
    service: RetrievalService,
) -> None:
    assert service.model_name == "fake-retrieval-model"


def test_text_similarity_compares_clip_text_embeddings(
    tmp_path,
) -> None:
    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=VectorRepository(tmp_path),
    )

    same_topic_score = service.text_similarity(
        "invoice with product rows",
        "an invoice document",
    )

    different_topic_score = service.text_similarity(
        "invoice with product rows",
        "a landscape photograph",
    )

    assert same_topic_score == 1.0
    assert different_topic_score == 0.0


def test_text_similarity_returns_zero_for_empty_text(
    tmp_path,
) -> None:
    service = RetrievalService(
        strategy=FakeRetrievalStrategy(),
        vector_repository=VectorRepository(tmp_path),
    )

    assert service.text_similarity("", "an invoice") == 0.0
    assert service.text_similarity("invoice", "   ") == 0.0