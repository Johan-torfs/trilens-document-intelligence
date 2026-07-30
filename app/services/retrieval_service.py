import numpy as np

from app.repositories.vector_repository import VectorRepository
from app.services.similarity import (
    cosine_similarity,
    rank_by_similarity,
)
from app.strategies.retrieval import RetrievalStrategy


class RetrievalService:
    def __init__(
        self,
        strategy: RetrievalStrategy,
        vector_repository: VectorRepository,
    ) -> None:
        self._strategy = strategy
        self._vector_repository = vector_repository

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    def search(
        self,
        query: str,
        artifact_ids: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("De zoekopdracht mag niet leeg zijn.")

        query_embedding = self._strategy.embed_text(cleaned_query)

        document_embeddings: dict[str, np.ndarray] = {
            artifact_id: self._vector_repository.load(artifact_id)
            for artifact_id in artifact_ids
        }

        return rank_by_similarity(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
            top_k=top_k,
        )

    def text_similarity(
        self,
        first_text: str,
        second_text: str,
    ) -> float:
        cleaned_first_text = first_text.strip()
        cleaned_second_text = second_text.strip()

        if not cleaned_first_text or not cleaned_second_text:
            return 0.0

        first_embedding = self._strategy.embed_text(
            cleaned_first_text
        )
        second_embedding = self._strategy.embed_text(
            cleaned_second_text
        )

        return cosine_similarity(
            first_embedding,
            second_embedding,
        )