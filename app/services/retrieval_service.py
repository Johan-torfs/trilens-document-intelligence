from dataclasses import dataclass

from app.repositories.vector_repository import VectorRepository
from app.services.similarity import cosine_similarity
from app.strategies.embedding import EmbeddingStrategy


VISUAL_VECTOR_NAME = "visual"
VISUAL_UNIT_TYPE = "page"
TEXT_VECTOR_NAME = "text"
TEXT_UNIT_TYPE = "chunk"

# Current document scope is a maximum of roughly ten pages.
MAX_PAGES_PER_DOCUMENT = 10
MAX_CHUNKS_PER_DOCUMENT = 20
MAX_EVIDENCE_PAGES = 3


@dataclass(frozen=True)
class PageSearchMatch:
    point_id: str
    document_id: str
    page_number: int
    score: float


@dataclass(frozen=True)
class DocumentSearchMatch:
    document_id: str
    score: float
    best_page_number: int
    pages: tuple[PageSearchMatch, ...]


class RetrievalService:
    def __init__(
        self,
        strategy: EmbeddingStrategy,
        vector_repository: VectorRepository,
        text_strategy: EmbeddingStrategy | None = None,
    ) -> None:
        self._strategy = strategy
        self._vector_repository = vector_repository
        # separate strategy for text–text embedding; falls back to strategy if None
        self._text_strategy = text_strategy

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    @property
    def model_version(self) -> str | None:
        return self._strategy.model_version

    @property
    def text_model_name(self) -> str:
        return (self._text_strategy or self._strategy).model_name

    @property
    def text_model_version(self) -> str | None:
        return (self._text_strategy or self._strategy).model_version

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_type: str | None = None,
    ) -> list[DocumentSearchMatch]:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "De zoekopdracht mag niet leeg zijn."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k moet groter zijn dan nul."
            )

        query_embedding = self._strategy.embed_text(
            cleaned_query
        )

        filters = {
            "unit_type": VISUAL_UNIT_TYPE,
            "vector_type": VISUAL_VECTOR_NAME,
            "model_name": self.model_name,
            "model_version": self.model_version or "",
        }

        if document_type is not None:
            filters["document_type"] = document_type

        page_limit = top_k * MAX_PAGES_PER_DOCUMENT

        point_results = self._vector_repository.search(
            vector_name=VISUAL_VECTOR_NAME,
            query_vector=query_embedding,
            limit=page_limit,
            filters=filters,
        )

        pages_by_document: dict[
            str,
            list[PageSearchMatch],
        ] = {}

        for point in point_results:
            document_id = point.payload.get("document_id")
            page_number = point.payload.get("page_number")

            if not isinstance(document_id, str):
                continue

            if not isinstance(page_number, int):
                continue

            page_match = PageSearchMatch(
                point_id=point.point_id,
                document_id=document_id,
                page_number=page_number,
                score=point.score,
            )

            pages_by_document.setdefault(
                document_id,
                [],
            ).append(page_match)

        document_matches: list[DocumentSearchMatch] = []

        for document_id, pages in pages_by_document.items():
            ranked_pages = sorted(
                pages,
                key=lambda page: page.score,
                reverse=True,
            )

            best_page = ranked_pages[0]

            document_matches.append(
                DocumentSearchMatch(
                    document_id=document_id,
                    score=best_page.score,
                    best_page_number=best_page.page_number,
                    pages=tuple(
                        ranked_pages[:MAX_EVIDENCE_PAGES]
                    ),
                )
            )

        document_matches.sort(
            key=lambda match: match.score,
            reverse=True,
        )

        return document_matches[:top_k]

    def text_similarity(
        self,
        first_text: str,
        second_text: str,
    ) -> float:
        cleaned_first = first_text.strip()
        cleaned_second = second_text.strip()

        if not cleaned_first or not cleaned_second:
            return 0.0

        first_embedding = self._strategy.embed_text(
            cleaned_first
        )
        second_embedding = self._strategy.embed_text(
            cleaned_second
        )

        return cosine_similarity(
            first_embedding,
            second_embedding,
        )

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        document_type: str | None = None,
    ) -> list[DocumentSearchMatch]:
        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "De zoekopdracht mag niet leeg zijn."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k moet groter zijn dan nul."
            )

        query_embedding = (self._text_strategy or self._strategy).embed_text(
            cleaned_query
        )

        filters: dict[str, str] = {
            "unit_type": TEXT_UNIT_TYPE,
            "vector_type": TEXT_VECTOR_NAME,
            "model_name": self.text_model_name,
            "model_version": self.text_model_version or "",
        }

        if document_type is not None:
            filters["document_type"] = document_type

        chunk_limit = top_k * MAX_CHUNKS_PER_DOCUMENT

        point_results = self._vector_repository.search(
            vector_name=TEXT_VECTOR_NAME,
            query_vector=query_embedding,
            limit=chunk_limit,
            filters=filters,
        )

        chunks_by_document: dict[
            str,
            list[PageSearchMatch],
        ] = {}

        for point in point_results:
            document_id = point.payload.get("document_id")
            page_number = point.payload.get("page_number")

            if not isinstance(document_id, str):
                continue

            if not isinstance(page_number, int):
                continue

            chunk_match = PageSearchMatch(
                point_id=point.point_id,
                document_id=document_id,
                page_number=page_number,
                score=point.score,
            )

            chunks_by_document.setdefault(
                document_id,
                [],
            ).append(chunk_match)

        document_matches: list[DocumentSearchMatch] = []

        for document_id, chunks in chunks_by_document.items():
            ranked_chunks = sorted(
                chunks,
                key=lambda c: c.score,
                reverse=True,
            )

            best_chunk = ranked_chunks[0]

            document_matches.append(
                DocumentSearchMatch(
                    document_id=document_id,
                    score=best_chunk.score,
                    best_page_number=best_chunk.page_number,
                    pages=tuple(
                        ranked_chunks[:MAX_EVIDENCE_PAGES]
                    ),
                )
            )

        document_matches.sort(
            key=lambda match: match.score,
            reverse=True,
        )

        return document_matches[:top_k]