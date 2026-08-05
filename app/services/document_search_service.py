from app.domain.search import SearchQuery, SearchResult
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.services.retrieval_service import RetrievalService


class DocumentSearchService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        document_repository: DocumentRepository,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._document_repository = document_repository

    def search(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        visual_matches = {
            m.document_id: m
            for m in self._retrieval_service.search(
                query=query.text,
                top_k=query.top_k,
                document_type=query.document_type,
            )
        }

        text_matches = {
            m.document_id: m
            for m in self._retrieval_service.search_text(
                query=query.text,
                top_k=query.top_k,
                document_type=query.document_type,
            )
        }

        fts_raw = self._document_repository.lexical_search(
            query=query.text,
            top_k=query.top_k,
        )
        max_fts = max((s for _, s in fts_raw), default=1.0)
        fts_matches = {
            doc_id: score / max_fts
            for doc_id, score in fts_raw
        }

        all_document_ids = (
            set(visual_matches)
            | set(text_matches)
            | set(fts_matches)
        )

        combined: list[SearchResult] = []

        for document_id in all_document_ids:
            document = self._document_repository.get_document(
                document_id
            )

            if document is None:
                continue

            visual = visual_matches.get(document_id)
            text = text_matches.get(document_id)

            visual_score = visual.score if visual else 0.0
            text_score = text.score if text else 0.0
            fts_score = fts_matches.get(document_id, 0.0)
            best_page = (
                (visual or text).best_page_number
                if (visual or text)
                else None
            )

            combined.append(
                SearchResult(
                    document_id=document.id,
                    score=visual_score,
                    rank=1,
                    stored_path=document.stored_path,
                    document_type=document.document_type,
                    page_number=best_page,
                    text_score=text_score,
                    fts_score=fts_score,
                )
            )

        combined.sort(
            key=lambda r: max(r.score, r.text_score, r.fts_score),
            reverse=True,
        )

        return [
            result.model_copy(update={"rank": index + 1})
            for index, result in enumerate(combined)
        ]