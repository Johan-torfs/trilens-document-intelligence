from app.domain.document import ArtifactType, ModelArtifact
from app.domain.search import SearchQuery, SearchResult
from app.repositories.document_repository import DocumentRepository
from app.services.caption_lookup import find_caption
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
        embedding_artifacts = (
            self._document_repository.find_artifacts(
                artifact_type=ArtifactType.IMAGE_EMBEDDING,
                model_name=self._retrieval_service.model_name,
                document_type=query.document_type,
            )
        )

        artifacts_by_id: dict[str, ModelArtifact] = {
            artifact.id: artifact
            for artifact in embedding_artifacts
        }

        ranked_artifacts = self._retrieval_service.search(
            query=query.text,
            artifact_ids=list(artifacts_by_id),
            top_k=query.top_k,
        )

        results: list[SearchResult] = []

        for artifact_id, score in ranked_artifacts:
            embedding_artifact = artifacts_by_id[artifact_id]

            document = self._document_repository.get_document(
                embedding_artifact.document_id
            )

            if document is None:
                continue

            document_artifacts = (
                self._document_repository.get_artifacts(document.id)
            )

            results.append(
                SearchResult(
                    document_id=document.id,
                    score=score,
                    rank=len(results) + 1,
                    caption=find_caption(document_artifacts),
                    stored_path=document.stored_path,
                    document_type=document.document_type,
                )
            )

        return results