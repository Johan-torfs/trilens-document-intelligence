from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from app.domain.document import (
    ArtifactType,
    DocumentRecord,
    ModelArtifact,
    ProcessingStatus,
)
from app.preprocessing.pipeline import preprocess_image
from app.repositories.document_repository import DocumentRepository
from app.services.document_caption_service import (
    DocumentCaptionService,
)
from app.services.indexing_service import IndexingService
from app.domain.search import SearchQuery, SearchResult
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.retrieval_service import RetrievalService
from app.services.analysis_service import (
    AnalysisResult,
    AnalysisService,
)


@dataclass(frozen=True)
class IndexDocumentOutcome:
    document: DocumentRecord
    embedding_artifact: ModelArtifact | None
    caption_artifact: ModelArtifact | None
    embedding_error: str | None
    caption_error: str | None
    reused_document: bool
    duration_ms: float

    @property
    def is_searchable(self) -> bool:
        return self.embedding_artifact is not None

    @property
    def has_caption(self) -> bool:
        return self.caption_artifact is not None

    @property
    def fully_succeeded(self) -> bool:
        return self.is_searchable and self.has_caption


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    rank: int

    final_score: float
    clip_score: float
    caption_score: float
    metadata_score: float

    caption: str | None
    stored_path: str
    document_type: str


@dataclass(frozen=True)
class SearchOutcome:
    query: SearchQuery
    ranking_mode: str
    results: list[RankedDocument]
    duration_ms: float


@dataclass(frozen=True)
class AnalyzeDocumentOutcome:
    document: DocumentRecord
    question: str
    analysis: AnalysisResult
    duration_ms: float

    @property
    def used_fallback(self) -> bool:
        return self.analysis.source == "caption_fallback"


class DocumentIntelligencePipeline:
    def __init__(
        self,
        document_repository: DocumentRepository,
        indexing_service: IndexingService,
        caption_service: DocumentCaptionService,
        document_search_service: DocumentSearchService,
        retrieval_service: RetrievalService,
        analysis_service: AnalysisService,
    ) -> None:
        self._document_repository = document_repository
        self._indexing_service = indexing_service
        self._caption_service = caption_service
        self._document_search_service = document_search_service
        self._retrieval_service = retrieval_service
        self._analysis_service = analysis_service

    def index_document(
        self,
        document: DocumentRecord,
        image_path: Path,
    ) -> IndexDocumentOutcome:
        started_at = perf_counter()

        existing_document = (
            self._document_repository.get_document_by_checksum(
                document.checksum
            )
        )

        reused_document = existing_document is not None
        stored_document = existing_document or document

        existing_artifacts = (
            self._document_repository.get_artifacts(
                stored_document.id
            )
            if existing_document is not None
            else []
        )

        embedding_artifact = self._find_embedding(
            existing_artifacts
        )
        caption_artifact = self._find_caption(
            existing_artifacts
        )

        # Volledig en met de actieve modellen verwerkt:
        # niets opnieuw uitvoeren.
        if (
            embedding_artifact is not None
            and caption_artifact is not None
        ):
            return IndexDocumentOutcome(
                document=stored_document,
                embedding_artifact=embedding_artifact,
                caption_artifact=caption_artifact,
                embedding_error=None,
                caption_error=None,
                reused_document=True,
                duration_ms=self._elapsed_ms(started_at),
            )

        # Alleen nodig wanneer minstens één modelartifact ontbreekt.
        preprocessing_result = preprocess_image(image_path)

        if existing_document is None:
            self._document_repository.save_document(
                stored_document
            )

        self._document_repository.update_processing_status(
            stored_document.id,
            ProcessingStatus.PROCESSING,
        )

        embedding_error: str | None = None
        caption_error: str | None = None

        if embedding_artifact is None:
            try:
                embedding_artifact = (
                    self._indexing_service.index_image(
                        document_id=stored_document.id,
                        image=preprocessing_result.image,
                    )
                )
            except RuntimeError as error:
                embedding_error = str(error)

        if caption_artifact is None:
            try:
                caption_artifact = (
                    self._caption_service.caption_document(
                        document_id=stored_document.id,
                        image=preprocessing_result.image,
                    )
                )
            except RuntimeError as error:
                caption_error = str(error)

        if embedding_artifact is None:
            final_status = ProcessingStatus.FAILED
            processing_error = (
                embedding_error
                or "Er kon geen CLIP-embedding worden gemaakt."
            )
        else:
            final_status = ProcessingStatus.COMPLETED
            processing_error = caption_error

        self._document_repository.update_processing_status(
            stored_document.id,
            final_status,
            processing_error,
        )

        return IndexDocumentOutcome(
            document=stored_document,
            embedding_artifact=embedding_artifact,
            caption_artifact=caption_artifact,
            embedding_error=embedding_error,
            caption_error=caption_error,
            reused_document=reused_document,
            duration_ms=self._elapsed_ms(started_at),
        )

    def search(
        self,
        query: SearchQuery,
        use_hybrid_ranking: bool = False,
    ) -> SearchOutcome:
        started_at = perf_counter()

        if not use_hybrid_ranking:
            baseline_results = (
                self._document_search_service.search(query)
            )

            results = [
                self._to_clip_ranked_document(result)
                for result in baseline_results
            ]

            return SearchOutcome(
                query=query,
                ranking_mode="clip",
                results=results,
                duration_ms=self._elapsed_ms(started_at),
            )

        candidate_top_k = min(query.top_k * 3, 100)

        candidate_query = query.model_copy(
            update={"top_k": candidate_top_k}
        )

        baseline_candidates = (
            self._document_search_service.search(
                candidate_query
            )
        )

        hybrid_candidates = [
            self._to_hybrid_ranked_document(
                query=query,
                result=result,
            )
            for result in baseline_candidates
        ]

        hybrid_candidates.sort(
            key=lambda result: result.final_score,
            reverse=True,
        )

        final_results = [
            RankedDocument(
                document_id=result.document_id,
                rank=rank,
                final_score=result.final_score,
                clip_score=result.clip_score,
                caption_score=result.caption_score,
                metadata_score=result.metadata_score,
                caption=result.caption,
                stored_path=result.stored_path,
                document_type=result.document_type,
            )
            for rank, result in enumerate(
                hybrid_candidates[: query.top_k],
                start=1,
            )
        ]

        return SearchOutcome(
            query=query,
            ranking_mode="hybrid",
            results=final_results,
            duration_ms=self._elapsed_ms(started_at),
        )


    def analyze_document(
        self,
        document_id: str,
        question: str,
    ) -> AnalyzeDocumentOutcome:
        started_at = perf_counter()

        cleaned_document_id = document_id.strip()
        cleaned_question = question.strip()

        if not cleaned_document_id:
            raise ValueError(
                "Het document-ID mag niet leeg zijn."
            )

        if not cleaned_question:
            raise ValueError(
                "De analysevraag mag niet leeg zijn."
            )

        document = self._document_repository.get_document(
            cleaned_document_id
        )

        if document is None:
            raise ValueError(
                f"Document '{cleaned_document_id}' werd niet gevonden."
            )

        preprocessing_result = preprocess_image(
            Path(document.stored_path)
        )

        analysis = self._analysis_service.analyze(
            image=preprocessing_result.image,
            question=cleaned_question,
        )

        return AnalyzeDocumentOutcome(
            document=document,
            question=cleaned_question,
            analysis=analysis,
            duration_ms=self._elapsed_ms(started_at),
        )


    def get_document_file(
        self,
        document_id: str,
    ) -> tuple[DocumentRecord, Path]:
        cleaned_document_id = document_id.strip()

        if not cleaned_document_id:
            raise ValueError(
                "Het document-ID mag niet leeg zijn."
            )

        document = self._document_repository.get_document(
            cleaned_document_id
        )

        if document is None:
            raise ValueError(
                f"Document '{cleaned_document_id}' werd niet gevonden."
            )

        image_path = Path(document.stored_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Het afbeeldingsbestand van document "
                f"'{cleaned_document_id}' werd niet gevonden."
            )

        return document, image_path
    

    @staticmethod
    def _to_clip_ranked_document(
        result: SearchResult,
    ) -> RankedDocument:
        return RankedDocument(
            document_id=result.document_id,
            rank=result.rank,
            final_score=result.score,
            clip_score=result.score,
            caption_score=0.0,
            metadata_score=0.0,
            caption=result.caption,
            stored_path=result.stored_path,
            document_type=result.document_type,
        )


    def _to_hybrid_ranked_document(
        self,
        query: SearchQuery,
        result: SearchResult,
    ) -> RankedDocument:
        caption_score = 0.0

        if result.caption:
            caption_score = (
                self._retrieval_service.text_similarity(
                    query.text,
                    result.caption,
                )
            )

        metadata_score = self._metadata_similarity(
            query=query,
            result=result,
        )

        final_score = (
            0.70 * result.score
            + 0.20 * caption_score
            + 0.10 * metadata_score
        )

        return RankedDocument(
            document_id=result.document_id,
            rank=result.rank,
            final_score=final_score,
            clip_score=result.score,
            caption_score=caption_score,
            metadata_score=metadata_score,
            caption=result.caption,
            stored_path=result.stored_path,
            document_type=result.document_type,
        )


    def _metadata_similarity(
        self,
        query: SearchQuery,
        result: SearchResult,
    ) -> float:
        if query.document_type is not None:
            return (
                1.0
                if result.document_type == query.document_type
                else 0.0
            )

        readable_document_type = (
            result.document_type.replace("_", " ")
        )

        return max(
            0.0,
            self._retrieval_service.text_similarity(
                query.text,
                readable_document_type,
            ),
        )

    def _find_embedding(
        self,
        artifacts: list[ModelArtifact],
    ) -> ModelArtifact | None:
        for artifact in reversed(artifacts):
            if (
                artifact.artifact_type
                == ArtifactType.IMAGE_EMBEDDING
                and artifact.model_name
                == self._indexing_service.model_name
            ):
                return artifact

        return None

    def _find_caption(
        self,
        artifacts: list[ModelArtifact],
    ) -> ModelArtifact | None:
        for artifact in reversed(artifacts):
            if (
                artifact.artifact_type
                == ArtifactType.CAPTION
                and artifact.model_name
                == self._caption_service.model_name
                and artifact.model_version
                == self._caption_service.model_version
                and artifact.content
            ):
                return artifact

        return None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (perf_counter() - started_at) * 1000