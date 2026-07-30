from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.analysis_service import AnalysisService
from app.services.caption_service import CaptionService
from app.services.document_caption_service import (
    DocumentCaptionService,
)
from app.services.document_intelligence_pipeline import (
    DocumentIntelligencePipeline,
)
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
from app.strategies.blip_caption import BlipCaptionStrategy
from app.strategies.clip_retrieval import ClipRetrievalStrategy
from app.strategies.open_flamingo_analysis import (
    OpenFlamingoAnalysisStrategy,
)


def create_document_intelligence_pipeline(
    document_repository: DocumentRepository,
    vector_repository: VectorRepository,
    open_flamingo_enabled: bool = False,
    open_flamingo_device: str | None = "cpu",
) -> DocumentIntelligencePipeline:
    retrieval_strategy = ClipRetrievalStrategy()

    indexing_service = IndexingService(
        strategy=retrieval_strategy,
        vector_repository=vector_repository,
        document_repository=document_repository,
    )

    retrieval_service = RetrievalService(
        strategy=retrieval_strategy,
        vector_repository=vector_repository,
    )

    caption_strategy = BlipCaptionStrategy()

    caption_service = CaptionService(
        strategy=caption_strategy,
    )

    document_caption_service = DocumentCaptionService(
        caption_service=caption_service,
        document_repository=document_repository,
    )

    document_search_service = DocumentSearchService(
        retrieval_service=retrieval_service,
        document_repository=document_repository,
    )

    analysis_strategy = OpenFlamingoAnalysisStrategy(
        device=open_flamingo_device,
    )

    analysis_service = AnalysisService(
        strategy=analysis_strategy,
        enabled=open_flamingo_enabled,
        fallback_caption_service=caption_service,
    )

    return DocumentIntelligencePipeline(
        document_repository=document_repository,
        indexing_service=indexing_service,
        caption_service=document_caption_service,
        document_search_service=document_search_service,
        retrieval_service=retrieval_service,
        analysis_service=analysis_service,
    )