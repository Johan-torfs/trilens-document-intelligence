from pathlib import Path

from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.analysis_service import AnalysisService
from app.services.document_intelligence_pipeline import (
    DocumentIntelligencePipeline,
)
from app.services.document_preparation_service import (
    DocumentPreparationService,
)
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
from app.strategies.image_document import ImageDocumentStrategy
from app.strategies.pdf_document import PDFDocumentStrategy
from app.strategies.siglip_retrieval import SiglipRetrievalStrategy
from app.strategies.open_flamingo_analysis import (
    OpenFlamingoAnalysisStrategy,
)
from app.services.document_ocr_service import DocumentOCRService
from app.strategies.doctr_ocr import DocTROCRStrategy


def create_document_intelligence_pipeline(
    document_repository: DocumentRepository,
    vector_repository: VectorRepository,
    upload_dir: Path,
    open_flamingo_enabled: bool = False,
    open_flamingo_device: str | None = "cpu",
) -> DocumentIntelligencePipeline:
    preparation_service = DocumentPreparationService(
        strategies=[
            ImageDocumentStrategy(),
            PDFDocumentStrategy(),
        ]
    )

    retrieval_strategy = SiglipRetrievalStrategy()
    ocr_strategy = DocTROCRStrategy()

    indexing_service = IndexingService(
        strategy=retrieval_strategy,
        vector_repository=vector_repository,
        document_repository=document_repository,
    )

    retrieval_service = RetrievalService(
        strategy=retrieval_strategy,
        vector_repository=vector_repository,
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
        fallback_caption_service=None,
    )

    ocr_service = DocumentOCRService(
        strategy=ocr_strategy,
        document_repository=document_repository,
    )

    return DocumentIntelligencePipeline(
        document_repository=document_repository,
        preparation_service=preparation_service,
        upload_dir=upload_dir,
        indexing_service=indexing_service,
        caption_service=None,
        document_search_service=document_search_service,
        retrieval_service=retrieval_service,
        analysis_service=analysis_service,
        ocr_service=ocr_service,
    )