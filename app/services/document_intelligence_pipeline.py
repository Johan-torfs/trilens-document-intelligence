from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.domain.checksum import calculate_checksum_bytes
from app.domain.document import (
    DocumentRecord,
    ProcessingStatus,
)
from app.domain.ocr import OCRResult
from app.domain.prepared_document import (
    DocumentPage,
    DocumentSource,
)
from app.repositories.document_repository import DocumentRepository
from app.services.document_preparation_service import (
    DocumentPreparationService,
)
from app.services.indexing_service import DocumentIndexingError, IndexingResult, IndexingService
from app.services.text_indexing_service import (
    TextIndexingError,
    TextIndexingResult,
    TextIndexingService,
)
from app.domain.search import SearchQuery, SearchResult
from app.services.document_search_service import (
    DocumentSearchService,
)
from app.services.retrieval_service import RetrievalService
from app.services.analysis_service import (
    AnalysisResult,
    AnalysisService,
)
from app.services.document_classification_service import (
    DocumentClassificationService,
)
from app.services.document_ocr_service import DocumentOCRService, OCRProcessingError
from app.services.score_calibration import (
    LinearScoreCalibrator,
)


_MIME_TO_EXTENSION: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "application/pdf": ".pdf",
}


@dataclass(frozen=True)
class IndexDocumentOutcome:
    document: DocumentRecord
    indexing_result: IndexingResult | None
    ocr_result: OCRResult | None
    text_indexing_result: TextIndexingResult | None
    indexing_error: str | None
    ocr_error: str | None
    text_indexing_error: str | None
    reused_document: bool
    duration_ms: float
    classification_confidence: float | None = None

    @property
    def is_searchable(self) -> bool:
        return self.indexing_result is not None

    @property
    def has_ocr(self) -> bool:
        return self.ocr_result is not None

    @property
    def fully_succeeded(self) -> bool:
        return self.is_searchable and self.has_ocr


@dataclass(frozen=True)
class RankedDocument:
    document_id: str
    rank: int

    final_score: float
    visual_score: float
    text_score: float
    fts_score: float
    calibrated_score: float

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


class DocumentIntelligencePipeline:
    def __init__(
        self,
        document_repository: DocumentRepository,
        preparation_service: DocumentPreparationService,
        upload_dir: Path,
        indexing_service: IndexingService,
        document_search_service: DocumentSearchService,
        retrieval_service: RetrievalService,
        analysis_service: AnalysisService,
        ocr_service: DocumentOCRService,
        score_calibrator: LinearScoreCalibrator,
        text_indexing_service: TextIndexingService | None = None,
        classification_service: DocumentClassificationService | None = None,
    ) -> None:
        self._document_repository = document_repository
        self._preparation_service = preparation_service
        self._upload_dir = upload_dir
        self._indexing_service = indexing_service
        self._document_search_service = document_search_service
        self._retrieval_service = retrieval_service
        self._analysis_service = analysis_service
        self._ocr_service = ocr_service
        self._score_calibrator = score_calibrator
        self._text_indexing_service = text_indexing_service
        self._classification_service = classification_service

    def index_document(
        self,
        source: DocumentSource,
        document_type: str | None = None,
    ) -> IndexDocumentOutcome:
        started_at = perf_counter()

        auto_classify = document_type is None

        if not auto_classify:
            cleaned_document_type: str = document_type.strip()  # type: ignore[union-attr]
            if not cleaned_document_type:
                raise ValueError(
                    "Het documenttype mag niet leeg zijn."
                )
        else:
            if self._classification_service is None:
                raise ValueError(
                    "Geen classificationservice geconfigureerd voor automatische detectie."
                )
            cleaned_document_type = "unknown"

        checksum = calculate_checksum_bytes(source.content)

        existing_document = (
            self._document_repository.get_document_by_checksum(
                checksum
            )
        )

        reused_document = existing_document is not None

        if existing_document is not None:
            try:
                existing_indexing = (
                    self._indexing_service.current_result(
                        existing_document
                    )
                )
            except DocumentIndexingError:
                existing_indexing = None    

            existing_ocr = self._ocr_service.current_result(
                existing_document
            )

            if (
                existing_indexing is not None
                and existing_ocr is not None
            ):
                existing_text = None

                if self._text_indexing_service is not None:
                    try:
                        existing_text = (
                            self._text_indexing_service.current_result(
                                existing_document
                            )
                        )
                    except TextIndexingError:
                        pass

                return IndexDocumentOutcome(
                    document=existing_document,
                    indexing_result=existing_indexing,
                    ocr_result=existing_ocr,
                    text_indexing_result=existing_text,
                    indexing_error=None,
                    ocr_error=None,
                    text_indexing_error=None,
                    reused_document=True,
                    duration_ms=self._elapsed_ms(started_at),
                )

        prepared = self._preparation_service.prepare(source)
        pages = prepared.pages

        if existing_document is None:
            stored_path = self._store_file(
                source=source,
                checksum=checksum,
            )

            stored_document = DocumentRecord(
                id=str(uuid4()),
                original_filename=source.filename,
                stored_path=str(stored_path),
                checksum=checksum,
                width=pages[0].width,
                height=pages[0].height,
                mime_type=source.mime_type,
                page_count=len(pages),
                document_type=cleaned_document_type,
            )

            self._document_repository.save_document(
                stored_document
            )

        else:
            stored_document = existing_document

            if stored_document.page_count != len(pages):
                raise ValueError(
                    f"Stored document '{stored_document.id}' "
                    f"contains {stored_document.page_count} pages, "
                    f"but preparation returned {len(pages)}."
                )

        preprocessing_results = (
            self._preparation_service.preprocess_pages(prepared)
        )

        preprocessed_pages = [
            DocumentPage(
                page_number=page.page_number,
                image=result.image,
            )
            for page, result in zip(
                pages,
                preprocessing_results,
                strict=True,
            )
        ]

        self._document_repository.update_processing_status(
            document_id=stored_document.id,
            status=ProcessingStatus.PROCESSING,
        )

        indexing_result: IndexingResult | None = None
        ocr_result: OCRResult | None = None
        text_indexing_result: TextIndexingResult | None = None

        indexing_error: str | None = None
        ocr_error: str | None = None
        text_indexing_error: str | None = None

        try:
            indexing_result = self._indexing_service.index_pages(
                document=stored_document,
                pages=preprocessed_pages,
            )
        except DocumentIndexingError as error:
            indexing_error = str(error)

        try:
            ocr_result = self._ocr_service.process_document(
                document=stored_document,
                pages=preprocessed_pages,
            )
        except OCRProcessingError as error:
            ocr_error = str(error)

        if (
            self._text_indexing_service is not None
            and ocr_result is not None
        ):
            try:
                text_indexing_result = (
                    self._text_indexing_service.index_document_text(
                        document=stored_document,
                        ocr_result=ocr_result,
                    )
                )
            except TextIndexingError as error:
                text_indexing_error = str(error)

        classification_confidence: float | None = None

        if auto_classify and self._classification_service is not None:
            ocr_text = ocr_result.text if ocr_result is not None else ""
            classification_result = self._classification_service.classify(
                image=preprocessed_pages[0].image,
                ocr_text=ocr_text,
            )
            classification_confidence = classification_result.confidence
            self._document_repository.update_document_type(
                document_id=stored_document.id,
                document_type=classification_result.document_type,
            )

        fully_succeeded = (
            indexing_result is not None
            and ocr_result is not None
        )

        final_status = (
            ProcessingStatus.COMPLETED
            if fully_succeeded
            else ProcessingStatus.FAILED
        )

        errors = [
            error
            for error in (
                indexing_error,
                ocr_error,
                text_indexing_error,
            )
            if error
        ]

        processing_error = (
            " | ".join(errors)
            if errors
            else None
        )

        self._document_repository.update_processing_status(
            document_id=stored_document.id,
            status=final_status,
            error=processing_error,
        )

        updated_document = (
            self._document_repository.get_document(
                stored_document.id
            )
            or stored_document
        )

        return IndexDocumentOutcome(
            document=updated_document,
            indexing_result=indexing_result,
            ocr_result=ocr_result,
            text_indexing_result=text_indexing_result,
            indexing_error=indexing_error,
            ocr_error=ocr_error,
            text_indexing_error=text_indexing_error,
            reused_document=reused_document,
            duration_ms=self._elapsed_ms(started_at),
            classification_confidence=classification_confidence,
        )

    def search(
        self,
        query: SearchQuery,
    ) -> SearchOutcome:
        started_at = perf_counter()

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
                visual_score=result.visual_score,
                text_score=result.text_score,
                fts_score=result.fts_score,
                calibrated_score=result.calibrated_score,
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

        stored_path = Path(document.stored_path)
        source = DocumentSource(
            filename=stored_path.name,
            mime_type=document.mime_type,
            content=stored_path.read_bytes(),
        )

        prepared = self._preparation_service.prepare(source)
        preprocessing_results = (
            self._preparation_service.preprocess_pages(prepared)
        )
        first_page_image = preprocessing_results[0].image

        analysis = self._analysis_service.analyze(
            image=first_page_image,
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
    

    def _store_file(
        self,
        source: DocumentSource,
        checksum: str,
    ) -> Path:
        extension = Path(source.filename).suffix.lower()

        if not extension:
            extension = _MIME_TO_EXTENSION.get(
                source.mime_type.lower(), ""
            )

        self._upload_dir.mkdir(parents=True, exist_ok=True)

        stored_path = self._upload_dir / f"{checksum}{extension}"

        if not stored_path.exists():
            stored_path.write_bytes(source.content)

        return stored_path

    def _to_hybrid_ranked_document(
        self,
        query: SearchQuery,
        result: SearchResult,
    ) -> RankedDocument:
        if result.text_score > 0.0 or result.fts_score > 0.0:
            final_score = (
                0.60 * result.score
                + 0.30 * result.text_score
                + 0.10 * result.fts_score
            )
        else:
            final_score = result.score

        return RankedDocument(
            document_id=result.document_id,
            rank=result.rank,
            final_score=final_score,
            visual_score=result.score,
            text_score=result.text_score,
            fts_score=result.fts_score,
            calibrated_score=self._score_calibrator.calibrate(final_score),
            stored_path=result.stored_path,
            document_type=result.document_type,
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (perf_counter() - started_at) * 1000