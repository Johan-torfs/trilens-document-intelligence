from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from app.domain.document import DocumentRecord
from app.domain.ocr import OCRResult
from app.domain.text_chunk import TextChunk
from app.domain.vector import VectorPoint
from app.repositories.vector_repository import VectorRepository
from app.services.text_chunker import TextChunker
from app.strategies.embedding import EmbeddingStrategy


TEXT_VECTOR_NAME = "text"
TEXT_UNIT_TYPE = "chunk"


class TextIndexingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TextIndexingResult:
    document_id: str
    chunk_count: int
    dimensions: int
    model_name: str
    model_version: str | None
    reused_existing: bool


class TextIndexingService:
    def __init__(
        self,
        strategy: EmbeddingStrategy,
        vector_repository: VectorRepository,
        chunker: TextChunker | None = None,
    ) -> None:
        self._strategy = strategy
        self._vector_repository = vector_repository
        self._chunker = chunker or TextChunker()

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    @property
    def model_version(self) -> str | None:
        return self._strategy.model_version

    def current_result(
        self,
        document: DocumentRecord,
    ) -> TextIndexingResult | None:
        try:
            stored_count = self._vector_repository.count(
                filters=self._base_filters(document)
            )
        except Exception as error:
            raise TextIndexingError(
                f"Could not inspect text vectors for document "
                f"'{document.id}': {error}"
            ) from error

        if stored_count == 0:
            return None

        return TextIndexingResult(
            document_id=document.id,
            chunk_count=stored_count,
            dimensions=0,
            model_name=self.model_name,
            model_version=self.model_version,
            reused_existing=True,
        )

    def index_document_text(
        self,
        document: DocumentRecord,
        ocr_result: OCRResult,
    ) -> TextIndexingResult:
        cached = self.current_result(document)

        if cached is not None:
            return cached

        chunks = self._chunker.chunk(ocr_result)

        if not chunks:
            raise TextIndexingError(
                f"No text chunks produced for document '{document.id}'."
            )

        try:
            embeddings = [
                self._strategy.embed_text(chunk.text)
                for chunk in chunks
            ]

            points = [
                VectorPoint(
                    id=self._point_id(
                        document_id=document.id,
                        page_number=chunk.page_number,
                        chunk_number=chunk.chunk_number,
                    ),
                    vector_name=TEXT_VECTOR_NAME,
                    values=embeddings[index],
                    payload=self._payload(
                        document=document,
                        chunk=chunk,
                    ),
                )
                for index, chunk in enumerate(chunks)
            ]

            self._vector_repository.save_batch(points)

        except TextIndexingError:
            raise

        except Exception as error:
            raise TextIndexingError(
                f"Text indexing failed for document "
                f"'{document.id}': {error}"
            ) from error

        dimensions = len(embeddings[0]) if embeddings else 0

        return TextIndexingResult(
            document_id=document.id,
            chunk_count=len(chunks),
            dimensions=dimensions,
            model_name=self.model_name,
            model_version=self.model_version,
            reused_existing=False,
        )

    def _base_filters(
        self,
        document: DocumentRecord,
    ) -> dict[str, str]:
        return {
            "document_id": document.id,
            "checksum": document.checksum,
            "unit_type": TEXT_UNIT_TYPE,
            "vector_type": TEXT_VECTOR_NAME,
            "model_name": self.model_name,
            "model_version": self.model_version or "",
        }

    def _payload(
        self,
        document: DocumentRecord,
        chunk: TextChunk,
    ) -> dict[str, str | int]:
        return {
            **self._base_filters(document),
            "document_type": document.document_type,
            "page_number": chunk.page_number,
            "chunk_number": chunk.chunk_number,
            "chunk_text": chunk.text,
        }

    @staticmethod
    def _point_id(
        document_id: str,
        page_number: int,
        chunk_number: int,
    ) -> str:
        identity = (
            f"{TEXT_VECTOR_NAME}:"
            f"{document_id}:"
            f"{page_number}:"
            f"{chunk_number}"
        )

        return str(uuid5(NAMESPACE_URL, identity))
