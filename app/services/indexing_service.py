from collections.abc import Sequence
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from app.domain.document import DocumentRecord
from app.domain.prepared_document import DocumentPage
from app.domain.vector import VectorPoint
from app.repositories.vector_repository import VectorRepository
from app.strategies.embedding import EmbeddingStrategy


VISUAL_VECTOR_NAME = "visual"
VISUAL_UNIT_TYPE = "page"


class DocumentIndexingError(RuntimeError):
    pass


@dataclass(frozen=True)
class IndexingResult:
    document_id: str
    page_count: int
    dimensions: int
    model_name: str
    model_version: str | None
    reused_existing: bool


class IndexingService:
    def __init__(
        self,
        strategy: EmbeddingStrategy,
        vector_repository: VectorRepository,
    ) -> None:
        self._strategy = strategy
        self._vector_repository = vector_repository

    @property
    def model_name(self) -> str:
        return self._strategy.model_name

    @property
    def model_version(self) -> str | None:
        return self._strategy.model_version

    def current_result(
        self,
        document: DocumentRecord,
    ) -> IndexingResult | None:
        try:
            indexed_page_count = self._vector_repository.count(
                filters=self._filters(document)
            )
        except Exception as error:
            raise DocumentIndexingError(
                f"Could not inspect vectors for document "
                f"'{document.id}': {error}"
            ) from error

        if indexed_page_count != document.page_count:
            return None

        return IndexingResult(
            document_id=document.id,
            page_count=document.page_count,
            dimensions=0,
            model_name=self.model_name,
            model_version=self.model_version,
            reused_existing=True,
        )

    def index_pages(
        self,
        document: DocumentRecord,
        pages: Sequence[DocumentPage],
    ) -> IndexingResult:
        self._validate_pages(document, pages)

        cached_result = self.current_result(document)

        if cached_result is not None:
            return cached_result

        try:
            embeddings = np.asarray(
                self._strategy.embed_images(
                    [page.image for page in pages]
                ),
                dtype=np.float32,
            )

            self._validate_embeddings(
                embeddings=embeddings,
                expected_count=len(pages),
            )

            points = [
                VectorPoint(
                    id=self._point_id(
                        document_id=document.id,
                        page_number=page.page_number,
                    ),
                    vector_name=VISUAL_VECTOR_NAME,
                    values=embeddings[index],
                    payload=self._payload(
                        document=document,
                        page_number=page.page_number,
                    ),
                )
                for index, page in enumerate(pages)
            ]

            self._vector_repository.save_batch(points)

            stored_page_count = self._vector_repository.count(
                filters=self._filters(document)
            )

        except DocumentIndexingError:
            raise

        except Exception as error:
            raise DocumentIndexingError(
                f"Visual indexing failed for document "
                f"'{document.id}': {error}"
            ) from error

        if stored_page_count != len(pages):
            raise DocumentIndexingError(
                f"Expected {len(pages)} visual vectors for document "
                f"'{document.id}', but found {stored_page_count}."
            )

        return IndexingResult(
            document_id=document.id,
            page_count=len(pages),
            dimensions=int(embeddings.shape[1]),
            model_name=self.model_name,
            model_version=self.model_version,
            reused_existing=False,
        )

    def _filters(
        self,
        document: DocumentRecord,
    ) -> dict[str, str]:
        return {
            "document_id": document.id,
            "checksum": document.checksum,
            "unit_type": VISUAL_UNIT_TYPE,
            "vector_type": VISUAL_VECTOR_NAME,
            "model_name": self.model_name,
            "model_version": self.model_version or "",
        }

    def _payload(
        self,
        document: DocumentRecord,
        page_number: int,
    ) -> dict[str, str | int]:
        return {
            **self._filters(document),
            "document_type": document.document_type,
            "page_number": page_number,
        }

    @staticmethod
    def _point_id(
        document_id: str,
        page_number: int,
    ) -> str:
        identity = (
            f"{VISUAL_VECTOR_NAME}:"
            f"{document_id}:"
            f"{page_number}"
        )

        return str(uuid5(NAMESPACE_URL, identity))

    @staticmethod
    def _validate_pages(
        document: DocumentRecord,
        pages: Sequence[DocumentPage],
    ) -> None:
        if not pages:
            raise ValueError(
                "At least one document page is required."
            )

        if len(pages) != document.page_count:
            raise ValueError(
                f"Document expects {document.page_count} pages, "
                f"but received {len(pages)}."
            )

        page_numbers = [
            page.page_number
            for page in pages
        ]

        expected = list(range(1, len(pages) + 1))

        if page_numbers != expected:
            raise ValueError(
                "Document pages must be ordered and numbered "
                "starting at one."
            )

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        expected_count: int,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                "Image embeddings must be a two-dimensional batch."
            )

        if embeddings.shape[0] != expected_count:
            raise ValueError(
                f"Expected {expected_count} image embeddings, "
                f"but received {embeddings.shape[0]}."
            )

        if embeddings.shape[1] <= 0:
            raise ValueError(
                "Image embeddings may not be empty."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Image embeddings contain non-finite values."
            )