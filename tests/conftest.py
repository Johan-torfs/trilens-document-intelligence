from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest
from PIL import Image
from qdrant_client import QdrantClient

from app.domain.checksum import calculate_checksum
from app.domain.document import DocumentRecord
from app.domain.ocr import OCRPageResult, OCRResult
from app.repositories.document_repository import DocumentRepository
from app.repositories.qdrant_vector_repository import (
    QdrantVectorRepository,
)
from app.repositories.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from app.repositories.vector_repository import VectorRepository
import app.repositories.qdrant_vector_repository as qdrant_repository_module


@pytest.fixture
def document_repository(
    tmp_path: Path,
) -> SQLiteDocumentRepository:
    return SQLiteDocumentRepository(
        tmp_path / "trilens-test.db"
    )


@pytest.fixture
def mock_document_repository() -> MagicMock:
    return create_autospec(
        DocumentRepository,
        instance=True,
    )


@pytest.fixture
def qdrant_client_mock() -> MagicMock:
    return create_autospec(
        QdrantClient,
        instance=True,
    )


@pytest.fixture
def vector_repository(
    monkeypatch: pytest.MonkeyPatch,
    qdrant_client_mock: MagicMock,
) -> QdrantVectorRepository:
    monkeypatch.setattr(
        qdrant_repository_module,
        "QdrantClient",
        MagicMock(return_value=qdrant_client_mock),
    )

    return QdrantVectorRepository(
        url="http://localhost:6333",
        collection_name="trilens_test_vectors",
        timeout_seconds=1,
        exact_search=True,
    )


@pytest.fixture
def mock_vector_repository() -> MagicMock:
    return create_autospec(
        VectorRepository,
        instance=True,
    )


@pytest.fixture
def sample_document(
    tmp_path: Path,
) -> DocumentRecord:
    image_path = tmp_path / "invoice_001.png"

    Image.new(
        mode="RGB",
        size=(200, 300),
        color="white",
    ).save(image_path)

    return DocumentRecord(
        id="invoice_001",
        original_filename=image_path.name,
        stored_path=image_path.as_posix(),
        checksum=calculate_checksum(image_path),
        width=200,
        height=300,
        page_count=1,
        mime_type="image/png",
        document_type="invoice",
        language="en",
    )


@pytest.fixture
def sample_ocr_result() -> OCRResult:
    return OCRResult(
        text="Invoice 123 Total 49.95",
        pages=[
            OCRPageResult(
                page_number=1,
                text="Invoice 123 Total 49.95",
                words=[],
                mean_confidence=0.96,
            )
        ],
        mean_confidence=0.96,
        model_name="doctr",
        model_version="1",
    )