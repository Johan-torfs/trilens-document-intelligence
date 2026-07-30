from pathlib import Path

import pytest
from PIL import Image

from app.domain.checksum import calculate_checksum
from app.domain.document import DocumentMetadata, DocumentRecord
from app.repositories.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from app.repositories.vector_repository import VectorRepository


@pytest.fixture
def document_repository(
    tmp_path: Path,
) -> SQLiteDocumentRepository:
    repository = SQLiteDocumentRepository(
        tmp_path / "trilens-test.db"
    )
    repository.initialize()
    return repository


@pytest.fixture
def vector_repository(
    tmp_path: Path,
) -> VectorRepository:
    return VectorRepository(tmp_path / "vectors")


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
        mime_type="image/png",
        document_type="invoice",
        metadata=DocumentMetadata(
            document_type="invoice",
            contains_table=True,
        ),
    )

@pytest.fixture
def sample_receipt_document(
    tmp_path: Path,
) -> DocumentRecord:
    image_path = tmp_path / "receipt_001.png"

    Image.new(
        mode="RGB",
        size=(200, 300),
        color="lightgray",
    ).save(image_path)

    return DocumentRecord(
        id="receipt_001",
        original_filename=image_path.name,
        stored_path=image_path.as_posix(),
        checksum=calculate_checksum(image_path),
        width=200,
        height=300,
        mime_type="image/png",
        document_type="receipt",
        metadata=DocumentMetadata(
            document_type="receipt",
            contains_table=True,
        ),
    )