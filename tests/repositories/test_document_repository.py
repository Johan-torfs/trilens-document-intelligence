import sqlite3

import pytest

from app.domain.document import (
    DocumentRecord,
    ProcessingStatus,
)
from app.domain.ocr import OCRResult
from app.repositories.sqlite_document_repository import (
    DuplicateDocumentError,
    SQLiteDocumentRepository,
)


def test_saves_and_retrieves_document(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    stored_document = document_repository.get_document(
        sample_document.id
    )

    assert stored_document is not None
    assert stored_document.id == sample_document.id
    assert stored_document.checksum == sample_document.checksum
    assert stored_document.document_type == "invoice"
    assert stored_document.language == "en"
    assert stored_document.page_count == 1
    assert stored_document.ocr is None


def test_retrieves_document_by_checksum(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    stored_document = (
        document_repository.get_document_by_checksum(
            sample_document.checksum
        )
    )

    assert stored_document is not None
    assert stored_document.id == sample_document.id


def test_returns_none_for_unknown_document(
    document_repository: SQLiteDocumentRepository,
) -> None:
    assert (
        document_repository.get_document("missing-document")
        is None
    )


def test_blocks_duplicate_checksum(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    duplicate = sample_document.model_copy(
        update={"id": "invoice_001_copy"}
    )

    with pytest.raises(
        DuplicateDocumentError,
        match="invoice_001",
    ):
        document_repository.save_document(duplicate)


def test_updates_processing_status_and_error(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
) -> None:
    document_repository.save_document(sample_document)

    document_repository.update_processing_status(
        document_id=sample_document.id,
        status=ProcessingStatus.FAILED,
        error="OCR processing failed",
    )

    stored_document = document_repository.get_document(
        sample_document.id
    )

    assert stored_document is not None
    assert (
        stored_document.processing_status
        is ProcessingStatus.FAILED
    )
    assert (
        stored_document.processing_error
        == "OCR processing failed"
    )


def test_updates_and_retrieves_ocr(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
    sample_ocr_result: OCRResult,
) -> None:
    document_repository.save_document(sample_document)

    document_repository.update_ocr(
        sample_document.id,
        sample_ocr_result,
    )

    stored_document = document_repository.get_document(
        sample_document.id
    )

    assert stored_document is not None
    assert stored_document.ocr == sample_ocr_result

    with sqlite3.connect(
        document_repository.database_path
    ) as connection:
        row = connection.execute(
            """
            SELECT ocr_text, ocr_json
            FROM documents
            WHERE id = ?
            """,
            (sample_document.id,),
        ).fetchone()

    assert row is not None
    assert row[0] == sample_ocr_result.text
    assert row[1] == sample_ocr_result.model_dump_json()


def test_clears_ocr(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
    sample_ocr_result: OCRResult,
) -> None:
    document_repository.save_document(sample_document)
    document_repository.update_ocr(
        sample_document.id,
        sample_ocr_result,
    )

    document_repository.update_ocr(
        sample_document.id,
        None,
    )

    stored_document = document_repository.get_document(
        sample_document.id
    )

    assert stored_document is not None
    assert stored_document.ocr is None


def test_update_status_rejects_unknown_document(
    document_repository: SQLiteDocumentRepository,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing-document",
    ):
        document_repository.update_processing_status(
            document_id="missing-document",
            status=ProcessingStatus.FAILED,
        )


def test_update_ocr_rejects_unknown_document(
    document_repository: SQLiteDocumentRepository,
    sample_ocr_result: OCRResult,
) -> None:
    with pytest.raises(
        ValueError,
        match="missing-document",
    ):
        document_repository.update_ocr(
            "missing-document",
            sample_ocr_result,
        )


def test_schema_does_not_create_artifact_table(
    document_repository: SQLiteDocumentRepository,
) -> None:
    with sqlite3.connect(
        document_repository.database_path
    ) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    assert "documents" in tables
    assert "model_artifacts" not in tables
    assert "schema_version" not in tables


def test_lexical_search_finds_indexed_document(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
    sample_ocr_result: OCRResult,
) -> None:
    document_repository.save_document(sample_document)
    document_repository.update_ocr(
        sample_document.id,
        sample_ocr_result,
    )

    results = document_repository.lexical_search("Invoice")

    assert len(results) == 1
    assert results[0][0] == sample_document.id
    assert results[0][1] > 0.0


def test_lexical_search_returns_empty_for_no_match(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
    sample_ocr_result: OCRResult,
) -> None:
    document_repository.save_document(sample_document)
    document_repository.update_ocr(
        sample_document.id,
        sample_ocr_result,
    )

    results = document_repository.lexical_search(
        "xyzzynotarealword"
    )

    assert results == []


def test_lexical_search_returns_empty_for_blank_query(
    document_repository: SQLiteDocumentRepository,
) -> None:
    assert document_repository.lexical_search("   ") == []


def test_lexical_search_reflects_updated_ocr(
    document_repository: SQLiteDocumentRepository,
    sample_document: DocumentRecord,
    sample_ocr_result: OCRResult,
) -> None:
    document_repository.save_document(sample_document)
    document_repository.update_ocr(
        sample_document.id,
        sample_ocr_result,
    )

    # Clear OCR — document should no longer be findable
    document_repository.update_ocr(sample_document.id, None)

    results = document_repository.lexical_search("Invoice")
    assert results == []