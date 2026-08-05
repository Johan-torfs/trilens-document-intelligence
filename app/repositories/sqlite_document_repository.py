import sqlite3
from pathlib import Path

from app.domain.document import (
    DocumentRecord,
    ProcessingStatus,
)
from app.domain.ocr import OCRResult
from app.repositories.document_repository import (
    DocumentRepository,
)


class DuplicateDocumentError(ValueError):
    pass


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,

                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,

                    mime_type TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    language TEXT,

                    ocr_text TEXT,
                    ocr_json TEXT,

                    created_at TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    processing_error TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
                    USING fts5(ocr_text, tokenize='porter unicode61');

                CREATE TABLE IF NOT EXISTS documents_fts_rowids (
                    document_id TEXT PRIMARY KEY,
                    fts_rowid   INTEGER NOT NULL
                );
                """
            )

    def save_document(
        self,
        document: DocumentRecord,
    ) -> None:
        existing = self.get_document_by_checksum(
            document.checksum
        )

        if existing and existing.id != document.id:
            raise DuplicateDocumentError(
                f"Document is duplicate van '{existing.id}'"
            )

        ocr_text, ocr_json = self._serialize_ocr(
            document.ocr
        )

        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id,
                    original_filename,
                    stored_path,
                    checksum,
                    width,
                    height,
                    page_count,
                    mime_type,
                    document_type,
                    language,
                    ocr_text,
                    ocr_json,
                    created_at,
                    processing_status,
                    processing_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    original_filename = excluded.original_filename,
                    stored_path = excluded.stored_path,
                    checksum = excluded.checksum,
                    width = excluded.width,
                    height = excluded.height,
                    page_count = excluded.page_count,
                    mime_type = excluded.mime_type,
                    document_type = excluded.document_type,
                    language = excluded.language,
                    ocr_text = excluded.ocr_text,
                    ocr_json = excluded.ocr_json,
                    created_at = excluded.created_at,
                    processing_status = excluded.processing_status,
                    processing_error = excluded.processing_error
                """,
                (
                    document.id,
                    document.original_filename,
                    document.stored_path,
                    document.checksum,
                    document.width,
                    document.height,
                    document.page_count,
                    document.mime_type,
                    document.document_type,
                    document.language,
                    ocr_text,
                    ocr_json,
                    document.created_at.isoformat(),
                    document.processing_status.value,
                    document.processing_error,
                ),
            )

    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        return self._row_to_document(row) if row else None

    def get_document_by_checksum(
        self,
        checksum: str,
    ) -> DocumentRecord | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE checksum = ?
                """,
                (checksum,),
            ).fetchone()

        return self._row_to_document(row) if row else None

    def update_processing_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET processing_status = ?,
                    processing_error = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    error,
                    document_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Document '{document_id}' werd niet gevonden."
                )

    def update_ocr(
        self,
        document_id: str,
        result: OCRResult | None,
    ) -> None:
        ocr_text, ocr_json = self._serialize_ocr(result)

        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET ocr_text = ?,
                    ocr_json = ?
                WHERE id = ?
                """,
                (
                    ocr_text,
                    ocr_json,
                    document_id,
                ),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Document '{document_id}' werd niet gevonden."
                )

        self._sync_fts(
            document_id=document_id,
            ocr_text=ocr_text,
        )

    def update_document_type(
        self,
        document_id: str,
        document_type: str,
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE documents SET document_type = ? WHERE id = ?",
                (document_type, document_id),
            )

            if cursor.rowcount == 0:
                raise ValueError(
                    f"Document '{document_id}' werd niet gevonden."
                )

    def _sync_fts(
        self,
        document_id: str,
        ocr_text: str | None,
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT fts_rowid
                FROM documents_fts_rowids
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

            if row is not None:
                connection.execute(
                    "DELETE FROM documents_fts WHERE rowid = ?",
                    (row[0],),
                )
                connection.execute(
                    """
                    DELETE FROM documents_fts_rowids
                    WHERE document_id = ?
                    """,
                    (document_id,),
                )

            if ocr_text:
                cursor = connection.execute(
                    "INSERT INTO documents_fts(ocr_text) VALUES(?)",
                    (ocr_text,),
                )
                connection.execute(
                    """
                    INSERT INTO documents_fts_rowids
                        (document_id, fts_rowid)
                    VALUES (?, ?)
                    """,
                    (document_id, cursor.lastrowid),
                )

    def lexical_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        cleaned = query.strip()

        if not cleaned:
            return []

        with sqlite3.connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT r.document_id, -f.rank AS bm25_score
                FROM documents_fts f
                JOIN documents_fts_rowids r ON f.rowid = r.fts_rowid
                WHERE documents_fts MATCH ?
                ORDER BY bm25_score DESC
                LIMIT ?
                """,
                (cleaned, top_k),
            ).fetchall()

        return [(row[0], float(row[1])) for row in rows]

    @staticmethod
    def _serialize_ocr(
        result: OCRResult | None,
    ) -> tuple[str | None, str | None]:
        if result is None:
            return None, None

        return result.text, result.model_dump_json()

    @staticmethod
    def _row_to_document(
        row: sqlite3.Row,
    ) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            original_filename=row["original_filename"],
            stored_path=row["stored_path"],
            checksum=row["checksum"],
            width=row["width"],
            height=row["height"],
            page_count=row["page_count"],
            mime_type=row["mime_type"],
            document_type=row["document_type"],
            language=row["language"],
            ocr=(
                OCRResult.model_validate_json(
                    row["ocr_json"]
                )
                if row["ocr_json"]
                else None
            ),
            created_at=row["created_at"],
            processing_status=ProcessingStatus(
                row["processing_status"]
            ),
            processing_error=row["processing_error"],
        )