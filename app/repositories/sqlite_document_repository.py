import sqlite3
import json
from pathlib import Path
from app.repositories.document_repository import DocumentRepository
from app.domain.document import (
    ArtifactType,
    DocumentMetadata,
    DocumentRecord,
    ModelArtifact,
    ProcessingStatus,
)


class DuplicateDocumentError(ValueError):
    pass


class SQLiteDocumentRepository(DocumentRepository):
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY
                );

                INSERT OR IGNORE INTO schema_version (version)
                VALUES (1);
                
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    checksum TEXT NOT NULL UNIQUE,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    mime_type TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    caption TEXT,
                    retrieval_model TEXT,
                    caption_model TEXT,
                    embedding_path TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    processing_error TEXT
                );

                CREATE TABLE IF NOT EXISTS model_artifacts (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    storage_path TEXT,
                    content TEXT,
                    dimensions INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id)
                        REFERENCES documents(id)
                        ON DELETE CASCADE
                );
                """
            )

    def save_document(self, document: DocumentRecord) -> None:
        existing = self.get_document_by_checksum(document.checksum)

        if existing and existing.id != document.id:
            raise DuplicateDocumentError(
                f"Document is duplicate van '{existing.id}'"
            )
        
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id,
                    original_filename,
                    stored_path,
                    checksum,
                    width,
                    height,
                    mime_type,
                    document_type,
                    caption,
                    retrieval_model,
                    caption_model,
                    embedding_path,
                    metadata_json,
                    created_at,
                    processing_status,
                    processing_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.original_filename,
                    document.stored_path,
                    document.checksum,
                    document.width,
                    document.height,
                    document.mime_type,
                    document.document_type,
                    document.caption,
                    document.retrieval_model,
                    document.caption_model,
                    document.embedding_path,
                    document.metadata.model_dump_json(),
                    document.created_at.isoformat(),
                    document.processing_status.value,
                    document.processing_error,
                ),
            )

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            original_filename=row["original_filename"],
            stored_path=row["stored_path"],
            checksum=row["checksum"],
            width=row["width"],
            height=row["height"],
            mime_type=row["mime_type"],
            document_type=row["document_type"],
            caption=row["caption"],
            retrieval_model=row["retrieval_model"],
            caption_model=row["caption_model"],
            embedding_path=row["embedding_path"],
            metadata=DocumentMetadata.model_validate_json(
                row["metadata_json"]
            ),
            created_at=row["created_at"],
            processing_status=ProcessingStatus(
                row["processing_status"]
            ),
            processing_error=row["processing_error"],
        )

    def _row_to_artifact(self, row: sqlite3.Row) -> ModelArtifact:
        return ModelArtifact(
            id=row["id"],
            document_id=row["document_id"],
            artifact_type=ArtifactType(row["artifact_type"]),
            model_name=row["model_name"],
            model_version=row["model_version"],
            storage_path=row["storage_path"],
            content=row["content"],
            dimensions=row["dimensions"],
            created_at=row["created_at"],
        )

    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
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
                "SELECT * FROM documents WHERE checksum = ?",
                (checksum,),
            ).fetchone()

        return self._row_to_document(row) if row else None

    def save_artifact(self, artifact: ModelArtifact) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO model_artifacts (
                    id,
                    document_id,
                    artifact_type,
                    model_name,
                    model_version,
                    storage_path,
                    content,
                    dimensions,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.document_id,
                    artifact.artifact_type.value,
                    artifact.model_name,
                    artifact.model_version,
                    artifact.storage_path,
                    artifact.content,
                    artifact.dimensions,
                    artifact.created_at.isoformat(),
                ),
            )

    def get_artifacts(
        self,
        document_id: str,
    ) -> list[ModelArtifact]:
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT *
                FROM model_artifacts
                WHERE document_id = ?
                ORDER BY created_at
                """,
                (document_id,),
            ).fetchall()

        return [
            self._row_to_artifact(row)
            for row in rows
        ]

    def update_processing_status(
        self,
        document_id: str,
        status: ProcessingStatus,
        error: str | None = None,
    ) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE documents
                SET processing_status = ?,
                    processing_error = ?
                WHERE id = ?
                """,
                (status.value, error, document_id),
            )

    def find_artifacts(
        self,
        artifact_type: ArtifactType,
        model_name: str,
        document_type: str | None = None,
    ) -> list[ModelArtifact]:
        query = """
            SELECT model_artifacts.*
            FROM model_artifacts
            JOIN documents
                ON documents.id = model_artifacts.document_id
            WHERE model_artifacts.artifact_type = ?
            AND model_artifacts.model_name = ?
        """

        parameters: list[str] = [
            artifact_type.value,
            model_name,
        ]

        if document_type is not None:
            query += """
            AND documents.document_type = ?
            """
            parameters.append(document_type)

        query += """
            ORDER BY model_artifacts.created_at ASC
        """

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            
            rows = connection.execute(
                query,
                parameters,
            ).fetchall()

        return [
            self._row_to_artifact(row)
            for row in rows
        ]
    