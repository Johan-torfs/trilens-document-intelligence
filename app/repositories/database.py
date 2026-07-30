from pathlib import Path

from app.repositories.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from app.repositories.vector_repository import VectorRepository


def create_repositories() -> tuple[
    SQLiteDocumentRepository,
    VectorRepository,
]:
    document_repository = SQLiteDocumentRepository(
        Path("data/trilens.db")
    )
    document_repository.initialize()

    vector_repository = VectorRepository(
        Path("data/vectors")
    )

    return document_repository, vector_repository