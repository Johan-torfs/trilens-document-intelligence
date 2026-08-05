import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.repositories.document_repository import (
    DocumentRepository,
)
from app.repositories.qdrant_vector_repository import (
    QdrantVectorRepository,
)
from app.repositories.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from app.repositories.vector_repository import (
    VectorRepository,
)


@dataclass(frozen=True)
class Repositories:
    documents: DocumentRepository
    vectors: VectorRepository


@lru_cache(maxsize=1)
def get_repositories() -> Repositories:
    document_repository = SQLiteDocumentRepository(
        Path(
            os.getenv(
                "DATABASE_PATH",
                "data/trilens.db",
            )
        )
    )

    vector_repository = QdrantVectorRepository(
        url=os.getenv(
            "QDRANT_URL",
            "http://localhost:6333",
        ),
        collection_name=os.getenv(
            "QDRANT_COLLECTION",
            "trilens_vectors_v1",
        ),
        timeout_seconds=float(
            os.getenv(
                "QDRANT_TIMEOUT_SECONDS",
                "10",
            )
        ),
        exact_search=(
            os.getenv(
                "QDRANT_EXACT_SEARCH",
                "true",
            ).lower()
            == "true"
        ),
    )

    return Repositories(
        documents=document_repository,
        vectors=vector_repository,
    )