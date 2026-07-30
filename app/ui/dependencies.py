import os
from pathlib import Path

import streamlit as st

from app.bootstrap import create_document_intelligence_pipeline
from app.repositories.sqlite_document_repository import (
    SQLiteDocumentRepository,
)
from app.repositories.vector_repository import VectorRepository
from app.services.document_intelligence_pipeline import (
    DocumentIntelligencePipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
DATABASE_PATH = RUNTIME_DIR / "trilens.sqlite3"
VECTOR_DIR = RUNTIME_DIR / "vectors"


@st.cache_resource
def get_pipeline() -> DocumentIntelligencePipeline:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    document_repository = SQLiteDocumentRepository(
        database_path=DATABASE_PATH,
    )

    vector_repository = VectorRepository(
        storage_dir=VECTOR_DIR,
    )

    open_flamingo_enabled = (
        os.getenv(
            "TRILENS_OPEN_FLAMINGO_ENABLED",
            "false",
        ).lower()
        == "true"
    )

    open_flamingo_device = os.getenv(
        "TRILENS_OPEN_FLAMINGO_DEVICE",
        "cpu",
    )

    return create_document_intelligence_pipeline(
        document_repository=document_repository,
        vector_repository=vector_repository,
        open_flamingo_enabled=open_flamingo_enabled,
        open_flamingo_device=open_flamingo_device,
    )