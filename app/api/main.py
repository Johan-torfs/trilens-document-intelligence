import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analysis import router as analysis_router
from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.search import router as search_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="TriLens Document Intelligence API",
        description=(
            "API voor documentindexering, visuele retrieval, "
            "captioning en experimentele documentanalyse."
        ),
        version="0.1.0",
    )

    frontend_origins = [
        origin.strip()
        for origin in os.getenv(
            "TRILENS_CORS_ORIGINS",
            "http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(analysis_router)

    return app


app = create_app()