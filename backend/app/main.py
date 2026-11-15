"""
Bureaucracy AI — FastAPI application entry point.

Registers all routers and runs database initialization on startup.
"""

import logging

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1 import admin, chat, conversations, documents, legal_search, reports
from backend.app.core.config import settings
from backend.app.services.database.connection import init_db
from backend.app.services.rag.rag_service import RAGService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Bureaucracy AI",
    description=(
        "AI-Powered Legal Intelligence & Citizen Rights Assistance System. "
        "Karnataka, India — Phase 1."
    ),
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# CORS — allow Streamlit frontend
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    """Initialize the database and warm up the RAG service on startup."""
    logger.info("Bureaucracy AI starting up...")
    try:
        init_db()
        logger.info("SQLite database ready")
    except Exception as exc:
        logger.error(f"Database initialization failed: {exc}")

    # Warm up RAG (loads FAISS index + embedding model)
    try:
        rag = RAGService.get_instance()
        logger.info(
            f"RAG service ready — {rag.vector_store.total_vectors} vectors indexed"
        )
    except Exception as exc:
        logger.error(f"RAG service startup failed: {exc}")


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(chat.router, prefix=API_PREFIX, tags=["Chat"])
app.include_router(conversations.router, prefix=API_PREFIX, tags=["Conversations"])
app.include_router(documents.router, prefix=API_PREFIX, tags=["Documents"])
app.include_router(legal_search.router, prefix=API_PREFIX, tags=["Legal Search"])
app.include_router(reports.router, prefix=API_PREFIX, tags=["Reports"])
app.include_router(admin.router, prefix=API_PREFIX, tags=["Admin"])


# ---------------------------------------------------------------------------
# Root + Health
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Bureaucracy AI",
        "version": "0.2.0",
        "scope": "Karnataka, India",
        "model": settings.LLM_MODEL,
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """
    Comprehensive health check — Ollama, model, FAISS index, SQLite.
    """
    ollama_ok = False
    model_found = False
    ollama_error = None

    try:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            ollama_ok = True
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            model_found = any(settings.LLM_MODEL in m for m in models)
    except Exception as exc:
        ollama_error = str(exc)

    rag = RAGService.get_instance()
    faiss_ok = rag.is_index_loaded
    vector_count = rag.vector_store.total_vectors if faiss_ok else 0

    from backend.app.services.database.connection import get_db_path
    db_path = get_db_path()
    db_ok = db_path.exists()

    all_healthy = ollama_ok and model_found and faiss_ok and db_ok

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": {
            "ollama_api": "connected" if ollama_ok else "unreachable",
            "llm_model": "available" if model_found else f"'{settings.LLM_MODEL}' not loaded",
            "faiss_index": "loaded" if faiss_ok else "not_loaded",
            "indexed_vectors": vector_count,
            "embedding_model": settings.EMBEDDING_MODEL_NAME,
            "sqlite_database": "ready" if db_ok else "missing",
        },
        "error": ollama_error,
    }
