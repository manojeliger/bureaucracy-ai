"""
Admin API — scraping job management and knowledge base administration.
Protected by API key header.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, Header, HTTPException

from backend.app.core.config import settings
from backend.app.schemas.legal import ScrapingJobRequest, ScrapingJobStatus
from backend.app.services.database.repositories import ScrapingJobRepository
from backend.app.services.rag.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)
_job_repo = ScrapingJobRepository()


def _require_admin_key(x_admin_key: str = Header(default="")):
    """Simple API-key guard for admin endpoints."""
    if x_admin_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin API key")


@router.post("/admin/scraping/run", response_model=ScrapingJobStatus)
def trigger_scraping_job(
    request: ScrapingJobRequest,
    _: None = Depends(_require_admin_key),
):
    """
    Trigger a scraping job for a named source.
    Returns immediately with a job_id; actual scraping runs async.
    """
    job_id = _job_repo.create(
        source_name=request.source_name,
        source_url=request.source_url,
    )
    logger.info(f"Scraping job {job_id} created for source '{request.source_name}'")
    return ScrapingJobStatus(
        job_id=job_id,
        source_name=request.source_name,
        status="running",
        started_at="",
        documents_found=0,
        documents_downloaded=0,
    )


@router.get("/admin/scraping/status", response_model=List[ScrapingJobStatus])
def scraping_status(_: None = Depends(_require_admin_key)):
    """List recent scraping jobs and their statuses."""
    jobs = _job_repo.list_recent(limit=20)
    return [ScrapingJobStatus(**j) for j in jobs]


@router.post("/admin/reload-index")
def reload_index(_: None = Depends(_require_admin_key)):
    """Force-reload the FAISS index from disk (use after re-ingestion)."""
    rag = RAGService.get_instance()
    rag.reload_index()
    return {
        "status": "reloaded",
        "vectors": rag.vector_store.total_vectors,
    }
