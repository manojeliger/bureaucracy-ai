"""
Request and response schemas for legal search, reports, and admin.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LegalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    jurisdiction: Optional[str] = "Karnataka"
    legal_domain: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.60, ge=0.0, le=1.0)


class LegalChunkResult(BaseModel):
    chunk_id: str
    text: str
    title: str
    department: str
    source_url: str
    page_number: Optional[int] = None
    section_reference: Optional[str] = None
    jurisdiction: str
    score: float
    retrieval_method: str
    is_verified: bool = False


class LegalSearchResponse(BaseModel):
    query: str
    results: List[LegalChunkResult]
    total_found: int
    has_context: bool


class ReportRequest(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    include_sources: bool = True


class ReportResponse(BaseModel):
    report_id: str
    title: str
    conversation_id: str
    file_path: Optional[str] = None
    created_at: str
    download_url: Optional[str] = None


class ScrapingJobRequest(BaseModel):
    source_name: str
    source_url: Optional[str] = None


class ScrapingJobStatus(BaseModel):
    job_id: str
    source_name: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    documents_found: int = 0
    documents_downloaded: int = 0
    error_details: Optional[str] = None
