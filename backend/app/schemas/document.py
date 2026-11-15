"""
Request and response schemas for document upload and processing.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    processing_status: str
    page_count: int = 0
    extracted_text: Optional[str] = None
    document_type: Optional[str] = None
    upload_timestamp: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    processing_status: str
    page_count: int = 0
    document_type: Optional[str] = None
    upload_timestamp: str
    processed_at: Optional[str] = None


class DocumentFactResponse(BaseModel):
    fact_type: str
    extracted_value: str
    page_number: Optional[int] = None
    verification_status: str


class DocumentDetailResponse(DocumentStatusResponse):
    facts: List[DocumentFactResponse] = []


class ExtractedDocumentInfo(BaseModel):
    """Lightweight extraction result for inline use in chat."""
    filename: str
    file_type: str
    extracted_text: str
    page_count: int
    document_type: str = "unknown"
