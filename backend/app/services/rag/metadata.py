from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    department: str
    source_url: str

    publication_date: Optional[date] = None
    document_type: Optional[str] = None
    language: str = "en"


class ChunkMetadata(BaseModel):
    document: DocumentMetadata
    page_number: Optional[int] = None
    chunk_index: int = Field(ge=0)
    total_chunks: int = Field(gt=0)