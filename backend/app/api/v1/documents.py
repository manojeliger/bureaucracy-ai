"""
Document upload and management API.

Handles secure storage of citizen-uploaded legal documents.
Documents are stored outside the web root with sanitized filenames.
"""

import hashlib
import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.schemas.document import (
    DocumentDetailResponse,
    DocumentFactResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from backend.app.services.database.repositories import DocumentRepository
from backend.app.services.ingestion.document_processor import classify_document_type
from backend.app.services.ingestion.pdf_extractor import PDFExtractor

router = APIRouter()
_doc_repo = DocumentRepository()
_pdf = PDFExtractor()
_uploads_dir = Path(settings.UPLOADS_DIR)
_uploads_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    conversation_id: str = Form(default=None),
):
    """
    Securely upload a legal document and extract its text.

    - Validates file type and size.
    - Saves with a UUID-based safe filename (no path traversal risk).
    - Extracts text immediately.
    - Stores metadata in SQLite.
    """
    filename = file.filename or "document"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Supported: {settings.ALLOWED_UPLOAD_EXTENSIONS}",
        )

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size {len(content)} bytes exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    # Safe filename: uuid + extension only
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    file_path = _uploads_dir / safe_name

    # Save to disk
    with open(file_path, "wb") as f:
        f.write(content)

    # Compute checksum
    checksum = hashlib.sha256(content).hexdigest()

    # Create DB record
    doc = _doc_repo.create(
        filename=filename,
        safe_filename=safe_name,
        file_path=str(file_path),
        file_type=ext,
        file_size_bytes=len(content),
        checksum=checksum,
        conversation_id=conversation_id,
    )
    document_id = doc["document_id"]

    # Extract text
    extracted_text = ""
    page_count = 0
    doc_type = "unknown"

    try:
        _doc_repo.update_processing(document_id, status="processing")

        if ext == "pdf":
            extracted_text, page_count, doc_type = _extract_pdf(content, filename)
        else:
            extracted_text, page_count, doc_type = _extract_image(content, filename)

        # Store extracted text as a document fact
        if extracted_text.strip():
            _doc_repo.add_fact(
                document_id=document_id,
                fact_type="extracted_text",
                extracted_value=extracted_text,
                page_number=None,
                verification_status="unverified",
            )

        _doc_repo.update_processing(
            document_id=document_id,
            status="completed",
            document_type=doc_type,
            page_count=page_count,
        )
    except Exception as exc:
        _doc_repo.update_processing(document_id, status="failed")
        # Still return the record but note the failure
        extracted_text = f"[Text extraction failed: {exc}]"

    from datetime import datetime, timezone
    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        file_type=ext,
        file_size_bytes=len(content),
        processing_status="completed" if extracted_text else "failed",
        page_count=page_count,
        extracted_text=extracted_text[:5000],  # Return preview only
        document_type=doc_type,
        upload_timestamp=doc["upload_timestamp"],
    )


# ---------------------------------------------------------------------------
# Status and detail
# ---------------------------------------------------------------------------

@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str):
    """Get document metadata and extracted facts."""
    doc = _doc_repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    facts = _doc_repo.get_facts(document_id)
    # Exclude raw extracted_text from facts list (too long for API response)
    display_facts = [
        DocumentFactResponse(
            fact_type=f["fact_type"],
            extracted_value=f["extracted_value"][:500],
            page_number=f.get("page_number"),
            verification_status=f.get("verification_status", "unverified"),
        )
        for f in facts
        if f["fact_type"] != "extracted_text"
    ]

    return DocumentDetailResponse(**doc, facts=display_facts)


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str):
    """Check processing status of a document."""
    doc = _doc_repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(**doc)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

@router.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """Delete a document record and its file from disk."""
    doc = _doc_repo.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete file from disk
    file_path = Path(doc.get("file_path", ""))
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass  # Non-fatal — record is still removed

    deleted = _doc_repo.delete(document_id)
    return {"deleted": deleted, "document_id": document_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_pdf(content: bytes, filename: str):
    try:
        import pymupdf
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = len(doc)
        pages = []
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text").strip()
            if txt:
                pages.append(f"--- Page {i} ---\n{txt}")
        full_text = "\n\n".join(pages)
        if not full_text.strip():
            full_text = "[This PDF appears to contain scanned images. OCR required for text extraction.]"
        doc_type = classify_document_type(full_text, filename)
        return full_text, page_count, doc_type
    except Exception as exc:
        return f"[PDF extraction failed: {exc}]", 0, "unknown"


def _extract_image(content: bytes, filename: str):
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(content))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang="eng")
        doc_type = classify_document_type(text, filename)
        return text or "[No text detected]", 1, doc_type
    except ImportError:
        return (
            f"[Tesseract OCR not installed. Image: {filename}]",
            1,
            "image_document",
        )
    except Exception as exc:
        return f"[Image OCR failed: {exc}]", 1, "image_document"
