"""
Chat API — the primary citizen interaction endpoint.

Handles:
- Chat-first: message only
- Document-first: document already uploaded, then chat
- Combined: message + document text together

Persists every conversation and message to SQLite.
"""

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.core.config import settings
from backend.app.schemas.chat import ChatRequest, ChatResponse, SourceItem
from backend.app.schemas.document import ExtractedDocumentInfo
from backend.app.services.database.repositories import (
    ConversationRepository,
    DocumentRepository,
    MessageRepository,
    CitationRepository,
)
from backend.app.services.ingestion.document_processor import classify_document_type
from backend.app.services.ingestion.pdf_extractor import PDFExtractor
from backend.app.services.rag.rag_service import RAGService

router = APIRouter()

# Singletons
_rag = RAGService.get_instance()
_pdf = PDFExtractor()
_conv_repo = ConversationRepository()
_msg_repo = MessageRepository()
_doc_repo = DocumentRepository()
_cit_repo = CitationRepository()


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main legal assistance chat endpoint.

    Accepts a message and optional conversation_id.
    Creates a new conversation if no conversation_id is supplied.
    Stores the message and response in SQLite.
    """
    # --- Conversation management ---
    conversation_id = request.conversation_id
    if conversation_id:
        conv = _conv_repo.get(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Auto-create conversation with a title derived from the first message
        title = _derive_title(request.message)
        conv = _conv_repo.create(
            title=title,
            language=request.language,
            jurisdiction=request.state,
        )
        conversation_id = conv["conversation_id"]

    # --- Build conversation history ---
    # Use history from request if provided, otherwise load from DB
    if request.history:
        history_list = [{"role": m.role, "content": m.content} for m in request.history]
    else:
        recent = _msg_repo.get_recent(conversation_id, n=12)
        history_list = [{"role": m["role"], "content": m["content"]} for m in recent]

    # --- Document context ---
    doc_text = request.document_text
    if not doc_text and request.document_id:
        doc = _doc_repo.get(request.document_id)
        if doc:
            # Retrieve stored extracted text from document facts
            facts = _doc_repo.get_facts(request.document_id)
            if facts:
                doc_text = "\n".join(f["extracted_value"] for f in facts if f["fact_type"] == "extracted_text")

    # --- RAG generation ---
    try:
        result = _rag.generate_response(
            query=request.message,
            conversation_history=history_list,
            language=request.language,
            state=request.state,
            document_text=doc_text,
            legal_domain=request.legal_domain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG generation failed: {exc}")

    answer = result["answer"]
    sources = result["sources"]
    has_context = result["has_context"]
    follow_up = result.get("follow_up_question")
    citations = result.get("citations", [])

    # --- Persist messages ---
    user_msg = _msg_repo.add(
        conversation_id=conversation_id,
        role="user",
        content=request.message,
        has_context=False,
    )
    assistant_msg = _msg_repo.add(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        has_context=has_context,
    )
    _conv_repo.touch(conversation_id)

    # --- Persist citations ---
    for cit in citations:
        try:
            _cit_repo.add(
                message_id=assistant_msg["message_id"],
                source_id=cit.get("source_id"),
                chunk_id=cit.get("chunk_id"),
                retrieval_score=cit.get("score"),
                page_number=cit.get("page_number"),
                section_reference=cit.get("section_reference"),
                is_verified=cit.get("is_verified", False),
            )
        except Exception:
            pass  # Citation persistence failure is non-fatal

    # --- Update conversation title if it's still default ---
    if conv["title"] == "New conversation":
        _conv_repo.update_title(conversation_id, _derive_title(request.message))

    return ChatResponse(
        response=answer,
        conversation_id=conversation_id,
        message_id=assistant_msg["message_id"],
        sources=[SourceItem(**s) for s in sources],
        follow_up_question=follow_up,
        has_grounded_context=has_context,
        legal_domain=request.legal_domain,
    )


# ---------------------------------------------------------------------------
# Document extraction endpoint
# ---------------------------------------------------------------------------

@router.post("/extract-document", response_model=ExtractedDocumentInfo)
async def extract_document(file: UploadFile = File(...)):
    """
    Extract text from an uploaded PDF or image document.

    For PDFs: uses PyMuPDF.
    For images: uses Tesseract OCR if available, otherwise returns a note.

    The extracted text is NOT stored here — it's returned to the frontend
    to include in the next chat message. For persistent storage, use
    POST /api/v1/documents/upload instead.
    """
    filename = file.filename or "unknown"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    allowed = settings.ALLOWED_UPLOAD_EXTENSIONS
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(allowed)}",
        )

    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit.",
        )

    if ext == "pdf":
        extracted_text, page_count, doc_type = _extract_pdf_bytes(content, filename)
    else:
        extracted_text, page_count, doc_type = _extract_image_bytes(content, filename)

    return ExtractedDocumentInfo(
        filename=filename,
        file_type=ext,
        extracted_text=extracted_text,
        page_count=page_count,
        document_type=doc_type,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pdf_bytes(content: bytes, filename: str):
    """Extract text from PDF bytes using PyMuPDF."""
    try:
        import pymupdf
        doc = pymupdf.open(stream=content, filetype="pdf")
        page_count = len(doc)
        pages_text = []
        for i, page in enumerate(doc, start=1):
            txt = page.get_text("text").strip()
            if txt:
                pages_text.append(f"--- Page {i} ---\n{txt}")
        full_text = "\n\n".join(pages_text)
        if not full_text.strip():
            full_text = "[PDF appears to contain only scanned images. OCR processing required.]"
        doc_type = classify_document_type(full_text, filename)
        return full_text, page_count, doc_type
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}")


def _extract_image_bytes(content: bytes, filename: str):
    """Extract text from an image using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(content))
        # Convert to RGB if needed (handles RGBA, palette modes)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        extracted = pytesseract.image_to_string(img, lang="eng")
        if not extracted.strip():
            extracted = "[No text detected in image. The document may be blank or non-textual.]"
        return extracted, 1, "scanned_document"
    except ImportError:
        return (
            f"[Image OCR is not available. Install Tesseract and pytesseract to enable image text extraction. "
            f"File: {filename}, size: {len(content)} bytes]",
            1,
            "image_document",
        )
    except Exception as exc:
        return f"[Image processing failed: {exc}]", 1, "image_document"


def _derive_title(message: str, max_len: int = 60) -> str:
    """Generate a short conversation title from the first message."""
    # Remove punctuation, clean up whitespace
    clean = re.sub(r"[^\w\s]", "", message).strip()
    if len(clean) <= max_len:
        return clean.capitalize()
    return clean[:max_len].rsplit(" ", 1)[0].capitalize() + "..."
