"""
Legal search API — direct access to the RAG retrieval layer.
"""

from fastapi import APIRouter, HTTPException
from backend.app.schemas.legal import LegalSearchRequest, LegalSearchResponse, LegalChunkResult
from backend.app.services.rag.rag_service import RAGService

router = APIRouter()
_rag = RAGService.get_instance()


@router.post("/legal/search", response_model=LegalSearchResponse)
def legal_search(request: LegalSearchRequest):
    """
    Direct legal knowledge base search.
    Returns the top-k most relevant chunks with full metadata.
    """
    try:
        chunks = _rag.retrieve(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            jurisdiction=request.jurisdiction,
            legal_domain=request.legal_domain,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    results = [
        LegalChunkResult(
            chunk_id=c.chunk_id or "",
            text=c.text or "",
            title=c.title or "",
            department=c.department or "",
            source_url=c.source_url or "",
            page_number=c.page_number,
            section_reference=c.section_reference,
            jurisdiction=c.jurisdiction or "",
            score=round(c.score, 4),
            retrieval_method=c.retrieval_method or "semantic",
            is_verified=c.verification_status == "verified_official",
        )
        for c in chunks
    ]

    return LegalSearchResponse(
        query=request.query,
        results=results,
        total_found=len(results),
        has_context=len(results) > 0,
    )
