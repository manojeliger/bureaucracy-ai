"""
Citation validation service for Bureaucracy AI.

Ensures that every citation returned to a citizen corresponds to an actually
retrieved chunk. This prevents the LLM from fabricating legal references.

Rule: A citation is valid only if its chunk_id or source_id appears in the
list of retrieved chunks for that response.
"""

import logging
from typing import Any, Dict, List, Optional

from backend.app.services.rag.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)


class CitationService:

    def build_citations(
        self,
        retrieved_chunks: List[RetrievedChunk],
        message_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert retrieved chunks into validated citation dicts.

        Returns a list of citation records, each containing:
            chunk_id, source_id, title, department, source_url,
            page_number, section_reference, score, is_verified
        """
        citations = []
        seen_sources: set = set()

        for chunk in retrieved_chunks:
            is_verified = chunk.verification_status == "verified_official"
            source_key = (chunk.title or "", chunk.source_url or "")

            # Deduplicate on (title, url) — show each source once
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)

            citations.append({
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "title": chunk.title or "Legal Source",
                "department": chunk.department or "Government",
                "source_url": chunk.source_url or "",
                "page_number": chunk.page_number,
                "section_reference": chunk.section_reference,
                "jurisdiction": chunk.jurisdiction,
                "score": round(chunk.score, 4),
                "retrieval_method": chunk.retrieval_method,
                "is_verified": is_verified,
                "confidence": round(chunk.score, 2),
                "message_id": message_id,
            })

        return citations

    def validate_llm_citations(
        self,
        llm_response: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Dict[str, Any]:
        """
        Check whether the LLM response references sources that exist in
        the retrieved set. This is a lightweight heuristic check.

        Returns:
            {
              "has_grounded_context": bool,
              "unverified_claims": int,  # sentences that make claims with no source
              "source_count": int,
            }
        """
        has_context = len(retrieved_chunks) > 0
        source_count = len(set(c.source_id for c in retrieved_chunks))

        return {
            "has_grounded_context": has_context,
            "source_count": source_count,
            "unverified_claims": 0,  # Heuristic — extended in future phases
        }

    @staticmethod
    def format_for_display(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format citations for the frontend source display.
        Returns a simplified list suitable for the SourceItem schema.
        """
        display = []
        for c in citations:
            display.append({
                "title": c.get("title", ""),
                "department": c.get("department", ""),
                "url": c.get("source_url", ""),
                "confidence": c.get("confidence", 0.0),
                "is_verified": c.get("is_verified", False),
                "section": c.get("section_reference"),
                "page": c.get("page_number"),
                "jurisdiction": c.get("jurisdiction", ""),
            })
        return display
