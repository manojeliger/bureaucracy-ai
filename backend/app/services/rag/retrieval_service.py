"""
Retrieval service for Bureaucracy AI — the brain of the RAG pipeline.

Combines:
  1. Semantic vector search (FAISS)
  2. Keyword full-text search (SQLite FTS5)
  3. Metadata filtering (jurisdiction, domain, verification status)
  4. Result merging, deduplication, and ranking

Returns structured retrieval results that the LLM can cite directly.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.services.rag.embedding_service import EmbeddingService
from backend.app.services.rag.keyword_search import KeywordSearchService
from backend.app.services.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure returned by this service
# ---------------------------------------------------------------------------

class RetrievedChunk:
    """
    A single retrieved legal passage with all metadata needed for citation.
    """
    __slots__ = [
        "chunk_id", "text", "title", "department", "source_url",
        "source_id", "page_number", "section_reference", "jurisdiction",
        "legal_domain", "verification_status", "score", "retrieval_method",
    ]

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {s: getattr(self, s, None) for s in self.__slots__}


# ---------------------------------------------------------------------------
# Retrieval service
# ---------------------------------------------------------------------------

class RetrievalService:
    """
    Standalone retrieval service. This is NOT part of RAGService —
    it is injected into RAGService so retrieval can be tested independently.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        keyword_search: Optional[KeywordSearchService] = None,
    ):
        self._embedder = embedding_service
        self._vector_store = vector_store
        self._keyword = keyword_search or KeywordSearchService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        jurisdiction: Optional[str] = None,
        legal_domain: Optional[str] = None,
        require_verified: bool = False,
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid retrieval for a citizen query.

        Args:
            query:            The citizen's question or extracted legal query.
            top_k:            Maximum number of results to return.
            threshold:        Minimum semantic similarity score (0–1).
            jurisdiction:     Filter to a specific jurisdiction (e.g. "Karnataka").
            legal_domain:     Filter to a specific legal domain.
            require_verified: If True, only return chunks from verified sources.

        Returns:
            List of RetrievedChunk objects, highest relevance first.
        """
        top_k = top_k or settings.RAG_TOP_K
        threshold = threshold if threshold is not None else settings.RAG_SCORE_THRESHOLD

        # Step 1: Semantic retrieval
        semantic_results = self._semantic_search(query, top_k=top_k * 2, threshold=threshold)

        # Step 2: Keyword retrieval
        keyword_results = self._keyword_search(query, top_k=settings.RAG_KEYWORD_TOP_K)

        # Step 3: Merge and deduplicate
        merged = self._merge(semantic_results, keyword_results)

        # Step 4: Apply metadata filters
        filtered = self._apply_filters(
            merged,
            jurisdiction=jurisdiction,
            legal_domain=legal_domain,
            require_verified=require_verified,
        )

        # Step 5: Return top-k after filtering
        return filtered[:top_k]

    def has_sufficient_context(self, chunks: List[RetrievedChunk], threshold: float = 0.60) -> bool:
        """Check whether retrieved chunks provide enough grounding."""
        if not chunks:
            return False
        return any(c.score >= threshold for c in chunks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _semantic_search(
        self,
        query: str,
        top_k: int,
        threshold: float,
    ) -> List[RetrievedChunk]:
        """Run FAISS semantic similarity search."""
        if not self._vector_store.is_loaded():
            logger.warning("FAISS index is empty — skipping semantic search")
            return []

        try:
            query_embedding = self._embedder.embed_query(query)
            raw_results = self._vector_store.search(query_embedding, top_k=top_k)
        except Exception as exc:
            logger.error(f"Semantic search failed: {exc}")
            return []

        chunks = []
        for item in raw_results:
            score = item["score"]
            if score < threshold:
                continue

            meta = item["metadata"]
            chunks.append(self._build_chunk(meta, score, "semantic"))

        return chunks

    def _keyword_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Run FTS5 keyword search."""
        # Detect explicit Act/section references to boost keyword weight
        has_section_ref = bool(re.search(
            r"section\s+\d+|article\s+\d+|rule\s+\d+|act\s+\d{4}",
            query, re.IGNORECASE
        ))

        raw = self._keyword.search(query, top_k=top_k if has_section_ref else top_k // 2)

        chunks = []
        for item in raw:
            # Map keyword score to comparable range
            score = item.get("keyword_score", 0.5)
            # Build a minimal metadata dict for chunk construction
            meta = {
                "chunk_id": item.get("chunk_id", ""),
                "text": item.get("chunk_text", ""),
                "title": "",
                "department": "",
                "source_url": "",
                "source_id": item.get("source_id", ""),
                "page_number": item.get("page_number"),
                "section_reference": item.get("section_reference"),
                "jurisdiction": "",
                "legal_domain": "",
                "verification_status": "",
            }
            chunks.append(self._build_chunk(meta, score, "keyword"))

        return chunks

    def _merge(
        self,
        semantic: List[RetrievedChunk],
        keyword: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """
        Merge semantic and keyword results, deduplicating on chunk_id.
        When both methods return the same chunk, keep the higher score.
        """
        seen: Dict[str, RetrievedChunk] = {}

        for chunk in semantic:
            if chunk.chunk_id:
                seen[chunk.chunk_id] = chunk

        for chunk in keyword:
            if not chunk.chunk_id:
                continue
            if chunk.chunk_id in seen:
                # Keep whichever score is higher
                existing = seen[chunk.chunk_id]
                if chunk.score > existing.score:
                    chunk.retrieval_method = "hybrid"
                    seen[chunk.chunk_id] = chunk
            else:
                seen[chunk.chunk_id] = chunk

        # Sort by score descending
        return sorted(seen.values(), key=lambda c: c.score, reverse=True)

    @staticmethod
    def _apply_filters(
        chunks: List[RetrievedChunk],
        jurisdiction: Optional[str],
        legal_domain: Optional[str],
        require_verified: bool,
    ) -> List[RetrievedChunk]:
        """Filter chunks by metadata constraints."""
        result = []
        for chunk in chunks:
            if jurisdiction and chunk.jurisdiction:
                if chunk.jurisdiction.lower() not in (
                    jurisdiction.lower(), "central", "unknown", ""
                ):
                    continue
            if legal_domain and chunk.legal_domain:
                if chunk.legal_domain.lower() not in (legal_domain.lower(), "unknown", ""):
                    continue
            if require_verified and chunk.verification_status:
                if chunk.verification_status.lower() != "verified_official":
                    continue
            result.append(chunk)
        return result

    @staticmethod
    def _build_chunk(
        meta: Dict[str, Any],
        score: float,
        method: str,
    ) -> RetrievedChunk:
        """Build a RetrievedChunk from a flat metadata dict."""
        return RetrievedChunk(
            chunk_id=meta.get("chunk_id", ""),
            text=meta.get("text", ""),
            title=meta.get("title", ""),
            department=meta.get("department", ""),
            source_url=meta.get("source_url", ""),
            source_id=meta.get("source_id") or meta.get("chunk_id", "").split("-p")[0],
            page_number=meta.get("page_number"),
            section_reference=meta.get("section_reference"),
            jurisdiction=meta.get("jurisdiction", ""),
            legal_domain=meta.get("legal_domain", ""),
            verification_status=meta.get("verification_status", ""),
            score=score,
            retrieval_method=method,
        )
