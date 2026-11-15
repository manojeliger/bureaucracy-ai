"""
Keyword-based retrieval using SQLite FTS5.

Complements semantic FAISS search with exact-match and keyword-match
retrieval — critical for queries that name specific Acts, sections,
or government portal names (e.g. "Section 14 of Payment of Wages Act",
"Sakala", "Nadakacheri").
"""

import logging
from typing import Any, Dict, List, Optional

from backend.app.services.database.repositories import LegalChunkRepository

logger = logging.getLogger(__name__)


class KeywordSearchService:
    """
    Wraps SQLite FTS5 full-text search over the legal_chunks table.

    Results are returned in the same dict structure as semantic results
    so the retrieval service can merge them uniformly.
    """

    def __init__(self):
        self._repo = LegalChunkRepository()

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run FTS5 keyword search and return structured results.

        Each result dict has keys:
            chunk_id, chunk_text, source_id, section_reference,
            page_number, keyword_score (float, lower FTS rank = better)
        """
        if not query or not query.strip():
            return []

        # Sanitize query for FTS5 — escape special characters
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        try:
            rows = self._repo.keyword_search(safe_query, top_k=top_k, source_id=source_id)
        except Exception as exc:
            logger.warning(f"Keyword search failed for query '{query}': {exc}")
            return []

        results = []
        for row in rows:
            # FTS5 rank is negative (more negative = more relevant)
            # Normalize to a 0–1 score (approximate)
            raw_rank = row.get("rank", -1.0)
            keyword_score = max(0.0, min(1.0, 1.0 / (1.0 + abs(raw_rank))))

            results.append({
                "chunk_id": row.get("chunk_id", ""),
                "chunk_text": row.get("chunk_text", ""),
                "source_id": row.get("source_id", ""),
                "section_reference": row.get("section_reference"),
                "page_number": row.get("page_number"),
                "chunk_index": row.get("chunk_index", 0),
                "keyword_score": keyword_score,
                "retrieval_method": "keyword_fts5",
            })

        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Remove FTS5 special characters that would cause syntax errors.
        Preserves quoted phrases if the user typed them.
        """
        # Remove characters that break FTS5 parsing
        bad_chars = ['"', "'", "*", "^", "~", "(", ")", "[", "]", "{", "}", "\\"]
        for ch in bad_chars:
            query = query.replace(ch, " ")
        query = " ".join(query.split())
        return query.strip()
