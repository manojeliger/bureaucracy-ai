"""
Repository layer for Bureaucracy AI — all SQLite CRUD operations.

Each repository class owns one or more related tables.
They accept and return plain dicts or typed Pydantic models.
No business logic lives here — only database access.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.app.services.database.connection import get_connection

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str = "") -> str:
    uid = str(uuid.uuid4())
    return f"{prefix}{uid}" if prefix else uid


# ---------------------------------------------------------------------------
# Conversation repository
# ---------------------------------------------------------------------------

class ConversationRepository:

    def create(
        self,
        title: str = "New conversation",
        legal_domain: Optional[str] = None,
        jurisdiction: str = "Karnataka",
        language: str = "English",
    ) -> Dict[str, Any]:
        now = _now_iso()
        cid = _new_id("conv-")
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversations
                    (conversation_id, title, legal_domain, jurisdiction, language, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, title, legal_domain, jurisdiction, language, now, now),
            )
        return self.get(cid)

    def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_title(self, conversation_id: str, title: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                (title, _now_iso(), conversation_id),
            )

    def touch(self, conversation_id: str) -> None:
        """Update the updated_at timestamp (call after adding a message)."""
        with get_connection() as conn:
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (_now_iso(), conversation_id),
            )

    def delete(self, conversation_id: str) -> bool:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Message repository
# ---------------------------------------------------------------------------

class MessageRepository:

    def add(
        self,
        conversation_id: str,
        role: str,
        content: str,
        has_context: bool = False,
    ) -> Dict[str, Any]:
        mid = _new_id("msg-")
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages
                    (message_id, conversation_id, role, content, has_context, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (mid, conversation_id, role, content, int(has_context), now),
            )
        return {"message_id": mid, "conversation_id": conversation_id,
                "role": role, "content": content, "has_context": has_context, "created_at": now}

    def list_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_recent(self, conversation_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """Return the n most recent messages (chronological order)."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
                """,
                (conversation_id, n),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Document repository
# ---------------------------------------------------------------------------

class DocumentRepository:

    def create(
        self,
        filename: str,
        safe_filename: str,
        file_path: str,
        file_type: str,
        file_size_bytes: int,
        checksum: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        did = _new_id("doc-")
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (document_id, conversation_id, filename, safe_filename,
                     file_path, file_type, file_size_bytes, checksum,
                     processing_status, upload_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (did, conversation_id, filename, safe_filename,
                 file_path, file_type, file_size_bytes, checksum, now),
            )
        return self.get(did)

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_processing(
        self,
        document_id: str,
        status: str,
        document_type: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> None:
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET processing_status = ?,
                    document_type = COALESCE(?, document_type),
                    page_count = COALESCE(?, page_count),
                    processed_at = ?
                WHERE document_id = ?
                """,
                (status, document_type, page_count, now, document_id),
            )

    def list_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE conversation_id = ? ORDER BY upload_timestamp DESC",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete(self, document_id: str) -> bool:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (document_id,),
            )
        return cur.rowcount > 0

    def add_fact(
        self,
        document_id: str,
        fact_type: str,
        extracted_value: str,
        page_number: Optional[int] = None,
        verification_status: str = "unverified",
    ) -> None:
        fid = _new_id("fact-")
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO document_facts
                    (fact_id, document_id, fact_type, extracted_value, page_number, verification_status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fid, document_id, fact_type, extracted_value, page_number, verification_status),
            )

    def get_facts(self, document_id: str) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_facts WHERE document_id = ?",
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Legal source repository
# ---------------------------------------------------------------------------

class LegalSourceRepository:

    def upsert(self, source: Dict[str, Any]) -> str:
        """Insert or replace a legal source record. Returns source_id."""
        sid = source.get("source_id") or _new_id("src-")
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO legal_sources
                    (source_id, title, document_type, legal_domain, jurisdiction,
                     state, act_name, court_name, department, source_url,
                     publication_date, commencement_date, effective_from,
                     effective_until, version, amendment_status, language,
                     file_path, verification_status, retrieved_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    source.get("title", ""),
                    source.get("document_type", "unknown"),
                    source.get("legal_domain", "unknown"),
                    source.get("jurisdiction", "Karnataka"),
                    source.get("state", "Karnataka"),
                    source.get("act_name"),
                    source.get("court_name"),
                    source.get("department", ""),
                    source.get("source_url", ""),
                    source.get("publication_date"),
                    source.get("commencement_date"),
                    source.get("effective_from"),
                    source.get("effective_until"),
                    source.get("version", "1.0"),
                    source.get("amendment_status"),
                    source.get("language", "en"),
                    source.get("file_path"),
                    source.get("verification_status", "unverified"),
                    source.get("retrieved_at"),
                    now,
                ),
            )
        return sid

    def get(self, source_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM legal_sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return dict(row) if row else None

    def search_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM legal_sources WHERE source_url = ?",
                (url,),
            ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Legal chunk repository (also manages FTS index)
# ---------------------------------------------------------------------------

class LegalChunkRepository:

    def upsert(self, chunk: Dict[str, Any]) -> None:
        """Insert or replace a chunk record and update the FTS index."""
        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO legal_chunks
                    (chunk_id, source_id, chunk_text, section_reference,
                     chapter_reference, page_number, chunk_index, total_chunks,
                     embedding_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk["chunk_id"],
                    chunk.get("source_id"),
                    chunk["chunk_text"],
                    chunk.get("section_reference"),
                    chunk.get("chapter_reference"),
                    chunk.get("page_number"),
                    chunk.get("chunk_index", 0),
                    chunk.get("total_chunks", 1),
                    chunk.get("embedding_reference"),
                ),
            )
            # Keep FTS index in sync
            conn.execute(
                "INSERT OR REPLACE INTO legal_chunks_fts(chunk_id, source_id, chunk_text, section_reference) VALUES (?, ?, ?, ?)",
                (
                    chunk["chunk_id"],
                    chunk.get("source_id", ""),
                    chunk["chunk_text"],
                    chunk.get("section_reference", ""),
                ),
            )

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search over chunk text using SQLite FTS5.
        Returns rows from legal_chunks joined with their FTS rank.
        """
        try:
            with get_connection() as conn:
                if source_id:
                    rows = conn.execute(
                        """
                        SELECT lc.*, fts.rank
                        FROM legal_chunks_fts fts
                        JOIN legal_chunks lc ON fts.chunk_id = lc.chunk_id
                        WHERE legal_chunks_fts MATCH ?
                          AND fts.source_id = ?
                        ORDER BY fts.rank
                        LIMIT ?
                        """,
                        (query, source_id, top_k),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT lc.*, fts.rank
                        FROM legal_chunks_fts fts
                        JOIN legal_chunks lc ON fts.chunk_id = lc.chunk_id
                        WHERE legal_chunks_fts MATCH ?
                        ORDER BY fts.rank
                        LIMIT ?
                        """,
                        (query, top_k),
                    ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.warning(f"FTS keyword search failed: {exc}")
            return []

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM legal_chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Citation repository
# ---------------------------------------------------------------------------

class CitationRepository:

    def add(
        self,
        message_id: str,
        source_id: Optional[str],
        chunk_id: Optional[str],
        retrieval_score: Optional[float] = None,
        page_number: Optional[int] = None,
        section_reference: Optional[str] = None,
        is_verified: bool = False,
    ) -> str:
        cid = _new_id("cit-")
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO citations
                    (citation_id, message_id, source_id, chunk_id, page_number,
                     section_reference, retrieval_score, is_verified, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cid, message_id, source_id, chunk_id, page_number,
                 section_reference, retrieval_score, int(is_verified), now),
            )
        return cid

    def list_for_message(self, message_id: str) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM citations WHERE message_id = ?",
                (message_id,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Scraping job repository
# ---------------------------------------------------------------------------

class ScrapingJobRepository:

    def create(self, source_name: str, source_url: Optional[str] = None) -> str:
        jid = _new_id("job-")
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO scraping_jobs (job_id, source_name, source_url, started_at, status)
                VALUES (?, ?, ?, ?, 'running')
                """,
                (jid, source_name, source_url, now),
            )
        return jid

    def complete(
        self,
        job_id: str,
        documents_found: int = 0,
        documents_downloaded: int = 0,
        documents_skipped: int = 0,
        error: Optional[str] = None,
    ) -> None:
        status = "failed" if error else "completed"
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE scraping_jobs
                SET status = ?, completed_at = ?, documents_found = ?,
                    documents_downloaded = ?, documents_skipped = ?, error_details = ?
                WHERE job_id = ?
                """,
                (status, now, documents_found, documents_downloaded,
                 documents_skipped, error, job_id),
            )

    def list_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scraping_jobs ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Report repository
# ---------------------------------------------------------------------------

class ReportRepository:

    def create(
        self,
        title: str,
        conversation_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> str:
        rid = _new_id("rpt-")
        now = _now_iso()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO reports (report_id, conversation_id, title, file_path, created_at) VALUES (?, ?, ?, ?, ?)",
                (rid, conversation_id, title, file_path, now),
            )
        return rid

    def get(self, report_id: str) -> Optional[Dict[str, Any]]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM reports WHERE conversation_id = ? ORDER BY created_at DESC",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]
