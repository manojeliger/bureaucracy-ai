"""
SQLite database connection and schema management for Bureaucracy AI.

All tables are created here on first startup. The connection uses WAL mode
for better concurrent read performance (FastAPI + background tasks).
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- -----------------------------------------------------------------------
-- Conversations
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  TEXT PRIMARY KEY,
    title            TEXT NOT NULL DEFAULT 'New conversation',
    legal_domain     TEXT,
    jurisdiction     TEXT DEFAULT 'Karnataka',
    language         TEXT DEFAULT 'English',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- -----------------------------------------------------------------------
-- Messages
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    message_id       TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content          TEXT NOT NULL,
    has_context      INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);

-- -----------------------------------------------------------------------
-- Uploaded documents (citizen uploads)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    conversation_id  TEXT REFERENCES conversations(conversation_id) ON DELETE SET NULL,
    filename         TEXT NOT NULL,
    safe_filename    TEXT NOT NULL,
    file_path        TEXT NOT NULL,
    file_type        TEXT NOT NULL,
    file_size_bytes  INTEGER NOT NULL DEFAULT 0,
    checksum         TEXT,
    document_type    TEXT DEFAULT 'unknown',
    page_count       INTEGER DEFAULT 0,
    processing_status TEXT NOT NULL DEFAULT 'pending'
        CHECK(processing_status IN ('pending','processing','completed','failed')),
    upload_timestamp TEXT NOT NULL,
    processed_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_conversation
    ON documents(conversation_id);

-- -----------------------------------------------------------------------
-- Extracted facts from uploaded documents
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_facts (
    fact_id          TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    fact_type        TEXT NOT NULL,
    extracted_value  TEXT NOT NULL,
    page_number      INTEGER,
    verification_status TEXT DEFAULT 'unverified'
);

-- -----------------------------------------------------------------------
-- Legal sources (official documents in the knowledge base)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS legal_sources (
    source_id          TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    document_type      TEXT NOT NULL DEFAULT 'unknown',
    legal_domain       TEXT DEFAULT 'unknown',
    jurisdiction       TEXT NOT NULL DEFAULT 'Karnataka',
    state              TEXT DEFAULT 'Karnataka',
    act_name           TEXT,
    section_number     TEXT,
    court_name         TEXT,
    case_number        TEXT,
    department         TEXT,
    source_url         TEXT NOT NULL DEFAULT '',
    publication_date   TEXT,
    commencement_date  TEXT,
    effective_from     TEXT,
    effective_until    TEXT,
    version            TEXT DEFAULT '1.0',
    amendment_status   TEXT,
    language           TEXT DEFAULT 'en',
    file_path          TEXT,
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    checksum           TEXT,
    local_path         TEXT,
    full_text          TEXT,
    retrieved_at       TEXT,
    created_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_legal_sources_domain
    ON legal_sources(legal_domain);
CREATE INDEX IF NOT EXISTS idx_legal_sources_jurisdiction
    ON legal_sources(jurisdiction);

-- -----------------------------------------------------------------------
-- Legal chunks (text chunks in the knowledge base, links to FAISS)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS legal_chunks (
    chunk_id           TEXT PRIMARY KEY,
    source_id          TEXT REFERENCES legal_sources(source_id) ON DELETE CASCADE,
    chunk_text         TEXT NOT NULL,
    section_reference  TEXT,
    chapter_reference  TEXT,
    page_number        INTEGER,
    chunk_index        INTEGER NOT NULL DEFAULT 0,
    total_chunks       INTEGER NOT NULL DEFAULT 1,
    embedding_reference TEXT   -- e.g. faiss vector index position (stored as string)
);

-- Full-text search virtual table for keyword retrieval (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS legal_chunks_fts
    USING fts5(
        chunk_id UNINDEXED,
        source_id UNINDEXED,
        chunk_text,
        section_reference,
        content='legal_chunks',
        content_rowid='rowid'
    );

-- -----------------------------------------------------------------------
-- Citations (links from messages to source chunks)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS citations (
    citation_id      TEXT PRIMARY KEY,
    message_id       TEXT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    source_id        TEXT REFERENCES legal_sources(source_id),
    chunk_id         TEXT REFERENCES legal_chunks(chunk_id),
    page_number      INTEGER,
    section_reference TEXT,
    retrieval_score  REAL,
    is_verified      INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_citations_message
    ON citations(message_id);

-- -----------------------------------------------------------------------
-- Scraping jobs
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS scraping_jobs (
    job_id               TEXT PRIMARY KEY,
    source_name          TEXT NOT NULL DEFAULT 'manual',
    source_url           TEXT,
    started_at           TEXT NOT NULL,
    completed_at         TEXT,
    finished_at          TEXT,
    status               TEXT NOT NULL DEFAULT 'running'
        CHECK(status IN ('running','completed','failed','cancelled')),
    documents_found      INTEGER DEFAULT 0,
    documents_downloaded INTEGER DEFAULT 0,
    documents_ingested   INTEGER DEFAULT 0,
    documents_skipped    INTEGER DEFAULT 0,
    result_summary       TEXT,
    error_details        TEXT
);

-- -----------------------------------------------------------------------
-- Reports
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    report_id        TEXT PRIMARY KEY,
    conversation_id  TEXT REFERENCES conversations(conversation_id) ON DELETE SET NULL,
    title            TEXT NOT NULL,
    file_path        TEXT,
    created_at       TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def get_db_path() -> Path:
    db_path = Path(settings.resolved_db_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """
    Apply additive schema migrations using ALTER TABLE … ADD COLUMN IF NOT EXISTS
    (emulated for SQLite which doesn't support IF NOT EXISTS on ADD COLUMN).

    Safe to call multiple times — columns are only added if missing.
    """
    migrations = [
        # legal_sources — columns added in Batch 6 for scraper persistence
        ("legal_sources", "section_number", "TEXT"),
        ("legal_sources", "case_number",    "TEXT"),
        ("legal_sources", "checksum",       "TEXT"),
        ("legal_sources", "local_path",     "TEXT"),
        ("legal_sources", "full_text",      "TEXT"),
        # scraping_jobs — columns added for ScraperManager
        ("scraping_jobs", "finished_at",       "TEXT"),
        ("scraping_jobs", "documents_ingested", "INTEGER DEFAULT 0"),
        ("scraping_jobs", "result_summary",     "TEXT"),
        ("scraping_jobs", "source_name",        "TEXT NOT NULL DEFAULT 'manual'"),
    ]

    for table, column, col_type in migrations:
        # Check if column already exists.
        try:
            existing = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                logger.info(f"Migration: added column {table}.{column}")
        except Exception as exc:
            logger.warning(f"Migration skipped for {table}.{column}: {exc}")


def init_db() -> None:
    """
    Create all tables if they do not exist.
    Call once at application startup.
    """
    db_path = get_db_path()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            _apply_migrations(conn)
            conn.commit()
        logger.info(f"SQLite database initialized at {db_path}")
    except Exception as exc:
        logger.error(f"Failed to initialize SQLite database: {exc}")
        raise


@contextmanager
def get_connection():
    """
    Context manager that yields a configured SQLite connection.
    Automatically commits on clean exit, rolls back on exception.

    Usage:
        with get_connection() as conn:
            conn.execute(...)
    """
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
