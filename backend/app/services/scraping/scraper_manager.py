"""
Bureaucracy AI — Scraper Manager

Coordinates all scrapers, runs change detection, and passes new/updated
documents to the ingestion pipeline.

Responsibilities:
1. Instantiate and run each configured scraper.
2. Pass results through the ChangeDetector.
3. Persist new/updated documents to the database (legal_sources table).
4. Queue documents for chunking + embedding (calls ingest_knowledge logic).
5. Update scraping_jobs table with run results.
6. Provide a simple public API for the admin endpoint to trigger scraping.

Design constraints:
- Never silently replace verified documents.
- Preserve all historical versions.
- Fail gracefully: one scraper failing must not stop others.
- All scheduling via APScheduler (optional) or direct call.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.app.services.scraping.base_scraper import ScrapedDocument
from backend.app.services.scraping.change_detector import ChangeDetector, ChangeStatus
from backend.app.services.database.connection import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scraper registry — add new scrapers here
# ---------------------------------------------------------------------------

def _build_scrapers() -> list:
    """
    Instantiate all configured scrapers. Returns only those that are available
    (import errors are caught gracefully).
    """
    scrapers = []

    try:
        from backend.app.services.scraping.india_code_scraper import IndiaCodeScraper
        scrapers.append(IndiaCodeScraper(output_dir=Path("data/raw/india_code")))
    except Exception as exc:
        logger.warning(f"IndiaCodeScraper unavailable: {exc}")

    try:
        from backend.app.services.scraping.gazette_scraper import GazetteScraper
        scrapers.append(GazetteScraper(output_dir=Path("data/raw/gazette")))
    except Exception as exc:
        logger.warning(f"GazetteScraper unavailable: {exc}")

    try:
        from backend.app.services.scraping.karnataka_court_scraper import KarnatakaCourtScraper
        scrapers.append(KarnatakaCourtScraper(output_dir=Path("data/raw/karnataka_court")))
    except Exception as exc:
        logger.warning(f"KarnatakaCourtScraper unavailable: {exc}")

    return scrapers


# ---------------------------------------------------------------------------
# Scraping result
# ---------------------------------------------------------------------------

class ScrapingRunResult:
    """Summary of one complete scraping run."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.started_at = datetime.utcnow().isoformat()
        self.finished_at: Optional[str] = None
        self.scrapers_run: list[str] = []
        self.total_scraped: int = 0
        self.new_documents: int = 0
        self.updated_documents: int = 0
        self.unchanged_documents: int = 0
        self.ingested: int = 0
        self.errors: list[str] = []

    def finish(self):
        self.finished_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scrapers_run": self.scrapers_run,
            "total_scraped": self.total_scraped,
            "new_documents": self.new_documents,
            "updated_documents": self.updated_documents,
            "unchanged_documents": self.unchanged_documents,
            "ingested": self.ingested,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Scraper Manager
# ---------------------------------------------------------------------------

class ScraperManager:
    """
    Runs all scrapers, detects changes, and ingests new/updated documents.

    Usage::

        manager = ScraperManager()
        result = manager.run_all()
        print(result.to_dict())

    Or to run a single scraper by name::

        result = manager.run_one("IndiaCodeScraper")
    """

    def __init__(self):
        self._detector = ChangeDetector()

    def run_all(self) -> ScrapingRunResult:
        """Run every configured scraper in sequence."""
        job_id = str(uuid.uuid4())[:8]
        result = ScrapingRunResult(job_id=job_id)

        self._persist_job_start(job_id)
        logger.info(f"ScraperManager: starting run {job_id}")

        scrapers = _build_scrapers()
        if not scrapers:
            msg = "No scrapers available — check that required packages are installed."
            logger.warning(msg)
            result.errors.append(msg)
            result.finish()
            return result

        for scraper in scrapers:
            scraper_name = scraper.__class__.__name__
            result.scrapers_run.append(scraper_name)
            logger.info(f"Running {scraper_name}...")
            try:
                docs = scraper.scrape()
                result.total_scraped += len(docs)
                self._process_documents(docs, result)
            except Exception as exc:
                msg = f"{scraper_name} failed: {exc}"
                logger.error(msg)
                result.errors.append(msg)

        result.finish()
        self._persist_job_finish(job_id, result)
        logger.info(
            f"ScraperManager: run {job_id} complete — "
            f"{result.new_documents} new, {result.updated_documents} updated, "
            f"{result.unchanged_documents} unchanged, {result.ingested} ingested"
        )
        return result

    def run_one(self, scraper_name: str) -> ScrapingRunResult:
        """Run a single scraper identified by its class name."""
        job_id = str(uuid.uuid4())[:8]
        result = ScrapingRunResult(job_id=job_id)

        scrapers = _build_scrapers()
        target = next((s for s in scrapers if s.__class__.__name__ == scraper_name), None)

        if target is None:
            msg = f"Scraper '{scraper_name}' not found. Available: {[s.__class__.__name__ for s in scrapers]}"
            logger.error(msg)
            result.errors.append(msg)
            result.finish()
            return result

        result.scrapers_run.append(scraper_name)
        try:
            docs = target.scrape()
            result.total_scraped += len(docs)
            self._process_documents(docs, result)
        except Exception as exc:
            result.errors.append(f"{scraper_name} failed: {exc}")
            logger.error(result.errors[-1])

        result.finish()
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_documents(
        self, docs: list[ScrapedDocument], result: ScrapingRunResult
    ) -> None:
        """Run change detection and ingest new/updated documents."""
        if not docs:
            return

        change_results = self._detector.classify(docs)
        summary = self._detector.summary(change_results)
        result.new_documents += summary.get("new", 0)
        result.updated_documents += summary.get("updated", 0)
        result.unchanged_documents += summary.get("unchanged", 0)

        to_ingest = ChangeDetector.filter_new_or_updated(change_results)
        for doc in to_ingest:
            try:
                self._persist_document(doc)
                result.ingested += 1
            except Exception as exc:
                msg = f"Failed to persist '{doc.title[:60]}': {exc}"
                logger.error(msg)
                result.errors.append(msg)

        # Trigger FAISS re-index if any new content was ingested.
        if to_ingest:
            self._trigger_reindex()

    def _persist_document(self, doc: ScrapedDocument) -> None:
        """
        Write the scraped document to the ``legal_sources`` table.
        Text chunking and embedding are deferred to the ingestion pipeline.
        """
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO legal_sources (
                    title, document_type, jurisdiction, state,
                    act_name, section_number, court_name, case_number,
                    source_url, publication_date, commencement_date,
                    effective_from, effective_until, version, amendment_status,
                    language, verification_status, checksum, local_path, full_text
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    doc.title,
                    doc.document_type,
                    doc.jurisdiction,
                    "karnataka" if doc.jurisdiction == "karnataka" else None,
                    doc.act_name,
                    doc.section_number,
                    doc.court_name,
                    doc.case_number,
                    doc.source_url,
                    doc.publication_date,
                    doc.commencement_date,
                    doc.effective_from,
                    doc.effective_until,
                    doc.version,
                    doc.amendment_status,
                    doc.language,
                    "unverified",           # requires human review
                    doc.checksum,
                    str(doc.local_path) if doc.local_path else None,
                    doc.text_content,
                ),
            )
            conn.commit()
        logger.debug(f"Persisted legal source: {doc.title[:60]}")

    @staticmethod
    def _trigger_reindex() -> None:
        """
        Signal the RAG service to reload the FAISS index after new documents
        have been persisted. Calls ``RAGService.reload_index()`` if available.
        """
        try:
            from backend.app.services.rag.rag_service import RAGService
            rag = RAGService.get_instance()
            rag.reload_index()
            logger.info("FAISS index reloaded after new document ingestion.")
        except Exception as exc:
            logger.warning(f"Could not reload FAISS index: {exc}")

    def _persist_job_start(self, job_id: str) -> None:
        try:
            with get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO scraping_jobs (job_id, status, started_at)
                    VALUES (?, 'running', ?)
                    """,
                    (job_id, datetime.utcnow().isoformat()),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"Could not persist job start: {exc}")

    def _persist_job_finish(self, job_id: str, result: ScrapingRunResult) -> None:
        try:
            import json
            with get_connection() as conn:
                conn.execute(
                    """
                    UPDATE scraping_jobs
                    SET status = 'completed',
                        finished_at = ?,
                        documents_found = ?,
                        documents_ingested = ?,
                        result_summary = ?
                    WHERE job_id = ?
                    """,
                    (
                        result.finished_at,
                        result.total_scraped,
                        result.ingested,
                        json.dumps(result.to_dict()),
                        job_id,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.debug(f"Could not persist job finish: {exc}")


# ---------------------------------------------------------------------------
# Convenience function for the admin API endpoint
# ---------------------------------------------------------------------------

def trigger_scraping_run(scraper_name: Optional[str] = None) -> dict:
    """
    Public entry point called by the admin API.

    Parameters
    ----------
    scraper_name:
        If given, run only that scraper. Otherwise run all scrapers.

    Returns
    -------
    A summary dict suitable for JSON serialization.
    """
    manager = ScraperManager()
    if scraper_name:
        result = manager.run_one(scraper_name)
    else:
        result = manager.run_all()
    return result.to_dict()


# ---------------------------------------------------------------------------
# Optional APScheduler integration
# ---------------------------------------------------------------------------

def setup_scheduled_scraping(scheduler, interval_hours: int = 24) -> None:
    """
    Register a periodic scraping job with an existing APScheduler instance.

    Parameters
    ----------
    scheduler:
        An ``apscheduler.schedulers.background.BackgroundScheduler`` instance.
    interval_hours:
        How often to run all scrapers (default: once per day).
    """
    try:
        scheduler.add_job(
            trigger_scraping_run,
            trigger="interval",
            hours=interval_hours,
            id="scraping_run",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        logger.info(f"Scheduled scraping every {interval_hours} hours.")
    except Exception as exc:
        logger.error(f"Could not schedule scraping: {exc}")
