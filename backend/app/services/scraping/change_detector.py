"""
Bureaucracy AI — Change Detector

Detects whether a newly scraped document differs from the version already
stored in the database, to avoid re-ingesting unchanged documents.

Logic:
1. Compare SHA-256 checksum of the new document against stored checksum.
2. If checksums differ (or no stored record), mark as NEW or UPDATED.
3. If checksums match, mark as UNCHANGED.
4. Never silently replace stored versions — always create a new version record.
5. Historical versions are preserved; only the latest is flagged as current.

IMPORTANT:
- A newly published amendment does NOT automatically overwrite existing law.
- The ``amendment_status`` field must be set by the scraper.
- Human validation is required before marking a document as verified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.app.services.scraping.base_scraper import ScrapedDocument
from backend.app.services.database.connection import get_connection

logger = logging.getLogger(__name__)


class ChangeStatus(str, Enum):
    NEW = "new"             # No prior version exists
    UPDATED = "updated"     # Content has changed (different checksum)
    UNCHANGED = "unchanged" # Checksum matches stored version — skip


@dataclass
class ChangeResult:
    document: ScrapedDocument
    status: ChangeStatus
    prior_version: Optional[int] = None   # DB row ID of prior version, if any
    prior_checksum: Optional[str] = None


class ChangeDetector:
    """
    Compares scraped documents against the stored legal_sources table
    and classifies each as NEW, UPDATED, or UNCHANGED.

    Usage::

        detector = ChangeDetector()
        results = detector.classify(scraped_documents)
        to_ingest = [r for r in results if r.status != ChangeStatus.UNCHANGED]
    """

    def classify(self, documents: list[ScrapedDocument]) -> list[ChangeResult]:
        """
        Classify each document as NEW, UPDATED, or UNCHANGED.

        Parameters
        ----------
        documents:
            List of freshly scraped documents.

        Returns
        -------
        List of :class:`ChangeResult` in the same order as *documents*.
        """
        results: list[ChangeResult] = []
        for doc in documents:
            result = self._classify_one(doc)
            results.append(result)
            logger.debug(
                f"[{result.status.value.upper()}] {doc.title[:60]} "
                f"(checksum={doc.checksum[:8] if doc.checksum else 'none'}...)"
            )
        return results

    def _classify_one(self, doc: ScrapedDocument) -> ChangeResult:
        """Classify a single document against the database."""
        try:
            with get_connection() as conn:
                row = conn.execute(
                    """
                    SELECT id, checksum, version
                    FROM legal_sources
                    WHERE source_url = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (doc.source_url,),
                ).fetchone()
        except Exception as exc:
            logger.warning(f"DB lookup failed for '{doc.title}': {exc} — treating as NEW")
            return ChangeResult(document=doc, status=ChangeStatus.NEW)

        if row is None:
            return ChangeResult(document=doc, status=ChangeStatus.NEW)

        stored_id, stored_checksum, stored_version = row["id"], row["checksum"], row["version"]

        if doc.checksum and stored_checksum and doc.checksum == stored_checksum:
            return ChangeResult(
                document=doc,
                status=ChangeStatus.UNCHANGED,
                prior_version=stored_id,
                prior_checksum=stored_checksum,
            )

        # Bump version number.
        try:
            major, minor = str(stored_version or "1.0").split(".", 1)
            new_version = f"{major}.{int(minor) + 1}"
        except (ValueError, AttributeError):
            new_version = "1.1"

        doc.version = new_version
        return ChangeResult(
            document=doc,
            status=ChangeStatus.UPDATED,
            prior_version=stored_id,
            prior_checksum=stored_checksum,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def filter_new_or_updated(results: list[ChangeResult]) -> list[ScrapedDocument]:
        """Return only the documents that need (re-)ingestion."""
        return [r.document for r in results if r.status != ChangeStatus.UNCHANGED]

    @staticmethod
    def summary(results: list[ChangeResult]) -> dict:
        """Return a counts summary dict."""
        counts: dict[str, int] = {s.value: 0 for s in ChangeStatus}
        for r in results:
            counts[r.status.value] += 1
        return counts
