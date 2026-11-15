"""
Bureaucracy AI — Base Scraper

Abstract base class for all legal-document scrapers.
Provides shared utilities: HTTP session, rate limiting, robots.txt checking,
checksum deduplication, and structured document passing to the ingestion queue.
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScrapedDocument:
    """
    Represents one legal document harvested from an external source.

    Fields map directly to the ``legal_sources`` database table so the
    ingestion pipeline can persist them without further transformation.
    """
    title: str
    source_url: str
    document_type: str                     # act / gazette / judgment / order …
    jurisdiction: str                      # central / karnataka
    language: str = "english"
    publication_date: Optional[str] = None  # ISO-8601 date string or None
    commencement_date: Optional[str] = None
    effective_from: Optional[str] = None
    effective_until: Optional[str] = None
    act_name: Optional[str] = None
    section_number: Optional[str] = None
    court_name: Optional[str] = None
    case_number: Optional[str] = None
    amendment_status: str = "unknown"      # original / amended / repealed / unknown
    version: str = "1.0"
    local_path: Optional[Path] = None      # path to saved raw file
    text_content: Optional[str] = None     # extracted plain text (may be large)
    checksum: Optional[str] = None         # SHA-256 of raw bytes
    scraped_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base scraper
# ---------------------------------------------------------------------------

class BaseScraper(ABC):
    """
    Abstract base class every scraper must inherit from.

    Subclasses implement ``scrape()`` and optionally override
    ``_get_scraper_name()`` for logging.

    Shared behaviour provided here:
    - Configured ``requests.Session`` with retries and a legal User-Agent.
    - ``_get()`` wrapper that respects rate limiting and robots.txt.
    - ``_save_raw()`` to persist downloaded bytes to disk.
    - ``_checksum()`` for SHA-256 deduplication.
    - ``_check_robots()`` to verify access permission.
    """

    # Subclasses can override these defaults.
    DEFAULT_DELAY_SECONDS: float = 2.0        # seconds between requests
    DEFAULT_TIMEOUT_SECONDS: int = 30
    MAX_RETRIES: int = 3

    def __init__(
        self,
        output_dir: Path,
        delay_seconds: float | None = None,
        respect_robots: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.delay = delay_seconds if delay_seconds is not None else self.DEFAULT_DELAY_SECONDS
        self.respect_robots = respect_robots
        self._last_request_time: float = 0.0
        self._robots_cache: dict[str, RobotFileParser] = {}

        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def scrape(self) -> list[ScrapedDocument]:
        """
        Execute the full scraping run for this source.

        Returns a list of ``ScrapedDocument`` objects ready for ingestion.
        May raise ``ScraperError`` on unrecoverable failures.
        """

    def _get_scraper_name(self) -> str:
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _build_session(self):
        """Create a requests Session with retry logic and a respectful UA."""
        try:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            session = requests.Session()
            retry = Retry(
                total=self.MAX_RETRIES,
                backoff_factor=1.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "HEAD"],
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            session.headers.update({
                "User-Agent": (
                    "BureaucracyAI-ResearchBot/1.0 "
                    "(Academic legal-information project; "
                    "contact: research@bureaucracy.ai)"
                ),
                "Accept": "text/html,application/pdf,application/json",
            })
            return session
        except ImportError:
            logger.warning("requests not installed — HTTP scraping unavailable")
            return None

    def _rate_limit(self) -> None:
        """Block until the configured inter-request delay has elapsed."""
        elapsed = time.monotonic() - self._last_request_time
        remaining = self.delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_time = time.monotonic()

    def _check_robots(self, url: str) -> bool:
        """
        Return ``True`` if this scraper's User-Agent is allowed to fetch *url*.
        Always returns ``True`` when ``respect_robots`` is ``False``.
        Caches robots.txt per domain.
        """
        if not self.respect_robots:
            return True

        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        if domain not in self._robots_cache:
            rp = RobotFileParser()
            robots_url = f"{domain}/robots.txt"
            try:
                rp.set_url(robots_url)
                rp.read()
                self._robots_cache[domain] = rp
                logger.debug(f"Loaded robots.txt from {robots_url}")
            except Exception as exc:
                logger.warning(f"Could not read robots.txt for {domain}: {exc} — allowing")
                # If we cannot read robots.txt we default to allowing.
                return True

        rp = self._robots_cache[domain]
        agent = "BureaucracyAI-ResearchBot"
        allowed = rp.can_fetch(agent, url)
        if not allowed:
            logger.info(f"robots.txt disallows: {url}")
        return allowed

    def _get(self, url: str, stream: bool = False, **kwargs):
        """
        Rate-limited HTTP GET with robots.txt checking.

        Returns the ``requests.Response`` or ``None`` on failure.
        """
        if self._session is None:
            logger.error("No HTTP session available — requests package missing")
            return None

        if not self._check_robots(url):
            logger.info(f"Skipping (disallowed by robots.txt): {url}")
            return None

        self._rate_limit()
        try:
            resp = self._session.get(
                url,
                timeout=self.DEFAULT_TIMEOUT_SECONDS,
                stream=stream,
                **kwargs,
            )
            resp.raise_for_status()
            logger.debug(f"GET {url} → {resp.status_code}")
            return resp
        except Exception as exc:
            logger.warning(f"GET {url} failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _save_raw(self, filename: str, content: bytes) -> Path:
        """Write *content* bytes to ``output_dir / filename``. Returns the path."""
        path = self.output_dir / filename
        path.write_bytes(content)
        logger.debug(f"Saved {len(content):,} bytes → {path}")
        return path

    @staticmethod
    def _checksum(data: bytes) -> str:
        """Return the SHA-256 hex digest of *data*."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _extract_pdf_text(path: Path) -> str:
        """Extract plain text from a PDF using PyMuPDF. Returns empty string on failure."""
        try:
            import pymupdf  # fitz
            text_parts: list[str] = []
            with pymupdf.open(path) as doc:
                for page in doc:
                    text_parts.append(page.get_text("text").strip())
            return "\n\n".join(filter(None, text_parts))
        except Exception as exc:
            logger.warning(f"PDF text extraction failed for {path}: {exc}")
            return ""


class ScraperError(Exception):
    """Raised when a scraper encounters an unrecoverable error."""
