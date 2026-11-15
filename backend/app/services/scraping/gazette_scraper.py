"""
Bureaucracy AI — e-Gazette Scraper

Harvests official government notifications and gazette publications
from egazette.gov.in (official e-Gazette of India).

e-Gazette provides:
- Searchable gazette notifications
- Downloadable PDFs for each notification
- Metadata: ministry, date, gazette type, notification number

Strategy:
1. Query the search interface for Karnataka-relevant notifications.
2. Download each PDF.
3. Extract text with PyMuPDF.
4. Return ScrapedDocument list for ingestion.

IMPORTANT:
- Only publicly available documents are accessed.
- robots.txt is respected.
- No login, CAPTCHA bypass, or rate-limit circumvention.
- Documents are marked unverified until human review.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from backend.app.services.scraping.base_scraper import BaseScraper, ScrapedDocument, ScraperError

logger = logging.getLogger(__name__)

GAZETTE_BASE = "https://egazette.gov.in"


class GazetteScraper(BaseScraper):
    """
    Scrapes official gazette notifications from egazette.gov.in.

    Targets Karnataka-related central notifications and labour/consumer
    ministry announcements.

    Usage::

        scraper = GazetteScraper(output_dir=Path("data/raw/gazette"))
        documents = scraper.scrape()
    """

    DEFAULT_DELAY_SECONDS = 3.0

    # Search queries and their metadata
    SEARCH_TARGETS = [
        {
            "query": "Karnataka minimum wages notification",
            "document_type": "government_notification",
            "jurisdiction": "karnataka",
            "act_name": "Minimum Wages Act, 1948",
        },
        {
            "query": "consumer protection Karnataka notification",
            "document_type": "government_notification",
            "jurisdiction": "central",
            "act_name": "Consumer Protection Act, 2019",
        },
        {
            "query": "right to information Karnataka",
            "document_type": "government_notification",
            "jurisdiction": "central",
            "act_name": "Right to Information Act, 2005",
        },
        {
            "query": "labour welfare Karnataka",
            "document_type": "government_notification",
            "jurisdiction": "karnataka",
            "act_name": None,
        },
    ]

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            output_dir = Path("data/raw/gazette")
        super().__init__(output_dir=output_dir)

    def scrape(self) -> list[ScrapedDocument]:
        """
        Attempt to fetch gazette notifications for each search target.
        Returns all successfully harvested documents.
        """
        logger.info(f"GazetteScraper: starting — {len(self.SEARCH_TARGETS)} search targets")
        results: list[ScrapedDocument] = []

        # First, verify the base URL is accessible.
        probe = self._get(GAZETTE_BASE)
        if probe is None:
            logger.warning(
                "GazetteScraper: egazette.gov.in is unreachable or scraping disallowed. "
                "Consider downloading bulk datasets from the official portal manually."
            )
            return results

        for target in self.SEARCH_TARGETS:
            try:
                docs = self._scrape_target(target)
                results.extend(docs)
                logger.info(f"  ✓ '{target['query']}' — {len(docs)} document(s)")
            except ScraperError as exc:
                logger.error(f"  ✗ '{target['query']}' — {exc}")
            except Exception as exc:
                logger.error(f"  ✗ '{target['query']}' — unexpected: {exc}")

        logger.info(f"GazetteScraper: complete — {len(results)} total documents harvested")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scrape_target(self, target: dict) -> list[ScrapedDocument]:
        """Search the gazette and return documents for one search target."""
        # Build a search URL for the e-Gazette search interface.
        encoded_query = target["query"].replace(" ", "+")
        search_url = f"{GAZETTE_BASE}/WriteReadData/search?q={encoded_query}"

        resp = self._get(search_url)
        if resp is None:
            return []

        # Extract PDF links from the results page.
        pdf_links = self._extract_pdf_links(resp.text, search_url)
        if not pdf_links:
            logger.debug(f"No PDF links found for query: {target['query']}")
            return []

        documents: list[ScrapedDocument] = []
        # Limit to first 3 results per query to be conservative.
        for pdf_url, link_text in pdf_links[:3]:
            doc = self._download_gazette_pdf(
                pdf_url=pdf_url,
                link_text=link_text,
                target=target,
                search_url=search_url,
            )
            if doc is not None:
                documents.append(doc)

        return documents

    def _extract_pdf_links(self, html: str, base_url: str) -> list[tuple[str, str]]:
        """
        Extract (url, link_text) pairs for PDF links from the HTML.
        Returns empty list if beautifulsoup4 is not installed.
        """
        pairs: list[tuple[str, str]] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf") or "WriteReadData" in href:
                    full_url = urljoin(base_url, href)
                    pairs.append((full_url, a.get_text(strip=True)))
        except ImportError:
            # Fallback: regex extraction
            pattern = r'href="([^"]*\.pdf[^"]*)"[^>]*>([^<]*)'
            for m in re.finditer(pattern, html, re.IGNORECASE):
                pairs.append((urljoin(base_url, m.group(1)), m.group(2).strip()))
        return pairs

    def _download_gazette_pdf(
        self,
        pdf_url: str,
        link_text: str,
        target: dict,
        search_url: str,
    ) -> Optional[ScrapedDocument]:
        """Download one gazette PDF and return a ScrapedDocument."""
        resp = self._get(pdf_url, stream=True)
        if resp is None:
            return None

        raw = resp.content
        checksum = self._checksum(raw)

        # Build a safe filename from the URL tail.
        url_tail = pdf_url.split("/")[-1]
        safe_name = re.sub(r"[^\w\-\.]", "_", url_tail)
        if not safe_name.endswith(".pdf"):
            safe_name += ".pdf"

        local_path = self._save_raw(safe_name, raw)
        text_content = self._extract_pdf_text(local_path)

        if not text_content:
            logger.debug(f"Empty text from gazette PDF: {pdf_url}")
            return None

        # Try to extract a date from the link text or filename.
        pub_date = self._extract_date(link_text) or self._extract_date(url_tail)

        title = link_text if link_text else f"Gazette Notification — {url_tail}"

        return ScrapedDocument(
            title=title,
            source_url=pdf_url,
            document_type=target["document_type"],
            jurisdiction=target["jurisdiction"],
            act_name=target.get("act_name"),
            language="english",
            publication_date=pub_date,
            amendment_status="original",
            local_path=local_path,
            text_content=text_content,
            checksum=checksum,
            metadata={
                "search_url": search_url,
                "search_query": target["query"],
            },
        )

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        """
        Try to extract a date in YYYY-MM-DD format from arbitrary text.
        Returns None if no date is found.
        """
        # Patterns: 2024-05-01 / 01-05-2024 / 01/05/2024
        iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
        if iso:
            return iso.group(0)
        dmy = re.search(r"(?<!\d)(\d{1,2})[-/](\d{1,2})[-/](\d{4})(?!\d)", text)
        if dmy:
            day, month, year = dmy.group(1), dmy.group(2), dmy.group(3)
            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        return None
