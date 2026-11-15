"""
Bureaucracy AI — India Code Scraper

Harvests Central Acts and their sections from India Code (indiacode.nic.in).

India Code provides:
- A bulk download page at https://www.indiacode.nic.in/handle/123456789/1362
- Individual act pages with metadata and section text
- Downloadable PDFs for individual acts

Strategy (order of preference):
1. Use the publicly available act-list API endpoint (if present).
2. Parse the acts listing page and download individual act PDFs.
3. Respect robots.txt and rate limits throughout.

IMPORTANT:
- This scraper collects act text as published.
- It does NOT interpret legal meaning.
- Verification status is set to "unverified" until manually reviewed.
- Amendments are not automatically applied.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from backend.app.services.scraping.base_scraper import BaseScraper, ScrapedDocument, ScraperError

logger = logging.getLogger(__name__)

INDIA_CODE_BASE = "https://www.indiacode.nic.in"


class IndiaCodeScraper(BaseScraper):
    """
    Scrapes central legislation from India Code.

    Usage::

        scraper = IndiaCodeScraper(output_dir=Path("data/raw/india_code"))
        documents = scraper.scrape()

    ``documents`` is a list of :class:`~backend.app.services.scraping.base_scraper.ScrapedDocument`
    objects ready for the ingestion pipeline.
    """

    DEFAULT_DELAY_SECONDS = 3.0  # be conservative with the NIC server

    # Acts to target in the initial run. Extend this list to cover more acts.
    TARGET_ACTS = [
        {
            "title": "Payment of Wages Act, 1936",
            "search_term": "Payment+of+Wages+Act",
            "act_name": "Payment of Wages Act, 1936",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
        {
            "title": "Minimum Wages Act, 1948",
            "search_term": "Minimum+Wages+Act",
            "act_name": "Minimum Wages Act, 1948",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
        {
            "title": "Consumer Protection Act, 2019",
            "search_term": "Consumer+Protection+Act",
            "act_name": "Consumer Protection Act, 2019",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
        {
            "title": "Right to Information Act, 2005",
            "search_term": "Right+to+Information+Act",
            "act_name": "Right to Information Act, 2005",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
        {
            "title": "Transfer of Property Act, 1882",
            "search_term": "Transfer+of+Property+Act",
            "act_name": "Transfer of Property Act, 1882",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
        {
            "title": "Code of Civil Procedure, 1908",
            "search_term": "Code+of+Civil+Procedure",
            "act_name": "Code of Civil Procedure, 1908",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
        {
            "title": "Indian Penal Code, 1860",
            "search_term": "Indian+Penal+Code",
            "act_name": "Indian Penal Code, 1860",
            "document_type": "central_act",
            "jurisdiction": "central",
        },
    ]

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            output_dir = Path("data/raw/india_code")
        super().__init__(output_dir=output_dir)

    def scrape(self) -> list[ScrapedDocument]:
        """
        Attempt to fetch act metadata and text for each configured target act.

        Returns all successfully harvested documents.
        """
        logger.info(f"IndiaCodeScraper: starting — {len(self.TARGET_ACTS)} target acts")
        results: list[ScrapedDocument] = []

        for act_config in self.TARGET_ACTS:
            try:
                doc = self._scrape_act(act_config)
                if doc is not None:
                    results.append(doc)
                    logger.info(f"  ✓ {act_config['title']}")
                else:
                    logger.warning(f"  ✗ {act_config['title']} — not found or inaccessible")
            except ScraperError as exc:
                logger.error(f"  ✗ {act_config['title']} — {exc}")
            except Exception as exc:
                logger.error(f"  ✗ {act_config['title']} — unexpected: {exc}")

        logger.info(f"IndiaCodeScraper: complete — {len(results)}/{len(self.TARGET_ACTS)} harvested")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scrape_act(self, act_config: dict) -> Optional[ScrapedDocument]:
        """Search India Code for one act and return a ScrapedDocument."""
        search_url = (
            f"{INDIA_CODE_BASE}/search?q={act_config['search_term']}"
            f"&searchtype=acts&sorttype=&searchphrase=allwords"
        )
        resp = self._get(search_url)
        if resp is None:
            return None

        # Find the first act link in the search results.
        act_url = self._find_act_url(resp.text, act_config["title"])
        if act_url is None:
            logger.debug(f"No act page found for: {act_config['title']}")
            return None

        # Attempt to find a PDF download link on the act page.
        act_resp = self._get(act_url)
        if act_resp is None:
            return None

        pdf_url = self._find_pdf_link(act_resp.text, act_url)
        text_content: str = ""
        local_path: Optional[Path] = None
        checksum: Optional[str] = None

        if pdf_url:
            pdf_resp = self._get(pdf_url, stream=True)
            if pdf_resp is not None:
                raw = pdf_resp.content
                checksum = self._checksum(raw)
                safe_name = re.sub(r"[^\w\-]", "_", act_config["act_name"]) + ".pdf"
                local_path = self._save_raw(safe_name, raw)
                text_content = self._extract_pdf_text(local_path)

        if not text_content:
            # Fall back to parsing HTML text from the act page.
            text_content = self._extract_html_text(act_resp.text)

        if not text_content:
            logger.debug(f"No usable text for: {act_config['title']}")
            return None

        return ScrapedDocument(
            title=act_config["title"],
            source_url=act_url,
            document_type=act_config["document_type"],
            jurisdiction=act_config["jurisdiction"],
            act_name=act_config["act_name"],
            language="english",
            amendment_status="original",
            local_path=local_path,
            text_content=text_content,
            checksum=checksum,
            metadata={"search_url": search_url, "pdf_url": pdf_url},
        )

    def _find_act_url(self, html: str, act_title: str) -> Optional[str]:
        """Return the first act-page URL from a search-result page."""
        # India Code act pages are under /handle/
        pattern = r'href="(/handle/[^"]+)"'
        matches = re.findall(pattern, html)
        if not matches:
            return None
        # Return the first match (most relevant result).
        return urljoin(INDIA_CODE_BASE, matches[0])

    def _find_pdf_link(self, html: str, base_url: str) -> Optional[str]:
        """Return the PDF download URL from an act page, or None."""
        # India Code PDF links often contain /bitstream/ or end with .pdf
        pattern = r'href="([^"]*(?:bitstream|\.pdf)[^"]*)"'
        matches = re.findall(pattern, html, re.IGNORECASE)
        for m in matches:
            if m.endswith(".pdf") or "bitstream" in m:
                return urljoin(base_url, m)
        return None

    def _extract_html_text(self, html: str) -> str:
        """Strip HTML tags and return plain text. Requires beautifulsoup4."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            # Crude fallback: strip tags with regex
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()
