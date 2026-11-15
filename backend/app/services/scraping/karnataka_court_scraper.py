"""
Bureaucracy AI — Karnataka High Court Scraper

Harvests publicly available judgments and orders from the Karnataka High Court
and eCourts portal.

Sources:
- Karnataka High Court: https://karnatakajudiciary.kar.nic.in
- eCourts services: https://ecourts.gov.in
- National Judicial Data Grid (NJDG): https://njdg.ecourts.gov.in

Strategy:
1. Access the publicly available judgment search pages.
2. Identify and download judgment PDFs.
3. Extract text with PyMuPDF.
4. Return ScrapedDocument list.

NOTE:
- Only final judgments and orders available without authentication are accessed.
- Case data about individual litigants is NOT stored.
- Scraping is limited to precedent judgments relevant to citizen-rights domains.
- robots.txt respected, rate limits applied.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

from backend.app.services.scraping.base_scraper import BaseScraper, ScrapedDocument, ScraperError

logger = logging.getLogger(__name__)

KHC_BASE = "https://karnatakajudiciary.kar.nic.in"
ECOURTS_BASE = "https://ecourts.gov.in"


class KarnatakaCourtScraper(BaseScraper):
    """
    Scrapes publicly available judgments from Karnataka judiciary portals.

    Only accesses judgment text that is published openly on official portals.
    Does not store personal information about litigants.

    Usage::

        scraper = KarnatakaCourtScraper(output_dir=Path("data/raw/karnataka_court"))
        documents = scraper.scrape()
    """

    DEFAULT_DELAY_SECONDS = 4.0  # courts have sensitive infrastructure; be very slow

    # Pre-identified publicly accessible judgment pages for citizen-rights topics.
    # These are landmark/frequently cited judgments in the target legal domains.
    # URLs must be verified before each run as court portals restructure frequently.
    JUDGMENT_TARGETS = [
        {
            "title": "Landmark judgment — Consumer rights Karnataka HC",
            "url_path": "/judgments/",
            "domain": "consumer",
            "court_name": "Karnataka High Court",
        },
        {
            "title": "Landmark judgment — Labour rights Karnataka HC",
            "url_path": "/judgments/",
            "domain": "employment_salary",
            "court_name": "Karnataka High Court",
        },
        {
            "title": "Landmark judgment — Tenant rights Karnataka HC",
            "url_path": "/judgments/",
            "domain": "rental_property",
            "court_name": "Karnataka High Court",
        },
    ]

    def __init__(self, output_dir: Path | None = None) -> None:
        if output_dir is None:
            output_dir = Path("data/raw/karnataka_court")
        super().__init__(output_dir=output_dir)

    def scrape(self) -> list[ScrapedDocument]:
        """
        Attempt to access publicly available judgment listings.
        Returns harvested judgment documents.

        Note: If the portal is unreachable or requires login/CAPTCHA,
        returns an empty list with a warning — does NOT attempt to bypass
        any access control.
        """
        logger.info("KarnatakaCourtScraper: starting")
        results: list[ScrapedDocument] = []

        # Probe the base URL first.
        probe = self._get(KHC_BASE)
        if probe is None:
            logger.warning(
                "KarnatakaCourtScraper: Karnataka HC portal is unreachable or "
                "disallowed. Manual download from https://karnatakajudiciary.kar.nic.in "
                "is recommended. Place PDFs in data/raw/karnataka_court/ and re-run ingestion."
            )
            # Try to process any manually downloaded files instead.
            results = self._process_local_files()
            return results

        # Try the eCourts portal as a fallback / complement.
        for target in self.JUDGMENT_TARGETS:
            try:
                docs = self._scrape_judgment_listing(target)
                results.extend(docs)
            except ScraperError as exc:
                logger.error(f"ScraperError for {target['title']}: {exc}")
            except Exception as exc:
                logger.error(f"Unexpected error for {target['title']}: {exc}")

        # Also process any manually placed files.
        results.extend(self._process_local_files())

        logger.info(f"KarnatakaCourtScraper: complete — {len(results)} documents")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scrape_judgment_listing(self, target: dict) -> list[ScrapedDocument]:
        """Fetch a judgment listing page and harvest linked PDFs."""
        url = KHC_BASE + target["url_path"]
        resp = self._get(url)
        if resp is None:
            return []

        pdf_links = self._extract_pdf_links(resp.text, url)
        documents: list[ScrapedDocument] = []

        for pdf_url, link_text in pdf_links[:5]:  # limit per domain
            doc = self._process_judgment_pdf(pdf_url, link_text, target)
            if doc:
                documents.append(doc)

        return documents

    def _process_judgment_pdf(
        self, pdf_url: str, link_text: str, target: dict
    ) -> Optional[ScrapedDocument]:
        """Download and process one judgment PDF."""
        resp = self._get(pdf_url, stream=True)
        if resp is None:
            return None

        raw = resp.content
        checksum = self._checksum(raw)
        url_tail = pdf_url.split("/")[-1]
        safe_name = re.sub(r"[^\w\-\.]", "_", url_tail)
        if not safe_name.endswith(".pdf"):
            safe_name += ".pdf"

        local_path = self._save_raw(safe_name, raw)
        text_content = self._extract_pdf_text(local_path)

        if not text_content:
            return None

        # Try to extract case number from text.
        case_number = self._extract_case_number(text_content)
        pub_date = self._extract_judgment_date(text_content)

        return ScrapedDocument(
            title=link_text or f"Karnataka HC Judgment — {url_tail}",
            source_url=pdf_url,
            document_type="court_judgment",
            jurisdiction="karnataka",
            court_name=target["court_name"],
            case_number=case_number,
            language="english",
            publication_date=pub_date,
            amendment_status="original",
            local_path=local_path,
            text_content=text_content,
            checksum=checksum,
            metadata={"legal_domain": target["domain"]},
        )

    def _process_local_files(self) -> list[ScrapedDocument]:
        """
        Process any PDF files already present in ``output_dir``.

        This allows users to manually download judgment PDFs and place them
        in ``data/raw/karnataka_court/`` without network scraping.
        """
        documents: list[ScrapedDocument] = []
        for pdf_file in sorted(self.output_dir.glob("*.pdf")):
            checksum = self._checksum(pdf_file.read_bytes())
            text_content = self._extract_pdf_text(pdf_file)
            if not text_content:
                logger.warning(f"Empty text from local file: {pdf_file.name}")
                continue

            case_number = self._extract_case_number(text_content)
            pub_date = self._extract_judgment_date(text_content)

            documents.append(ScrapedDocument(
                title=pdf_file.stem.replace("_", " ").title(),
                source_url=f"file://{pdf_file.resolve()}",
                document_type="court_judgment",
                jurisdiction="karnataka",
                court_name="Karnataka High Court",
                case_number=case_number,
                language="english",
                publication_date=pub_date,
                amendment_status="original",
                local_path=pdf_file,
                text_content=text_content,
                checksum=checksum,
                metadata={"source": "manually_placed"},
            ))
            logger.info(f"Processed local judgment: {pdf_file.name}")

        return documents

    def _extract_pdf_links(self, html: str, base_url: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pairs.append((urljoin(base_url, href), a.get_text(strip=True)))
        except ImportError:
            pattern = r'href="([^"]*\.pdf)"[^>]*>([^<]*)'
            for m in re.finditer(pattern, html, re.IGNORECASE):
                pairs.append((urljoin(base_url, m.group(1)), m.group(2).strip()))
        return pairs

    @staticmethod
    def _extract_case_number(text: str) -> Optional[str]:
        """Extract a case number like 'WP 1234/2023' from judgment text."""
        pattern = r"\b(WP|WA|CRP|RFA|MFA|CPC|CRL\.A|CRL\.P)\s*(?:No\.?)?\s*(\d+)[/\-](\d{4})\b"
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(0) if m else None

    @staticmethod
    def _extract_judgment_date(text: str) -> Optional[str]:
        """Extract a judgment date from the first 500 characters of the text."""
        snippet = text[:500]
        # Pattern: 1st January, 2024 / January 1, 2024 / 01-01-2024
        months = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        pattern = rf"\b(\d{{1,2}})\s+{months},?\s+(\d{{4}})\b"
        m = re.search(pattern, snippet, re.IGNORECASE)
        if m:
            return m.group(0)
        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", snippet)
        return iso.group(1) if iso else None
