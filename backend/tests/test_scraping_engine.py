"""
Bureaucracy AI — Scraping Engine Tests

Tests for:
- BaseScraper helpers (checksum, PDF text extraction)
- ChangeDetector logic (new / updated / unchanged classification)
- ScraperManager orchestration
- IndiaCodeScraper / GazetteScraper instantiation

All tests run offline — no actual HTTP requests are made.
Network-dependent tests are marked with @unittest.skip or mocked.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.app.services.scraping.base_scraper import BaseScraper, ScrapedDocument
from backend.app.services.scraping.change_detector import (
    ChangeDetector,
    ChangeStatus,
    ChangeResult,
)


# ---------------------------------------------------------------------------
# Minimal concrete scraper for testing BaseScraper
# ---------------------------------------------------------------------------

class _DummyScraper(BaseScraper):
    """Minimal scraper for testing base class helpers."""

    def scrape(self):
        return []


# ---------------------------------------------------------------------------
# Test: BaseScraper helpers
# ---------------------------------------------------------------------------

class TestBaseScraper(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.scraper = _DummyScraper(output_dir=self.tmp_dir, respect_robots=False)

    def test_checksum_deterministic(self):
        data = b"Hello, Karnataka Legal System"
        c1 = self.scraper._checksum(data)
        c2 = self.scraper._checksum(data)
        self.assertEqual(c1, c2)
        expected = hashlib.sha256(data).hexdigest()
        self.assertEqual(c1, expected)

    def test_checksum_different_data(self):
        c1 = self.scraper._checksum(b"abc")
        c2 = self.scraper._checksum(b"xyz")
        self.assertNotEqual(c1, c2)

    def test_save_raw(self):
        content = b"Test legal document bytes"
        path = self.scraper._save_raw("test_doc.pdf", content)
        self.assertTrue(path.exists())
        self.assertEqual(path.read_bytes(), content)

    def test_output_dir_created(self):
        new_dir = self.tmp_dir / "subdir" / "nested"
        scraper = _DummyScraper(output_dir=new_dir)
        self.assertTrue(new_dir.exists())

    def test_robots_allowed_when_respect_false(self):
        """When respect_robots=False all URLs are allowed."""
        allowed = self.scraper._check_robots("http://example.com/anything")
        self.assertTrue(allowed)

    def test_rate_limit_no_crash(self):
        """Rate limiter should not raise for normal calls."""
        self.scraper._rate_limit()
        self.scraper._rate_limit()


# ---------------------------------------------------------------------------
# Test: ScrapedDocument dataclass
# ---------------------------------------------------------------------------

class TestScrapedDocument(unittest.TestCase):

    def test_required_fields(self):
        doc = ScrapedDocument(
            title="Test Act",
            source_url="https://example.com/test",
            document_type="central_act",
            jurisdiction="central",
        )
        self.assertEqual(doc.title, "Test Act")
        self.assertEqual(doc.jurisdiction, "central")
        self.assertEqual(doc.language, "english")
        self.assertEqual(doc.amendment_status, "unknown")
        self.assertIsNone(doc.checksum)

    def test_scraped_at_set_automatically(self):
        doc = ScrapedDocument(
            title="Test", source_url="https://x.com", document_type="act", jurisdiction="central"
        )
        self.assertIsNotNone(doc.scraped_at)
        self.assertTrue(len(doc.scraped_at) > 0)


# ---------------------------------------------------------------------------
# Test: ChangeDetector
# ---------------------------------------------------------------------------

class TestChangeDetector(unittest.TestCase):

    def _make_doc(self, url: str, checksum: str | None = None) -> ScrapedDocument:
        return ScrapedDocument(
            title="Test Act",
            source_url=url,
            document_type="central_act",
            jurisdiction="central",
            checksum=checksum,
        )

    def test_new_document_no_db_match(self):
        """Document not in DB should be classified as NEW."""
        detector = ChangeDetector()
        doc = self._make_doc("https://example.com/never-seen-before-12345")

        # Patch the DB lookup to simulate no existing record.
        with patch(
            "backend.app.services.scraping.change_detector.get_connection"
        ) as mock_conn_ctx:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            mock_conn_ctx.return_value.__enter__.return_value = mock_conn

            result = detector._classify_one(doc)

        self.assertEqual(result.status, ChangeStatus.NEW)
        self.assertIsNone(result.prior_version)

    def test_unchanged_same_checksum(self):
        """Same checksum → UNCHANGED."""
        detector = ChangeDetector()
        checksum = "abc123def456"
        doc = self._make_doc("https://example.com/act.pdf", checksum=checksum)

        with patch(
            "backend.app.services.scraping.change_detector.get_connection"
        ) as mock_conn_ctx:
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, key: {
                "id": 42, "checksum": checksum, "version": "1.0"
            }[key]
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_conn_ctx.return_value.__enter__.return_value = mock_conn

            result = detector._classify_one(doc)

        self.assertEqual(result.status, ChangeStatus.UNCHANGED)
        self.assertEqual(result.prior_version, 42)

    def test_updated_different_checksum(self):
        """Different checksum with existing DB record → UPDATED."""
        detector = ChangeDetector()
        doc = self._make_doc("https://example.com/act.pdf", checksum="new_checksum")

        with patch(
            "backend.app.services.scraping.change_detector.get_connection"
        ) as mock_conn_ctx:
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, key: {
                "id": 7, "checksum": "old_checksum", "version": "1.0"
            }[key]
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_conn_ctx.return_value.__enter__.return_value = mock_conn

            result = detector._classify_one(doc)

        self.assertEqual(result.status, ChangeStatus.UPDATED)
        self.assertEqual(result.prior_checksum, "old_checksum")
        # Version should be bumped.
        self.assertEqual(doc.version, "1.1")

    def test_summary_counts(self):
        """Summary returns correct counts for each status."""
        results = [
            ChangeResult(document=MagicMock(), status=ChangeStatus.NEW),
            ChangeResult(document=MagicMock(), status=ChangeStatus.NEW),
            ChangeResult(document=MagicMock(), status=ChangeStatus.UPDATED),
            ChangeResult(document=MagicMock(), status=ChangeStatus.UNCHANGED),
        ]
        summary = ChangeDetector.summary(results)
        self.assertEqual(summary["new"], 2)
        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["unchanged"], 1)

    def test_filter_new_or_updated(self):
        """filter_new_or_updated returns only NEW and UPDATED documents."""
        docs = [MagicMock(), MagicMock(), MagicMock()]
        results = [
            ChangeResult(document=docs[0], status=ChangeStatus.NEW),
            ChangeResult(document=docs[1], status=ChangeStatus.UNCHANGED),
            ChangeResult(document=docs[2], status=ChangeStatus.UPDATED),
        ]
        to_ingest = ChangeDetector.filter_new_or_updated(results)
        self.assertIn(docs[0], to_ingest)
        self.assertNotIn(docs[1], to_ingest)
        self.assertIn(docs[2], to_ingest)

    def test_db_error_treated_as_new(self):
        """DB lookup failure must default to NEW — never skip documents silently."""
        detector = ChangeDetector()
        doc = self._make_doc("https://example.com/act.pdf")

        with patch(
            "backend.app.services.scraping.change_detector.get_connection",
            side_effect=Exception("DB offline"),
        ):
            result = detector._classify_one(doc)

        self.assertEqual(result.status, ChangeStatus.NEW)


# ---------------------------------------------------------------------------
# Test: Scraper instantiation (no network)
# ---------------------------------------------------------------------------

class TestScraperInstantiation(unittest.TestCase):

    def test_india_code_scraper_instantiates(self):
        from backend.app.services.scraping.india_code_scraper import IndiaCodeScraper
        s = IndiaCodeScraper(output_dir=Path(tempfile.mkdtemp()))
        self.assertIsNotNone(s)
        self.assertEqual(s.DEFAULT_DELAY_SECONDS, 3.0)
        self.assertGreater(len(s.TARGET_ACTS), 0)

    def test_gazette_scraper_instantiates(self):
        from backend.app.services.scraping.gazette_scraper import GazetteScraper
        s = GazetteScraper(output_dir=Path(tempfile.mkdtemp()))
        self.assertIsNotNone(s)
        self.assertGreater(len(s.SEARCH_TARGETS), 0)

    def test_karnataka_court_scraper_instantiates(self):
        from backend.app.services.scraping.karnataka_court_scraper import KarnatakaCourtScraper
        s = KarnatakaCourtScraper(output_dir=Path(tempfile.mkdtemp()))
        self.assertIsNotNone(s)

    def test_scraper_manager_instantiates(self):
        from backend.app.services.scraping.scraper_manager import ScraperManager
        m = ScraperManager()
        self.assertIsNotNone(m)


# ---------------------------------------------------------------------------
# Test: IndiaCodeScraper helpers
# ---------------------------------------------------------------------------

class TestIndiaCodeScraperHelpers(unittest.TestCase):

    def setUp(self):
        from backend.app.services.scraping.india_code_scraper import IndiaCodeScraper
        self.scraper = IndiaCodeScraper(output_dir=Path(tempfile.mkdtemp()))

    def test_find_act_url_found(self):
        html = '<a href="/handle/123456789/5678">Payment of Wages Act</a>'
        url = self.scraper._find_act_url(html, "Payment of Wages Act")
        self.assertIsNotNone(url)
        self.assertIn("handle", url)

    def test_find_act_url_not_found(self):
        html = "<p>No results found</p>"
        url = self.scraper._find_act_url(html, "Nonexistent Act")
        self.assertIsNone(url)

    def test_find_pdf_link(self):
        html = '<a href="/bitstream/act.pdf">Download PDF</a>'
        url = self.scraper._find_pdf_link(html, "https://example.com/handle/123")
        self.assertIsNotNone(url)
        self.assertIn(".pdf", url)


# ---------------------------------------------------------------------------
# Test: GazetteScraper helpers
# ---------------------------------------------------------------------------

class TestGazetteScraperHelpers(unittest.TestCase):

    def setUp(self):
        from backend.app.services.scraping.gazette_scraper import GazetteScraper
        self.scraper = GazetteScraper(output_dir=Path(tempfile.mkdtemp()))

    def test_extract_date_iso(self):
        date = self.scraper._extract_date("notification_2024-05-15.pdf")
        self.assertEqual(date, "2024-05-15")

    def test_extract_date_dmy(self):
        date = self.scraper._extract_date("Gazette 01/06/2023")
        self.assertEqual(date, "2023-06-01")

    def test_extract_date_none(self):
        date = self.scraper._extract_date("no date here")
        self.assertIsNone(date)


# ---------------------------------------------------------------------------
# Test: KarnatakaCourtScraper helpers
# ---------------------------------------------------------------------------

class TestKarnatakaCourtScraperHelpers(unittest.TestCase):

    def setUp(self):
        from backend.app.services.scraping.karnataka_court_scraper import KarnatakaCourtScraper
        self.scraper = KarnatakaCourtScraper(output_dir=Path(tempfile.mkdtemp()))

    def test_extract_case_number_wp(self):
        text = "In the High Court of Karnataka. WP No. 1234/2023 dated 15 January, 2024."
        case_no = self.scraper._extract_case_number(text)
        self.assertIsNotNone(case_no)
        self.assertIn("1234", case_no)

    def test_extract_case_number_none(self):
        text = "This judgment has no recognizable case number pattern."
        case_no = self.scraper._extract_case_number(text)
        self.assertIsNone(case_no)

    def test_extract_judgment_date(self):
        text = "Pronounced on 15 March, 2024. The High Court of Karnataka."
        date = self.scraper._extract_judgment_date(text)
        self.assertIsNotNone(date)
        self.assertIn("2024", date)

    def test_process_local_files_empty_dir(self):
        """Empty output_dir should return empty list without error."""
        docs = self.scraper._process_local_files()
        self.assertEqual(docs, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
