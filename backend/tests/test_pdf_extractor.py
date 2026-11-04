import unittest
import tempfile
from pathlib import Path

import pymupdf

from backend.app.services.ingestion.pdf_extractor import PDFExtractor


class TestPDFExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = PDFExtractor()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "sample.pdf"

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_sample_pdf(self):
        document = pymupdf.open()

        page1 = document.new_page()
        page1.insert_text(
            (72, 72),
            "Income certificate eligibility requirements."
        )

        page2 = document.new_page()
        page2.insert_text(
            (72, 72),
            "Required documents and application procedure."
        )

        document.save(self.pdf_path)
        document.close()

    def test_extract_text(self):
        self.create_sample_pdf()

        pages = self.extractor.extract_text(str(self.pdf_path))

        self.assertEqual(len(pages), 2)
        self.assertIn("Income certificate", pages[0]["text"])

    def test_page_numbers(self):
        self.create_sample_pdf()

        pages = self.extractor.extract_text(str(self.pdf_path))

        self.assertEqual(pages[0]["page_number"], 1)
        self.assertEqual(pages[1]["page_number"], 2)

    def test_empty_page_detection(self):
        document = pymupdf.open()
        document.new_page()
        document.save(self.pdf_path)
        document.close()

        pages = self.extractor.extract_text(str(self.pdf_path))

        self.assertFalse(pages[0]["has_text"])

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_text(
                str(Path(self.temp_dir.name) / "missing.pdf")
            )

    def test_invalid_pdf(self):
        invalid_path = Path(self.temp_dir.name) / "invalid.pdf"
        invalid_path.write_text("This is not a PDF.")

        with self.assertRaises(ValueError):
            self.extractor.extract_text(str(invalid_path))


if __name__ == "__main__":
    unittest.main()