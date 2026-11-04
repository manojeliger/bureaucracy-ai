import unittest
import tempfile
from pathlib import Path

import pymupdf

from backend.app.services.ingestion.document_processor import DocumentProcessor
from backend.app.services.rag.metadata import DocumentMetadata


class TestDocumentProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = DocumentProcessor()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "government_document.pdf"

        self.document_metadata = DocumentMetadata(
            document_id="KA-001",
            title="Income Certificate Guidelines",
            department="Karnataka Revenue Department",
            source_url="https://example.gov.in",
            language="en"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_sample_pdf(self):
        document = pymupdf.open()

        page1 = document.new_page()
        page1.insert_text(
            (72, 72),
            "Income certificate eligibility requirements. " * 15
        )

        page2 = document.new_page()
        page2.insert_text(
            (72, 72),
            "Required documents and application procedure. " * 15
        )

        document.save(self.pdf_path)
        document.close()

    def test_process_pdf(self):
        self.create_sample_pdf()

        chunks = self.processor.process_pdf(
            str(self.pdf_path),
            self.document_metadata
        )

        self.assertGreater(len(chunks), 0)

    def test_chunk_text_exists(self):
        self.create_sample_pdf()

        chunks = self.processor.process_pdf(
            str(self.pdf_path),
            self.document_metadata
        )

        for chunk in chunks:
            self.assertTrue(chunk["text"].strip())

    def test_page_metadata_preserved(self):
        self.create_sample_pdf()

        chunks = self.processor.process_pdf(
            str(self.pdf_path),
            self.document_metadata
        )

        page_numbers = {
            chunk["metadata"].page_number
            for chunk in chunks
        }

        self.assertEqual(page_numbers, {1, 2})

    def test_document_metadata_preserved(self):
        self.create_sample_pdf()

        chunks = self.processor.process_pdf(
            str(self.pdf_path),
            self.document_metadata
        )

        for chunk in chunks:
            metadata = chunk["metadata"]

            self.assertEqual(
                metadata.document.document_id,
                "KA-001"
            )

    def test_chunk_indexes(self):
        self.create_sample_pdf()

        chunks = self.processor.process_pdf(
            str(self.pdf_path),
            self.document_metadata
        )

        for index, chunk in enumerate(chunks):
            self.assertEqual(
                chunk["metadata"].chunk_index,
                index
            )

            self.assertEqual(
                chunk["metadata"].total_chunks,
                len(chunks)
            )


if __name__ == "__main__":
    unittest.main()