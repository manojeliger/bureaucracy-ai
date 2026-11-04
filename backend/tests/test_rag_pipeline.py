import unittest

from backend.app.services.rag.chunker import DocumentChunker
from backend.app.services.rag.metadata import (
    DocumentMetadata,
    ChunkMetadata
)


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.chunker = DocumentChunker(
            chunk_size=1000,
            chunk_overlap=150
        )

        self.document = DocumentMetadata(
            document_id="KA-001",
            title="Income Certificate Guidelines",
            department="Karnataka Revenue Department",
            source_url="https://example.gov.in",
            language="en"
        )

    def test_document_is_chunked(self):
        text = (
            "An applicant must provide valid residence proof. "
            "The applicant must also provide income details. "
        ) * 15

        chunks = self.chunker.split_text(text)

        self.assertGreater(len(chunks), 1)

    def test_chunk_metadata_creation(self):
        metadata = ChunkMetadata(
            document=self.document,
            page_number=2,
            chunk_index=0,
            total_chunks=3
        )

        self.assertEqual(metadata.document.document_id, "KA-001")
        self.assertEqual(metadata.chunk_index, 0)
        self.assertEqual(metadata.page_number, 2)

    def test_metadata_for_all_chunks(self):
        text = "Government certificate application procedure. " * 20

        chunks = self.chunker.split_text(text)

        metadata_records = [
            ChunkMetadata(
                document=self.document,
                page_number=1,
                chunk_index=index,
                total_chunks=len(chunks)
            )
            for index in range(len(chunks))
        ]

        self.assertEqual(len(metadata_records), len(chunks))

        for index, record in enumerate(metadata_records):
            self.assertEqual(record.chunk_index, index)
            self.assertEqual(record.total_chunks, len(chunks))


if __name__ == "__main__":
    unittest.main()