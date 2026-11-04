import unittest

from backend.app.services.rag.chunker import DocumentChunker


class TestDocumentChunker(unittest.TestCase):
    def test_overlap_preserves_complete_words(self):
        chunker = DocumentChunker(
           chunk_size=100,
           chunk_overlap=20
    )

        text = "Eligibility requires valid residence proof. " * 8
        chunks = chunker.split_text(text)

        for chunk in chunks[1:]:
           self.assertTrue(
               chunk.startswith("Eligibility") or
               chunk[0].isalpha()
        )
           self.assertFalse(chunk.startswith("lid "))


    def setUp(self):
        self.chunker = DocumentChunker(
            chunk_size=1000,
            chunk_overlap=150
        )

    def test_empty_text(self):
        self.assertEqual(self.chunker.split_text(""), [])

    def test_normal_text(self):
        text = "This is the first sentence. This is the second sentence."
        chunks = self.chunker.split_text(text)
        self.assertGreater(len(chunks), 0)

    def test_chunk_size_limit(self):
        text = "A" * 2500
        chunks = self.chunker.split_text(text)

        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)

    def test_long_text_overlap(self):
        text = "A" * 2500
        chunks = self.chunker.split_text(text)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][-150:], chunks[1][:150])

    def test_overlap_does_not_exceed_chunk_size(self):
        chunker = DocumentChunker(
            chunk_size=100,
            chunk_overlap=20
        )

        text = ("A" * 89 + ". ") + ("B" * 89 + ".")
        chunks = chunker.split_text(text)

        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            DocumentChunker(chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()