import unittest

from backend.app.services.rag.embedding_service import EmbeddingService


class TestEmbeddingService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = EmbeddingService()

    def test_query_embedding_dimension(self):
        embedding = self.service.embed_query(
            "What documents are required for a residence certificate?"
        )

        self.assertEqual(embedding.shape, (384,))

    def test_document_embedding_dimension(self):
        embeddings = self.service.embed_documents([
            "Applicants must provide proof of residence.",
            "Applicants must submit an income certificate."
        ])

        self.assertEqual(embeddings.shape, (2, 384))

    def test_query_embedding_normalized(self):
        embedding = self.service.embed_query("query test")

        self.assertAlmostEqual(
            float((embedding ** 2).sum() ** 0.5),
            1.0,
            places=5
        )

    def test_empty_query_rejected(self):
        with self.assertRaises(ValueError):
            self.service.embed_query("")

    def test_empty_document_list(self):
        embeddings = self.service.embed_documents([])

        self.assertEqual(embeddings, [])

    def test_empty_document_rejected(self):
        with self.assertRaises(ValueError):
            self.service.embed_documents([
                "Valid document",
                "   "
            ])


if __name__ == "__main__":
    unittest.main()