import unittest
from backend.app.services.rag.rag_service import RAGService


class TestRAGService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = RAGService.get_instance()

    def test_faiss_index_is_loaded(self):
        self.assertTrue(self.service.is_index_loaded)
        self.assertGreater(self.service.vector_store.index.ntotal, 0)

    def test_retrieval_income_certificate(self):
        query = "How can I apply for an income certificate in Karnataka?"
        results = self.service.retrieve(query, top_k=3, threshold=0.6)
        self.assertGreater(len(results), 0)
        top_result = results[0]
        self.assertIn("score", top_result)
        self.assertGreater(top_result["score"], 0.6)
        self.assertTrue(
            "revenue" in top_result["department"].lower() or
            "karnataka" in top_result["title"].lower() or
            "certificate" in top_result["text"].lower()
        )

    def test_retrieval_bhoomi_portal(self):
        query = "What is the purpose of Bhoomi?"
        results = self.service.retrieve(query, top_k=3, threshold=0.6)
        self.assertGreater(len(results), 0)
        found_bhoomi = any("bhoomi" in r["text"].lower() for r in results)
        self.assertTrue(found_bhoomi)

    def test_context_formatting(self):
        results = self.service.retrieve("Sakala", top_k=2, threshold=0.5)
        context = self.service.format_context(results)
        self.assertIn("[Source 1:", context)

    def test_empty_query_for_completely_irrelevant_topic(self):
        # Extremely high threshold should yield 0 results
        results = self.service.retrieve("Quantum astrophysical planetary fusion", top_k=3, threshold=0.95)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
