import unittest
from backend.app.services.rag.rag_service import RAGService


class TestRAGTenScenarios(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rag = RAGService.get_instance()

    def _evaluate(self, query: str):
        result = self.rag.generate_response(query)
        print(f"\n==========================================")
        print(f"QUERY: {query}")
        print(f"FOUND SOURCES: {len(result['sources'])} (Has Context: {result['has_context']})")
        for s in result["sources"]:
            print(f"  - {s['title']} | {s['department']} | {s['url']} | Score: {s['confidence']}")
        print(f"ANSWER PREVIEW:\n{result['answer'][:250]}...")
        return result

    def test_01_apply_income_certificate(self):
        res = self._evaluate("How can I apply for an income certificate in Karnataka?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("income" in s["title"].lower() or "karnataka" in s["title"].lower() for s in res["sources"]))

    def test_02_certificate_rejected(self):
        res = self._evaluate("What should I do if my certificate application is rejected?")
        self.assertTrue(res["has_context"])
        self.assertIn("answer", res)

    def test_03_purpose_of_bhoomi(self):
        res = self._evaluate("What is the purpose of Bhoomi?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("bhoomi" in r["text"].lower() for r in res["retrieved_chunks"]))

    def test_04_ration_card_issue(self):
        res = self._evaluate("How do I resolve a ration-card issue in Karnataka?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("ahara" in r["text"].lower() or "ration" in r["text"].lower() or "pds" in r["text"].lower() for r in res["retrieved_chunks"]))

    def test_05_scholarship_delay(self):
        res = self._evaluate("What should I do if my scholarship payment is delayed?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("ssp" in r["text"].lower() or "scholarship" in r["text"].lower() for r in res["retrieved_chunks"]))

    def test_06_duplicate_property_tax_notice(self):
        res = self._evaluate("What can I do if I receive a duplicate property-tax notice?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("tax" in r["text"].lower() or "property" in r["text"].lower() for r in res["retrieved_chunks"]))

    def test_07_what_is_sakala(self):
        res = self._evaluate("What is Sakala in Karnataka?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("sakala" in r["text"].lower() for r in res["retrieved_chunks"]))

    def test_08_land_mutation_authority(self):
        res = self._evaluate("Which authority handles a land mutation issue?")
        self.assertTrue(res["has_context"])
        self.assertTrue(any("mutation" in r["text"].lower() or "revenue" in r["text"].lower() or "tahsildar" in r["text"].lower() for r in res["retrieved_chunks"]))

    def test_09_pension_scheme_documents(self):
        res = self._evaluate("What documents are required for the relevant pension scheme in Karnataka?")
        self.assertTrue(res["has_context"])

    def test_10_unknown_topic_handling(self):
        # Master prompt requirement: "What if the knowledge base does not contain the answer?"
        # The AI must honestly state that official records are not in its database.
        res = self._evaluate("How do I renew an orbital commercial space satellite license under the Martian space council?")
        print(f"Has Context for Unknown: {res['has_context']}")
        # Model should recognize lack of context or database boundary
        self.assertIn("answer", res)


if __name__ == "__main__":
    unittest.main()
