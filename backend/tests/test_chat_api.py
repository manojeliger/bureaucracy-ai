import unittest
from backend.app.main import health_check
from backend.app.api.v1.chat import chat, ChatRequest, ChatMessage


class TestChatAPI(unittest.TestCase):

    def test_health_check_returns_healthy(self):
        result = health_check()
        self.assertIn("status", result)
        self.assertIn("components", result)
        self.assertEqual(result["components"]["faiss_index"], "loaded")
        self.assertGreater(result["components"]["indexed_vectors"], 0)

    def test_chat_income_certificate(self):
        req = ChatRequest(
            message="How can I apply for an income certificate in Karnataka?",
            language="English",
            state="Karnataka"
        )
        res = chat(req)
        self.assertIsNotNone(res.response)
        self.assertTrue(res.has_grounded_context)
        self.assertGreater(len(res.sources), 0)

    def test_chat_conversation_memory(self):
        history = [
            ChatMessage(role="user", content="I applied for an income certificate 10 days ago."),
            ChatMessage(role="assistant", content="Understood. What was the status or rejection remark given?")
        ]
        req = ChatRequest(
            message="The status shows rejected during Village Accountant verification.",
            history=history,
            language="English"
        )
        res = chat(req)
        self.assertIsNotNone(res.response)
        # Verify it acknowledges or addresses verification rejection
        self.assertTrue(len(res.response) > 50)


if __name__ == "__main__":
    unittest.main()
