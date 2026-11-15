from sentence_transformers import SentenceTransformer


class EmbeddingService:

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small"
    ):
        self.model = SentenceTransformer(
            model_name,
            device="cpu"
        )

    def embed_query(self, text: str):
        """
        Convert a user query into a normalized embedding.
        """
        if not text or not text.strip():
            raise ValueError("Query cannot be empty.")

        return self.model.encode(
            f"query: {text.strip()}",
            normalize_embeddings=True
        )

    def embed_documents(self, texts: list[str]):
        """
        Convert document chunks into normalized embeddings.
        """
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError(
                "Document texts cannot contain empty values."
            )

        passages = [
            f"passage: {text.strip()}"
            for text in texts
        ]

        return self.model.encode(
            passages,
            normalize_embeddings=True
        )