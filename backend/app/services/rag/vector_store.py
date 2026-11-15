"""
FAISS vector store for Bureaucracy AI.

Uses IndexFlatIP (inner-product) with pre-normalized vectors so cosine
similarity equals the inner product. Embeddings are normalized once — by
EmbeddingService — before being passed here. This store does NOT normalize
again (that was the original double-normalization bug).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, embeddings, metadata: List[Dict[str, Any]]) -> None:
        """
        Add pre-normalized embeddings and their metadata to the index.

        IMPORTANT: Embeddings MUST already be L2-normalized before calling
        this method (EmbeddingService.embed_documents handles this).
        Do NOT normalize here — that would cause double-normalization.
        """
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (n, {self.dimension}), got {vectors.shape}"
            )

        if len(vectors) != len(metadata):
            raise ValueError(
                f"Embeddings count ({len(vectors)}) != metadata count ({len(metadata)})"
            )

        if len(vectors) == 0:
            return

        self.index.add(vectors)
        self.metadata.extend(metadata)
        logger.debug(f"Added {len(vectors)} vectors. Total: {self.index.ntotal}")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most similar chunks for a query embedding.

        The query embedding MUST already be L2-normalized (EmbeddingService
        does this). We do NOT normalize here.

        Returns a list of dicts: {"score": float, "metadata": dict, "faiss_index": int}
        """
        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension {query.shape[1]} != index dimension {self.dimension}"
            )

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if idx >= len(self.metadata):
                logger.warning(f"FAISS returned index {idx} but metadata has {len(self.metadata)} entries")
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx],
                "faiss_index": int(idx),
            })

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str = "data/processed/faiss") -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "knowledge.index"))

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved FAISS index with {self.index.ntotal} vectors to {path}")

    def load(self, directory: str = "data/processed/faiss") -> None:
        path = Path(directory)

        self.index = faiss.read_index(str(path / "knowledge.index"))

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.index.d != self.dimension:
            raise ValueError(
                f"Stored index dimension {self.index.d} != configured dimension {self.dimension}"
            )

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index has {self.index.ntotal} vectors but metadata has {len(self.metadata)} entries. "
                "Re-run the ingestion pipeline."
            )

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors from {path}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def is_loaded(self) -> bool:
        return self.index.ntotal > 0
