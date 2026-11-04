import re
from typing import List


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "Overlap must be non-negative and smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_overlap_words(self, words: List[str]) -> List[str]:
        overlap = []
        length = 0

        for word in reversed(words):
            extra_length = len(word) + (1 if overlap else 0)

            if length + extra_length > self.chunk_overlap:
                break

            overlap.insert(0, word)
            length += extra_length

        return overlap

    def split_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []

        text = re.sub(r"\s+", " ", text).strip()

        words = text.split()
        chunks = []
        current_words = []

        for word in words:

            # Handle exceptionally long words separately.
            if len(word) > self.chunk_size:
                if current_words:
                    chunks.append(" ".join(current_words))
                    current_words = []

                step = self.chunk_size - self.chunk_overlap

                for start in range(0, len(word), step):
                    chunks.append(
                        word[start:start + self.chunk_size]
                    )

                continue

            candidate = " ".join(current_words + [word])

            if len(candidate) <= self.chunk_size:
                current_words.append(word)

            else:
                if current_words:
                    chunks.append(" ".join(current_words))

                # Preserve complete words in the overlap.
                overlap_words = self._get_overlap_words(current_words)

                # Ensure overlap + next word never exceeds chunk size.
                while overlap_words and (
                    len(" ".join(overlap_words + [word]))
                    > self.chunk_size
                ):
                    overlap_words.pop(0)

                current_words = overlap_words + [word]

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks