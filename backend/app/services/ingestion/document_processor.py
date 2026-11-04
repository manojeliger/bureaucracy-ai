from typing import Dict, List

from backend.app.services.ingestion.pdf_extractor import PDFExtractor
from backend.app.services.rag.chunker import DocumentChunker
from backend.app.services.rag.metadata import (
    DocumentMetadata,
    ChunkMetadata
)


class DocumentProcessor:

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.chunker = DocumentChunker()

    def process_pdf(
        self,
        file_path: str,
        document_metadata: DocumentMetadata
    ) -> List[Dict]:

        pages = self.pdf_extractor.extract_text(file_path)

        processed_chunks = []

        # Process each PDF page separately.
        for page in pages:

            page_number = page["page_number"]
            page_text = page["text"]

            if not page_text:
                continue

            chunks = self.chunker.split_text(page_text)

            for chunk in chunks:
                processed_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "document": document_metadata,
                        "page_number": page_number
                    }
                })

        total_chunks = len(processed_chunks)

        # Attach final chunk indexes and total count.
        for index, record in enumerate(processed_chunks):
            record["metadata"] = ChunkMetadata(
                document=record["metadata"]["document"],
                page_number=record["metadata"]["page_number"],
                chunk_index=index,
                total_chunks=total_chunks
            )

        return processed_chunks