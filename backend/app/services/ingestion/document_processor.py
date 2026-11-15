"""
Document processor — converts a raw document file into text chunks
with structured metadata, ready for embedding and FAISS indexing.

Also provides classify_document_type(), used by the upload API.
"""

import re
from typing import Dict, List

from backend.app.services.ingestion.pdf_extractor import PDFExtractor
from backend.app.services.rag.chunker import DocumentChunker
from backend.app.services.rag.metadata import ChunkMetadata, DocumentMetadata


# ---------------------------------------------------------------------------
# Document type classification
# ---------------------------------------------------------------------------

_DOC_TYPE_PATTERNS = [
    ("legal_notice",        r"legal\s+notice|notice\s+to\s+show\s+cause|demand\s+notice"),
    ("rental_agreement",    r"rental\s+agreement|lease\s+deed|tenancy\s+agreement"),
    ("fir",                 r"\bfir\b|first\s+information\s+report|case\s+no\.?\s+\d"),
    ("court_order",         r"high\s+court|supreme\s+court|district\s+court|order\s+dated|judgment|decree"),
    ("government_notice",   r"government\s+of\s+karnataka|government\s+of\s+india|circular|gazette|notification"),
    ("rejection_order",     r"reject|not\s+eligible|application.*declined|hereby\s+rejected"),
    ("employment_contract", r"employment\s+agreement|appointment\s+letter|offer\s+of\s+employment|terms\s+of\s+service"),
    ("consumer_complaint",  r"consumer\s+complaint|consumer\s+forum|deficiency\s+of\s+service|consumer\s+protection"),
    ("tax_notice",          r"income\s+tax|property\s+tax|gst\s+notice|demand\s+for\s+payment"),
    ("certificate",         r"income\s+certificate|caste\s+certificate|domicile|residence\s+certificate"),
]


def classify_document_type(text: str, filename: str = "") -> str:
    """
    Heuristically classify a legal document type from its text content.

    Returns a string key like 'rental_agreement', 'fir', 'court_order', etc.
    Falls back to 'unknown'.
    """
    sample = (text[:3000] + " " + filename).lower()
    for doc_type, pattern in _DOC_TYPE_PATTERNS:
        if re.search(pattern, sample, re.IGNORECASE):
            return doc_type
    return "unknown"


# ---------------------------------------------------------------------------
# Document processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Converts a PDF file on disk into a list of chunk dicts,
    each carrying a ChunkMetadata object.
    """

    def __init__(self):
        self._pdf_extractor = PDFExtractor()
        self._chunker = DocumentChunker()

    def process_pdf(
        self,
        file_path: str,
        document_metadata: DocumentMetadata,
    ) -> List[Dict]:
        """
        Extract text from a PDF and split into overlapping chunks.

        Returns a list of dicts:
            {
                "text": str,
                "metadata": ChunkMetadata,
            }
        """
        pages = self._pdf_extractor.extract_text(file_path)

        processed = []
        for page in pages:
            page_number = page["page_number"]
            page_text = page["text"]

            if not page_text:
                continue

            chunks = self._chunker.split_text(page_text)
            for chunk in chunks:
                processed.append({
                    "text": chunk,
                    "metadata": ChunkMetadata(
                        document=document_metadata,
                        page_number=page_number,
                        chunk_index=0,       # Will be set after all pages are collected
                        total_chunks=1,      # Will be set after all pages are collected
                        section_reference=self._detect_section(chunk),
                    ),
                })

        total = len(processed)
        for idx, record in enumerate(processed):
            record["metadata"] = ChunkMetadata(
                document=record["metadata"].document,
                page_number=record["metadata"].page_number,
                chunk_index=idx,
                total_chunks=total,
                section_reference=record["metadata"].section_reference,
                chapter_reference=record["metadata"].chapter_reference,
            )

        return processed

    @staticmethod
    def _detect_section(text: str) -> str | None:
        """
        Detect a section/article/rule reference at the start of a chunk.
        e.g. "Section 14", "Rule 3", "Article 21", "Chapter IV"
        """
        match = re.match(
            r"^(section|sec\.?|rule|article|art\.?|chapter|clause|schedule)\s+[\dIVXivx]+",
            text.strip(),
            re.IGNORECASE,
        )
        return match.group(0).strip() if match else None
