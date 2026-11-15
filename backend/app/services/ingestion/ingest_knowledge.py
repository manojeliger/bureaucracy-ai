"""
Legal knowledge ingestion pipeline for Bureaucracy AI.

Processes raw data files from data/raw/ and builds the FAISS index + metadata.json.
Supports PDF, JSON, and CSV sources.

Fixes over v0.1:
- Chunk IDs are now globally unique (document_id + page + chunk_index — no duplicates).
- Metadata follows the full legal schema (jurisdiction, verification_status, etc.).
- Double-normalization bug removed (embeddings normalized once in EmbeddingService).
- Legal chunks are also written to SQLite for FTS keyword search.
"""

import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running directly from the project root
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.database.connection import init_db
from backend.app.services.database.repositories import LegalChunkRepository, LegalSourceRepository
from backend.app.services.ingestion.document_processor import DocumentProcessor
from backend.app.services.rag.chunker import DocumentChunker
from backend.app.services.rag.embedding_service import EmbeddingService
from backend.app.services.rag.metadata import (
    ChunkMetadata,
    DocumentMetadata,
    DocumentType,
    Jurisdiction,
    VerificationStatus,
)
from backend.app.services.rag.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed" / "faiss"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_chunk_id(document_id: str, page: int, chunk_index: int) -> str:
    """
    Generate a globally unique chunk ID.
    Format: {document_id}-p{page:04d}-c{chunk_index:04d}
    Example: KAR-KNOWLEDGE-001-p0001-c0003
    """
    return f"{document_id}-p{page:04d}-c{chunk_index:04d}"


def _make_flat_metadata(
    chunk_id: str,
    chunk_text: str,
    source: str,
    doc_type: str,
    chunk_meta: ChunkMetadata,
) -> dict:
    """Build the flat dict stored in metadata.json and FAISS."""
    doc = chunk_meta.document
    return {
        "chunk_id": chunk_id,
        "text": chunk_text,
        "source": source,
        "type": doc_type,
        "title": doc.title,
        "department": doc.department,
        "source_url": doc.source_url,
        "section_reference": chunk_meta.section_reference,
        "chapter_reference": chunk_meta.chapter_reference,
        "page_number": chunk_meta.page_number,
        "jurisdiction": doc.jurisdiction.value if hasattr(doc.jurisdiction, "value") else str(doc.jurisdiction),
        "legal_domain": doc.legal_domain.value if hasattr(doc.legal_domain, "value") else str(doc.legal_domain),
        "verification_status": doc.verification_status.value if hasattr(doc.verification_status, "value") else str(doc.verification_status),
        "metadata": chunk_meta.to_flat_dict(),
    }


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _read_csv(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Bureaucracy AI — Legal Knowledge Ingestion Pipeline")
    logger.info("=" * 60)

    # Initialise database (creates tables if missing)
    init_db()

    processor = DocumentProcessor()
    chunker = DocumentChunker()
    embedder = EmbeddingService()
    store = FAISSVectorStore(dimension=384)
    source_repo = LegalSourceRepository()
    chunk_repo = LegalChunkRepository()

    all_texts: list[str] = []
    all_metadata: list[dict] = []

    # -----------------------------------------------------------------------
    # 1. PDF — Karnataka Knowledge Base
    # -----------------------------------------------------------------------
    pdf_path = RAW_DIR / "Bureaucracy_AI_Karnataka_Knowledge_Base_v0.1.pdf"

    if pdf_path.exists():
        doc_meta = DocumentMetadata(
            document_id="KAR-KNOWLEDGE-001",
            title="Karnataka Knowledge Base",
            department="Government of Karnataka",
            source_url="https://karnataka.gov.in",
            document_type=DocumentType.KNOWLEDGE_BASE,
            jurisdiction=Jurisdiction.KARNATAKA,
            state="Karnataka",
            verification_status=VerificationStatus.VERIFIED_OFFICIAL,
            retrieved_at=_now_iso(),
        )

        # Register in SQLite
        source_repo.upsert({
            "source_id": doc_meta.document_id,
            "title": doc_meta.title,
            "document_type": doc_meta.document_type.value,
            "jurisdiction": doc_meta.jurisdiction.value,
            "state": doc_meta.state,
            "department": doc_meta.department,
            "source_url": doc_meta.source_url,
            "verification_status": doc_meta.verification_status.value,
            "retrieved_at": doc_meta.retrieved_at,
            "file_path": str(pdf_path),
        })

        records = processor.process_pdf(str(pdf_path), doc_meta)
        total = len(records)
        logger.info(f"PDF → {total} chunks from {pdf_path.name}")

        for idx, record in enumerate(records):
            page_no = record["metadata"].page_number or 1
            chunk_text = record["text"]

            # Globally unique chunk ID — page + position within page
            # We track a per-page counter to avoid duplicates
            chunk_id = _make_chunk_id(doc_meta.document_id, page_no, idx)

            chunk_meta = ChunkMetadata(
                document=doc_meta,
                page_number=page_no,
                chunk_index=idx,
                total_chunks=total,
                section_reference=None,
            )

            flat = _make_flat_metadata(chunk_id, chunk_text, pdf_path.name, "pdf", chunk_meta)
            all_texts.append(chunk_text)
            all_metadata.append(flat)

            # Write to SQLite
            chunk_repo.upsert({
                "chunk_id": chunk_id,
                "source_id": doc_meta.document_id,
                "chunk_text": chunk_text,
                "page_number": page_no,
                "chunk_index": idx,
                "total_chunks": total,
            })
    else:
        logger.warning(f"PDF not found: {pdf_path}")

    # -----------------------------------------------------------------------
    # 2. JSON — Karnataka Government Service Rules
    # -----------------------------------------------------------------------
    json_path = RAW_DIR / "Karnataka_Government_Service_Rules_v0.1.json"

    if json_path.exists():
        data = _read_json(json_path)
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
        chunks = chunker.split_text(json_text)
        logger.info(f"JSON → {len(chunks)} chunks from {json_path.name}")

        json_source_id = "KAR-SERVICE-RULES-001"
        source_repo.upsert({
            "source_id": json_source_id,
            "title": "Karnataka Government Service Rules",
            "document_type": "state_rule",
            "jurisdiction": "Karnataka",
            "state": "Karnataka",
            "department": "Government of Karnataka",
            "source_url": "",
            "verification_status": "unverified",
            "file_path": str(json_path),
        })

        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk_id = _make_chunk_id(json_source_id, 1, idx)
            flat = {
                "chunk_id": chunk_id,
                "text": chunk,
                "source": json_path.name,
                "type": "json",
                "title": "Karnataka Government Service Rules",
                "department": "Government of Karnataka",
                "source_url": "",
                "section_reference": None,
                "chapter_reference": None,
                "page_number": None,
                "jurisdiction": "Karnataka",
                "legal_domain": "government_service",
                "verification_status": "unverified",
                "metadata": {
                    "document": {"document_id": json_source_id, "title": "Karnataka Government Service Rules",
                                 "department": "Government of Karnataka", "source_url": ""},
                    "chunk_index": idx,
                    "total_chunks": total,
                },
            }
            all_texts.append(chunk)
            all_metadata.append(flat)
            chunk_repo.upsert({
                "chunk_id": chunk_id,
                "source_id": json_source_id,
                "chunk_text": chunk,
                "chunk_index": idx,
                "total_chunks": total,
            })
    else:
        logger.warning(f"JSON not found: {json_path}")

    # -----------------------------------------------------------------------
    # 3. CSV — Karnataka Service Registry (ksr.csv)
    # -----------------------------------------------------------------------
    csv_path = RAW_DIR / "ksr.csv"

    if csv_path.exists():
        rows = _read_csv(csv_path)
        total_csv_chunks = 0

        for row in rows:
            source_id = row.get("source_id", "").strip()
            source_name = row.get("source_name", "").strip()
            department = row.get("department", "").strip()
            url = row.get("url", "").strip()
            jurisdiction = row.get("jurisdiction", "State").strip()
            notes = row.get("notes", "").strip()

            if not source_id or not source_name:
                continue

            # Register in SQLite
            source_repo.upsert({
                "source_id": source_id,
                "title": source_name,
                "document_type": "portal_info",
                "jurisdiction": jurisdiction,
                "department": department,
                "source_url": url,
                "verification_status": "verified_official",
            })

            # Build a text chunk from the row
            row_lines = [
                f"Official Portal: {source_name}",
                f"Department: {department}",
                f"Portal URL: {url}",
                f"Jurisdiction: {jurisdiction}",
            ]
            if notes:
                row_lines.append(f"Description & Procedures: {notes}")

            row_text = "\n".join(row_lines)
            sub_chunks = chunker.split_text(row_text)

            for sub_idx, chunk in enumerate(sub_chunks):
                chunk_id = _make_chunk_id(source_id, 1, sub_idx)
                flat = {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "source": csv_path.name,
                    "type": "csv",
                    "title": source_name,
                    "department": department,
                    "source_url": url,
                    "section_reference": None,
                    "chapter_reference": None,
                    "page_number": None,
                    "jurisdiction": jurisdiction,
                    "legal_domain": "government_service",
                    "verification_status": "verified_official",
                    "metadata": {
                        "document": {"document_id": source_id, "title": source_name,
                                     "department": department, "source_url": url},
                        "chunk_index": sub_idx,
                        "total_chunks": len(sub_chunks),
                    },
                }
                all_texts.append(chunk)
                all_metadata.append(flat)
                chunk_repo.upsert({
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "chunk_text": chunk,
                    "chunk_index": sub_idx,
                    "total_chunks": len(sub_chunks),
                })
                total_csv_chunks += 1

        logger.info(f"CSV → {total_csv_chunks} chunks from {csv_path.name}")
    else:
        logger.warning(f"CSV not found: {csv_path}")

    # -----------------------------------------------------------------------
    # 4. Generate embeddings and build FAISS index
    # -----------------------------------------------------------------------
    if not all_texts:
        logger.error("No text found in data/raw/. Aborting.")
        return

    logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
    embeddings = embedder.embed_documents(all_texts)

    store.add(embeddings, all_metadata)
    store.save(str(OUTPUT_DIR))

    logger.info("=" * 60)
    logger.info(f"Ingestion complete — {len(all_texts)} chunks indexed")
    logger.info(f"FAISS index: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
