"""
RAG Service — grounded answer generation for Bureaucracy AI.

This service orchestrates:
  1. Hybrid retrieval (via RetrievalService)
  2. Context formatting
  3. LLM prompt construction and generation
  4. Citation extraction and validation
  5. Conversation history management

It is the single entry point called by the API layer for chat responses.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.services.llm.ollama_client import OllamaClient
from backend.app.services.rag.citation_service import CitationService
from backend.app.services.rag.embedding_service import EmbeddingService
from backend.app.services.rag.keyword_search import KeywordSearchService
from backend.app.services.rag.retrieval_service import RetrievalService, RetrievedChunk
from backend.app.services.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Use locally cached model — no internet calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class RAGService:
    """
    Singleton RAG service.
    Use RAGService.get_instance() — do not construct directly in API handlers.
    """

    _instance: Optional["RAGService"] = None

    def __init__(self):
        self._embedding_service = EmbeddingService(model_name=settings.EMBEDDING_MODEL_NAME)
        self._vector_store = FAISSVectorStore(dimension=settings.EMBEDDING_DIMENSION)
        self._llm_client = OllamaClient()
        self._citation_service = CitationService()
        self._keyword_service = KeywordSearchService()

        # Build retrieval service from components
        self._retrieval_service = RetrievalService(
            embedding_service=self._embedding_service,
            vector_store=self._vector_store,
            keyword_search=self._keyword_service,
        )

        self.is_index_loaded = False
        self._load_vector_store()

    @classmethod
    def get_instance(cls) -> "RAGService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Properties (for health check compatibility)
    # ------------------------------------------------------------------

    @property
    def vector_store(self) -> FAISSVectorStore:
        return self._vector_store

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_vector_store(self) -> None:
        from pathlib import Path
        index_dir = Path(settings.FAISS_INDEX_DIR)
        if (index_dir / "knowledge.index").exists() and (index_dir / "metadata.json").exists():
            try:
                self._vector_store.load(str(index_dir))
                self.is_index_loaded = True
                logger.info(f"FAISS index loaded — {self._vector_store.total_vectors} vectors")
            except Exception as exc:
                logger.error(f"Failed to load FAISS index: {exc}")
                self.is_index_loaded = False
        else:
            logger.warning(f"FAISS index not found at {index_dir} — run ingestion first")
            self.is_index_loaded = False

    def reload_index(self) -> None:
        """Reload the FAISS index from disk (call after re-ingestion)."""
        self._vector_store = FAISSVectorStore(dimension=settings.EMBEDDING_DIMENSION)
        self._retrieval_service._vector_store = self._vector_store
        self._load_vector_store()

    # ------------------------------------------------------------------
    # Retrieval (public, for direct use in tests or admin tools)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        jurisdiction: Optional[str] = None,
        legal_domain: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        if not self.is_index_loaded:
            self._load_vector_store()
        return self._retrieval_service.retrieve(
            query,
            top_k=top_k,
            threshold=threshold,
            jurisdiction=jurisdiction,
            legal_domain=legal_domain,
        )

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into an LLM-readable context block."""
        if not chunks:
            return "No relevant official legal information found in the knowledge base."

        lines = []
        for i, chunk in enumerate(chunks, start=1):
            source_label = chunk.title or "Legal Source"
            if chunk.source_url:
                source_label += f" ({chunk.source_url})"
            if chunk.section_reference:
                source_label += f" — {chunk.section_reference}"
            if chunk.page_number:
                source_label += f" [p.{chunk.page_number}]"

            score_pct = int(chunk.score * 100)
            lines.append(
                f"[Source {i} | Relevance: {score_pct}% | {source_label}]\n{chunk.text}"
            )

        # Truncate to stay within LLM context window
        full = "\n\n".join(lines)
        if len(full) > settings.RAG_MAX_CONTEXT_CHARS:
            full = full[: settings.RAG_MAX_CONTEXT_CHARS] + "\n...[context truncated]"
        return full

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        language: str = "English",
        state: str = "Karnataka",
        document_text: Optional[str] = None,
        legal_domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
          1. Retrieve relevant legal passages.
          2. Format context.
          3. Build grounded prompt.
          4. Generate LLM response.
          5. Validate and return citations.
        """
        # --- Retrieval ---
        retrieved_chunks = self.retrieve(
            query,
            jurisdiction=state,
            legal_domain=legal_domain,
        )
        context_str = self.format_context(retrieved_chunks)
        has_context = len(retrieved_chunks) > 0

        # --- Conversation history ---
        history_str = self._format_history(conversation_history)

        # --- Uploaded document context ---
        doc_context = ""
        if document_text and document_text.strip():
            doc_context = (
                f"CITIZEN'S UPLOADED DOCUMENT:\n"
                f"{'─' * 50}\n"
                f"{document_text.strip()[:3000]}\n"
                f"{'─' * 50}\n"
                f"[NOTE: This text was extracted from the citizen's uploaded document. "
                f"Treat it as citizen-provided information, not a verified legal source.]\n"
            )

        # --- Prompt ---
        prompt = self._build_prompt(
            query=query,
            context_str=context_str,
            history_str=history_str,
            doc_context=doc_context,
            language=language,
            state=state,
            has_context=has_context,
        )

        # --- LLM generation ---
        try:
            llm_response = self._llm_client.generate(prompt)
        except Exception as exc:
            logger.error(f"LLM generation failed: {exc}")
            llm_response = (
                "I could not connect to the AI reasoning service. "
                "Please ensure Ollama is running with 'ollama serve' and the "
                f"'qwen2.5:3b' model is available. (Error: {exc})"
            )

        # --- Citations ---
        citations = self._citation_service.build_citations(retrieved_chunks)
        display_sources = self._citation_service.format_for_display(citations)

        # --- Follow-up question extraction ---
        follow_up = self._extract_follow_up(llm_response)

        return {
            "answer": llm_response,
            "sources": display_sources,
            "citations": citations,
            "retrieved_chunks": [c.to_dict() for c in retrieved_chunks],
            "has_context": has_context,
            "follow_up_question": follow_up,
            "legal_domain": legal_domain,
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        query: str,
        context_str: str,
        history_str: str,
        doc_context: str,
        language: str,
        state: str,
        has_context: bool,
    ) -> str:
        newline = "\n"
        no_context_note = (
            newline + "WARNING: No matching official sources were found in the knowledge base "
            "for this query. Acknowledge this clearly. Do not fabricate legal provisions, "
            "section numbers, court decisions, deadlines, or government portal names." + newline
            if not has_context else ""
        )
        no_doc = "No document uploaded."
        doc_section = doc_context if doc_context else no_doc

        return f"""You are Bureaucracy AI — an intelligent, empathetic legal information and citizen rights assistant for {state}, India.

You help ordinary citizens understand their legal situation, their rights, and practical next steps — in plain language, without legal jargon.

You are NOT a lawyer. You provide legal INFORMATION, not legal ADVICE or formal legal representation.
Every substantive claim must be grounded in the retrieved knowledge below.

━━━ OFFICIAL KNOWLEDGE BASE (PRIMARY SOURCE) ━━━
{context_str}
{no_context_note}
━━━ CITIZEN'S UPLOADED DOCUMENT ━━━
{doc_section}
━━━ CONVERSATION HISTORY ━━━
{history_str}
━━━ CITIZEN'S CURRENT MESSAGE ━━━
"{query}"

━━━ YOUR RESPONSE RULES ━━━
1. GROUNDING: Answer only from the official knowledge above. If information is absent, say so explicitly.
2. NO FABRICATION: Never invent section numbers, Act names, court orders, deadlines, fees, addresses, or portal names.
3. CLARITY: Use plain language. Translate legal terms (e.g. "mutation" → "changing the land record owner name").
4. STRUCTURE: When analysing a legal problem, organise your answer into these sections:
   - **Understanding your situation** (1–2 sentences)
   - **Relevant legal information** (what the law or procedure says, cite sources)
   - **Your potential rights or options** (practical, grounded)
   - **Recommended next steps** (concrete, specific)
   - **What I still need to know** (1 focused question if information is missing, else skip)
5. SOURCES: At the end, list the sources you used. Do not cite sources not in the knowledge base above.
6. UNCERTAINTY: Use "potentially", "may", "based on available information" when you are not certain.
7. DISCLAIMER: End with a one-sentence reminder that this is legal information, not formal legal advice.
8. LANGUAGE: Respond entirely in {language}.

Respond now:"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return "No prior conversation."
        lines = []
        for turn in history[-8:]:  # Keep last 8 turns
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "").strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) if lines else "No prior conversation."

    @staticmethod
    def _extract_follow_up(response_text: str) -> Optional[str]:
        """
        Extract the follow-up question the AI embedded in its response.
        Looks for the structured 'What I still need to know' section.
        """
        patterns = [
            r"\*\*What I still need to know\*\*[:\s]*(.*?)(?=\n\*\*|\Z)",
            r"Important Follow-Up Question[:\s]*(.*?)(?=\n###|\Z)",
            r"### Follow-Up[:\s]*(.*?)(?=\n###|\Z)",
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                q = match.group(1).strip()
                if q and len(q) > 10 and "?" in q:
                    return q.split("?")[0].strip() + "?"
        return None
