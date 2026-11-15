"""
Legal investigation engine — connects citizen facts to retrieved
legal provisions and identifies possible rights and next steps.

This is the core intelligence module (Module 5 + 6 in the project spec).
"""

import logging
from typing import Any, Dict, List, Optional

from backend.app.services.legal.fact_extractor import ExtractedFact, FactCollection, FactExtractor
from backend.app.services.legal.query_analyzer import QueryAnalyzer
from backend.app.services.rag.retrieval_service import RetrievedChunk

logger = logging.getLogger(__name__)


class InvestigationEngine:
    """
    Connects citizen-provided facts with retrieved legal evidence to
    build a preliminary legal analysis.

    Maintains strict separation of information sources:
      - Citizen-stated facts
      - Document-extracted facts
      - Retrieved legal provisions
      - AI inferences (clearly labelled)
    """

    def __init__(self):
        self._analyzer = QueryAnalyzer()
        self._extractor = FactExtractor()

    def investigate(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        document_text: Optional[str] = None,
        document_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full legal investigation pipeline.

        Returns a structured analysis dict that the RAG service uses
        to build its LLM prompt context.
        """
        # --- Step 1: Analyze the query ---
        analysis = self._analyzer.analyze(query, conversation_history)
        domain = analysis["domain"]
        missing_from_query = analysis.get("missing_info", [])

        # --- Step 2: Extract citizen facts ---
        citizen_facts = self._extractor.extract_from_query(query, domain)

        # --- Step 3: Extract document facts ---
        doc_facts = FactCollection()
        if document_text and document_text.strip():
            doc_facts = self._extractor.extract_from_document(document_text, document_type or "unknown")

        # --- Step 4: Map facts to legal provisions ---
        legal_mapping = self._map_to_legal_provisions(
            citizen_facts, doc_facts, retrieved_chunks, domain
        )

        # --- Step 5: Identify contradictions or gaps ---
        contradictions = self._find_contradictions(citizen_facts, doc_facts)
        all_missing = list(set(missing_from_query + legal_mapping.get("missing_for_legal_analysis", [])))

        return {
            "domain": domain,
            "domain_confidence": analysis["domain_confidence"],
            "entities": analysis["entities"],
            "citizen_facts": citizen_facts.to_summary_dict()["citizen_facts"],
            "document_facts": doc_facts.to_summary_dict()["document_facts"],
            "applicable_provisions": legal_mapping["applicable_provisions"],
            "missing_information": all_missing[:5],
            "contradictions": contradictions,
            "is_document_query": analysis["is_document_query"],
            "has_specific_section": analysis["has_specific_section"],
            "search_terms": analysis["search_terms"],
        }

    @staticmethod
    def _map_to_legal_provisions(
        citizen_facts: FactCollection,
        doc_facts: FactCollection,
        retrieved_chunks: List[RetrievedChunk],
        domain: str,
    ) -> Dict[str, Any]:
        """
        Connect extracted facts with retrieved legal passages.
        Returns a list of applicable provisions with their relevance.
        """
        applicable = []
        missing_for_legal = []

        for chunk in retrieved_chunks[:5]:  # Top 5 chunks
            provision = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.title,
                "text_excerpt": (chunk.text or "")[:300],
                "section": chunk.section_reference,
                "relevance_score": chunk.score,
                "is_verified": chunk.verification_status == "verified_official",
                "jurisdiction": chunk.jurisdiction,
            }
            applicable.append(provision)

        # Domain-specific missing info for legal analysis
        domain_missing = {
            "employment_salary": [
                "Written employment contract or offer letter",
                "Payslips for the disputed period",
            ],
            "rental_property": [
                "Signed rental agreement",
                "Rent receipts",
                "Written communication about deposit",
            ],
            "consumer_dispute": [
                "Purchase invoice or receipt",
                "Written complaint to seller/service provider",
            ],
            "government_service": [
                "Application acknowledgement number",
                "Rejection letter or status screenshot",
            ],
        }
        missing_for_legal = domain_missing.get(domain, [])

        return {
            "applicable_provisions": applicable,
            "missing_for_legal_analysis": missing_for_legal,
        }

    @staticmethod
    def _find_contradictions(
        citizen_facts: FactCollection,
        doc_facts: FactCollection,
    ) -> List[str]:
        """
        Identify potential contradictions between citizen statements and document facts.
        Simple heuristic: compare amounts mentioned.
        """
        contradictions = []

        citizen_amounts = {f.value for f in citizen_facts.citizen_facts if f.fact_type == "amount"}
        doc_amounts = {f.value for f in doc_facts.document_facts if f.fact_type == "amount"}

        if citizen_amounts and doc_amounts and citizen_amounts != doc_amounts:
            contradictions.append(
                f"Amount mentioned in conversation ({', '.join(citizen_amounts)}) "
                f"differs from amount in document ({', '.join(doc_amounts)}). "
                f"Please clarify which is correct."
            )

        return contradictions


class RemedyEngine:
    """
    Identifies possible rights, remedies, and next steps
    based on the legal domain and retrieved evidence.
    """

    _DOMAIN_REMEDIES: Dict[str, Dict[str, Any]] = {
        "employment_salary": {
            "rights": [
                "Right to receive wages on time under the Payment of Wages Act, 1936",
                "Right to file a complaint with the Labour Commissioner",
                "Right to approach the Labour Court for unpaid wages",
            ],
            "authorities": [
                "Labour Commissioner, Karnataka",
                "Labour Court",
                "Employees' Provident Fund Organisation (EPFO) for PF disputes",
            ],
            "procedures": [
                "Send written demand notice to employer",
                "File complaint with District Labour Officer",
                "Approach Labour Court if no resolution within 30 days",
            ],
        },
        "rental_property": {
            "rights": [
                "Right to receive security deposit back after vacating",
                "Right against wrongful eviction without due notice",
                "Right to file complaint under Karnataka Rent Control Act",
            ],
            "authorities": [
                "Rent Control Court, Karnataka",
                "Civil Court (for deposit disputes above ₹3 lakh)",
                "Police / Magistrate (for wrongful eviction)",
            ],
            "procedures": [
                "Send written demand notice to landlord",
                "File complaint with Rent Control Court",
                "Approach civil court for recovery of deposit",
            ],
        },
        "consumer_dispute": {
            "rights": [
                "Right to seek refund, replacement, or compensation",
                "Right to file complaint under Consumer Protection Act, 2019",
                "Right to approach Consumer Disputes Redressal Commission",
            ],
            "authorities": [
                "District Consumer Disputes Redressal Commission (up to ₹1 crore)",
                "State Consumer Disputes Redressal Commission (₹1–10 crore)",
                "National Consumer Disputes Redressal Commission (above ₹10 crore)",
            ],
            "procedures": [
                "Send written complaint to seller/service provider first",
                "If no resolution in 30 days, file at the appropriate Consumer Commission",
                "File online at edaakhil.nic.in",
            ],
        },
        "government_service": {
            "rights": [
                "Right to receive guaranteed service within Sakala timelines",
                "Right to appeal rejection with the designated authority",
                "Right to file grievance on Janaspandana portal",
            ],
            "authorities": [
                "Sakala Mission Office, DPAR",
                "Janaspandana Grievance Portal (ipgrs.karnataka.gov.in)",
                "Karnataka Lokayukta (for corruption-related delays)",
            ],
            "procedures": [
                "Check Sakala timeline for your service at sakala.kar.nic.in",
                "File online grievance at Janaspandana portal",
                "Escalate to HOD if Sakala deadline is breached",
            ],
        },
    }

    def get_remedies(
        self,
        domain: str,
        retrieved_chunks: List[RetrievedChunk],
        missing_info: List[str],
    ) -> Dict[str, Any]:
        """
        Return domain-appropriate rights, remedies, and next steps.
        """
        remedies = self._DOMAIN_REMEDIES.get(domain, {
            "rights": ["Please describe your problem in more detail for a targeted analysis."],
            "authorities": ["Relevant government authority depends on your specific situation."],
            "procedures": ["Provide more details about your situation for specific guidance."],
        })

        return {
            "domain": domain,
            "potential_rights": remedies.get("rights", []),
            "relevant_authorities": remedies.get("authorities", []),
            "suggested_procedures": remedies.get("procedures", []),
            "supporting_sources": [
                {"chunk_id": c.chunk_id, "title": c.title, "url": c.source_url}
                for c in retrieved_chunks[:3]
                if c.source_url
            ],
            "missing_information": missing_info,
            "disclaimer": (
                "The above is legal information based on available knowledge. "
                "It is not formal legal advice. For consequential legal decisions, "
                "please consult a qualified advocate."
            ),
        }
