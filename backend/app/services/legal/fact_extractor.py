"""
Fact extractor — extracts structured facts from citizen queries
and uploaded legal documents.

Maintains a strict separation between:
  A. Citizen-provided facts (from conversation)
  B. Document-supported facts (from uploaded document)
  C. Retrieved legal information (from RAG)
  D. AI inferences (generated)
  E. Missing/unverified information
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FactSource(str, Enum):
    CITIZEN_STATED = "citizen_stated"
    DOCUMENT_EXTRACTED = "document_extracted"
    RETRIEVED_LEGAL = "retrieved_legal"
    AI_INFERRED = "ai_inferred"
    MISSING = "missing"
    UNVERIFIED = "unverified"


@dataclass
class ExtractedFact:
    fact_type: str
    value: str
    source: FactSource
    confidence: float = 1.0
    page_number: Optional[int] = None
    document_id: Optional[str] = None


@dataclass
class FactCollection:
    citizen_facts: List[ExtractedFact] = field(default_factory=list)
    document_facts: List[ExtractedFact] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)

    def add_citizen_fact(self, fact_type: str, value: str, confidence: float = 1.0):
        self.citizen_facts.append(ExtractedFact(
            fact_type=fact_type,
            value=value,
            source=FactSource.CITIZEN_STATED,
            confidence=confidence,
        ))

    def add_document_fact(self, fact_type: str, value: str, page: Optional[int] = None):
        self.document_facts.append(ExtractedFact(
            fact_type=fact_type,
            value=value,
            source=FactSource.DOCUMENT_EXTRACTED,
            page_number=page,
        ))

    def to_summary_dict(self) -> dict:
        return {
            "citizen_facts": [
                {"type": f.fact_type, "value": f.value, "source": f.source.value}
                for f in self.citizen_facts
            ],
            "document_facts": [
                {"type": f.fact_type, "value": f.value,
                 "source": f.source.value, "page": f.page_number}
                for f in self.document_facts
            ],
            "missing_info": self.missing_info,
        }


class FactExtractor:
    """
    Extracts structured facts from citizen messages and uploaded documents.
    """

    def extract_from_query(self, query: str, domain: str = "unknown") -> FactCollection:
        """Extract facts from a citizen's natural language query."""
        facts = FactCollection()
        q = query.strip()

        # --- Dates ---
        for date_match in re.finditer(
            r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{4}|"
            r"(?:last|past)\s+\d+\s+(?:days|weeks|months)",
            q, re.IGNORECASE,
        ):
            facts.add_citizen_fact("date_mentioned", date_match.group(), confidence=0.9)

        # --- Monetary amounts ---
        for amount_match in re.finditer(
            r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{2})?|[\d,]+\s*(?:rupees|lakh|crore)",
            q, re.IGNORECASE,
        ):
            facts.add_citizen_fact("amount", amount_match.group(), confidence=0.95)

        # --- Parties ---
        # Look for "my employer", "the landlord", "the builder", etc.
        for party_match in re.finditer(
            r"(?:my|the)\s+(employer|landlord|builder|contractor|company|bank|seller|vendor|"
            r"officer|authority|department|court|police|doctor|school|hospital)",
            q, re.IGNORECASE,
        ):
            facts.add_citizen_fact("party_mentioned", party_match.group(), confidence=0.8)

        # --- Locations ---
        for loc_match in re.finditer(
            r"(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
            r"(?:\s+(?:district|taluk|village|ward|city|town))?",
            q,
        ):
            loc = loc_match.group(1)
            if len(loc) > 2:
                facts.add_citizen_fact("location", loc, confidence=0.7)

        return facts

    def extract_from_document(self, text: str, document_type: str = "unknown") -> FactCollection:
        """Extract structured facts from a legal document's text."""
        facts = FactCollection()
        if not text or not text.strip():
            return facts

        # --- Parties ---
        for m in re.finditer(
            r"(?:Between|BETWEEN|Lessor|Lessee|Complainant|Respondent|Petitioner|Defendant|Plaintiff)"
            r"[:\s]+([A-Z][a-zA-Z\s\.]+?)(?:,|\n|and\b)",
            text,
        ):
            facts.add_document_fact("party", m.group(1).strip()[:100])

        # --- Dates ---
        for m in re.finditer(
            r"(?:dated|date|on)\s+(\d{1,2}[-/th|st|nd|rd\s]+(?:January|February|March|April|May|"
            r"June|July|August|September|October|November|December),?\s+\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            text, re.IGNORECASE,
        ):
            facts.add_document_fact("date", m.group(1).strip())

        # --- Monetary amounts ---
        for m in re.finditer(
            r"(?:Rs\.?|INR|₹|amount\s+of\s+Rs\.?)\s*[\d,]+(?:\.\d{2})?",
            text, re.IGNORECASE,
        ):
            facts.add_document_fact("amount", m.group().strip())

        # --- Section references ---
        for m in re.finditer(
            r"[Ss]ection\s+\d+[A-Za-z]?(?:\(\d+\))?|[Aa]rticle\s+\d+|[Rr]ule\s+\d+",
            text,
        ):
            facts.add_document_fact("section_reference", m.group().strip())

        # --- Authority/Issued by ---
        for m in re.finditer(
            r"(?:Issued\s+by|Signed\s+by|Authority|Commissioner|Tahsildar|Collector)"
            r"[:\s]+([A-Z][a-zA-Z\s\.]+?)(?:,|\n)",
            text,
        ):
            facts.add_document_fact("issuing_authority", m.group(1).strip()[:100])

        return facts
