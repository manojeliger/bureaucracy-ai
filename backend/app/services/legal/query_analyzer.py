"""
Legal query analyzer — classifies the legal domain and extracts
key facts from a citizen's natural-language query.

This is the entry point to the Legal Investigation Engine (Module 5).
It runs BEFORE retrieval so the retrieval service can apply domain filters.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

# Keyword patterns mapped to legal domains
_DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "employment_salary": [
        r"salary|wages|payslip|pay\s+slip|employer|employee|termination|fired|dismissed"
        r"|pf|provident\s+fund|esi|gratuity|notice\s+period|appraisal|bonus|overtime"
        r"|labour|labor|work(place|er|ing\s+hours)?|maternity|sexual\s+harassment"
    ],
    "rental_property": [
        r"landlord|tenant|rent|deposit|eviction|lease|tenancy|vacate|notice\s+to\s+quit"
        r"|rental\s+agreement|property\s+damage|maintenance|water\s+supply|electricity"
    ],
    "consumer_dispute": [
        r"consumer|product|refund|defective|warranty|guarantee|e-commerce|amazon|flipkart"
        r"|service\s+provider|billing\s+error|overcharge|misleading\s+advertisement"
        r"|consumer\s+forum|consumer\s+court|deficiency\s+of\s+service"
    ],
    "land_revenue": [
        r"land\s+record|mutation|pahani|bhoomi|khata|bbmp|rto|property\s+tax|patta"
        r"|encroachment|survey|revenue|tahsildar|village\s+accountant|panchayat"
    ],
    "government_service": [
        r"certificate|sakala|seva\s+sindhu|nadakacheri|ration\s+card|aadhaar|pan\s+card"
        r"|scholarship|ssp|pension|government\s+portal|application\s+rejected|delayed"
        r"|bhoomi|ahara|dbh|grievance|janaspandana"
    ],
    "family_civil": [
        r"divorce|marriage|maintenance|alimony|child\s+custody|property\s+dispute"
        r"|inheritance|will|succession|domestic\s+violence|dowry|marriage\s+registration"
    ],
    "criminal": [
        r"\bfir\b|police|arrest|bail|complaint|ipc|section\s+\d+|cognizable|non-cognizable"
        r"|anticipatory\s+bail|chargesheet|prosecution"
    ],
    "taxation": [
        r"income\s+tax|gst|tds|tax\s+notice|tax\s+demand|itr|assessment|penalty\s+tax"
        r"|tax\s+return|tax\s+exemption|80c|deduction"
    ],
    "pension_welfare": [
        r"pension|social\s+security|widow|disability|old\s+age|welfare\s+scheme"
        r"|anganwadi|asha|mgnrega|pm\s+kisan|dbh|beneficiary"
    ],
}


def classify_legal_domain(query: str) -> Tuple[str, float]:
    """
    Classify the most likely legal domain of a citizen query.

    Returns:
        (domain_name, confidence_score)  — confidence is 0.0–1.0 (heuristic)
    """
    query_lower = query.lower()
    scores: Dict[str, int] = {}

    for domain, patterns in _DOMAIN_PATTERNS.items():
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            count += len(matches)
        if count:
            scores[domain] = count

    if not scores:
        return "general_rights", 0.0

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = min(scores[best] / max(total, 1), 1.0)
    return best, round(confidence, 2)


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named legal entities from query text using regex patterns.
    
    Returns a dict of entity types → list of extracted values.
    """
    entities: Dict[str, List[str]] = {
        "acts": [],
        "sections": [],
        "dates": [],
        "amounts": [],
        "authorities": [],
        "portals": [],
        "locations": [],
    }

    # Acts (e.g. "Payment of Wages Act, 1936", "IPC", "CPC")
    act_matches = re.findall(
        r"(?:[A-Z][a-z]+\s+)+(?:Act|Code|Rules?|Order|Regulation),?\s*\d{4}|"
        r"\bIPC\b|\bCPC\b|\bCrPC\b|\bIT\s+Act\b|\bPCSA\b|\bMVA\b",
        text,
    )
    entities["acts"] = list(set(act_matches))

    # Sections
    section_matches = re.findall(
        r"[Ss]ection\s+\d+[A-Za-z]?(?:\(\d+\))?|"
        r"[Ss]ec\.?\s*\d+[A-Za-z]?|"
        r"[Aa]rticle\s+\d+",
        text,
    )
    entities["sections"] = list(set(section_matches))

    # Dates
    date_matches = re.findall(
        r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|"
        r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}",
        text,
        re.IGNORECASE,
    )
    entities["dates"] = list(set(date_matches))

    # Monetary amounts
    amount_matches = re.findall(
        r"(?:Rs\.?|INR|₹)\s*[\d,]+(?:\.\d{2})?|"
        r"[\d,]+\s*(?:rupees|lakh|lakhs|crore|crores)",
        text,
        re.IGNORECASE,
    )
    entities["amounts"] = list(set(amount_matches))

    # Karnataka-specific portals
    portal_matches = re.findall(
        r"Seva\s+Sindhu|Nadakacheri|Bhoomi|Ahara|Sakala|SSP|Janaspandana|"
        r"eCourts|BBMP|UIDAI|DigiLocker",
        text,
        re.IGNORECASE,
    )
    entities["portals"] = list(set(portal_matches))

    return entities


# ---------------------------------------------------------------------------
# Query analyzer — combines domain + entity extraction
# ---------------------------------------------------------------------------

class QueryAnalyzer:
    """
    Analyzes a citizen's query to identify legal domain, entities,
    and missing information before retrieval begins.
    """

    def analyze(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Full query analysis.

        Returns:
            {
              domain: str,
              domain_confidence: float,
              entities: dict,
              is_document_query: bool,
              has_specific_section: bool,
              missing_info: list[str],
              search_terms: str,  # Optimized search query for retrieval
            }
        """
        # Augment with recent history context if available
        full_context = query
        if conversation_history:
            recent_user = [
                m["content"] for m in conversation_history[-4:]
                if m.get("role") == "user"
            ]
            if recent_user:
                full_context = " ".join(recent_user) + " " + query

        domain, confidence = classify_legal_domain(full_context)
        entities = extract_named_entities(full_context)

        # Detect if this is a document-focused query
        is_document_query = bool(re.search(
            r"document|this\s+notice|the\s+letter|this\s+agreement|uploaded|attached",
            query, re.IGNORECASE,
        ))

        # Detect explicit section reference
        has_specific_section = bool(entities["sections"] or entities["acts"])

        # Build optimized search terms for retrieval
        # Prioritize explicit legal terms over filler words
        search_terms = self._build_search_terms(query, domain, entities)

        # Identify missing information
        missing_info = self._identify_missing_info(query, domain, entities)

        return {
            "domain": domain,
            "domain_confidence": confidence,
            "entities": entities,
            "is_document_query": is_document_query,
            "has_specific_section": has_specific_section,
            "missing_info": missing_info,
            "search_terms": search_terms,
        }

    @staticmethod
    def _build_search_terms(
        query: str,
        domain: str,
        entities: Dict[str, List[str]],
    ) -> str:
        """
        Build an optimized search query for the retrieval layer.
        
        Combines the original query with extracted legal entities
        so that section-specific queries still hit the vector index accurately.
        """
        parts = [query]

        # Prepend explicit legal references for better retrieval
        for act in entities.get("acts", [])[:2]:
            parts.insert(0, act)
        for section in entities.get("sections", [])[:2]:
            parts.insert(0, section)

        # Add domain-specific keywords to boost recall
        domain_keywords = {
            "employment_salary": "salary wages employer rights Karnataka",
            "rental_property": "landlord tenant deposit rent Karnataka",
            "consumer_dispute": "consumer rights refund complaint Karnataka",
            "land_revenue": "land record mutation property Karnataka",
            "government_service": "government service certificate Karnataka Sakala",
            "criminal": "FIR police complaint Karnataka IPC",
            "taxation": "tax notice income tax Karnataka",
        }
        if domain in domain_keywords:
            parts.append(domain_keywords[domain])

        return " ".join(parts)[:500]  # Keep query within reasonable length

    @staticmethod
    def _identify_missing_info(
        query: str,
        domain: str,
        entities: Dict[str, List[str]],
    ) -> List[str]:
        """
        Identify what information is missing for a complete legal analysis.
        """
        missing = []
        q_lower = query.lower()

        # Domain-specific missing info checks
        if domain == "employment_salary":
            if not any(w in q_lower for w in ["month", "week", "days", "period"]):
                missing.append("Duration or period of the salary dispute")
            if not any(w in q_lower for w in ["written", "email", "notice", "contacted"]):
                missing.append("Whether you have communicated the issue in writing to the employer")

        elif domain == "rental_property":
            if not any(w in q_lower for w in ["agreement", "lease", "contract"]):
                missing.append("Whether a written rental agreement exists")
            if not entities["amounts"]:
                missing.append("The deposit amount involved")

        elif domain == "consumer_dispute":
            if not entities["amounts"]:
                missing.append("The amount or value of the product/service in dispute")
            if not any(w in q_lower for w in ["receipt", "invoice", "bill", "proof"]):
                missing.append("Whether you have purchase proof or receipt")

        elif domain == "government_service":
            if not any(w in q_lower for w in ["reference", "application", "number", "id"]):
                missing.append("Your application reference number or service request ID")
            if not any(w in q_lower for w in ["reject", "pending", "status", "delay"]):
                missing.append("Current status of your application")

        return missing[:3]  # Return at most 3 missing items
