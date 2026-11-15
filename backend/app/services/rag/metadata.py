"""
Legal metadata models for Bureaucracy AI.

Every ingested legal document and chunk carries structured metadata so that
retrieval can be filtered by jurisdiction, date range, verification status,
legal domain, and source type — and so that citations can be validated against
real source records.
"""

from datetime import date
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    """Tracks whether a document has been verified as an authoritative source."""
    VERIFIED_OFFICIAL = "verified_official"
    UNVERIFIED = "unverified"
    PENDING = "pending"
    SUPERSEDED = "superseded"


class DocumentType(str, Enum):
    """Broad classification of the legal document."""
    CENTRAL_ACT = "central_act"
    STATE_ACT = "state_act"
    CENTRAL_RULE = "central_rule"
    STATE_RULE = "state_rule"
    GOVERNMENT_NOTIFICATION = "government_notification"
    AMENDMENT = "amendment"
    COURT_JUDGMENT = "court_judgment"
    COURT_ORDER = "court_order"
    GOVERNMENT_PROCEDURE = "government_procedure"
    CITIZEN_SERVICE = "citizen_service"
    KNOWLEDGE_BASE = "knowledge_base"
    PORTAL_INFO = "portal_info"
    UNKNOWN = "unknown"


class LegalDomain(str, Enum):
    """Legal domain / subject-matter category."""
    CONSUMER_DISPUTE = "consumer_dispute"
    RENTAL_PROPERTY = "rental_property"
    EMPLOYMENT_SALARY = "employment_salary"
    GOVERNMENT_SERVICE = "government_service"
    FAMILY_CIVIL = "family_civil"
    CRIMINAL = "criminal"
    LAND_REVENUE = "land_revenue"
    TAXATION = "taxation"
    PENSION_WELFARE = "pension_welfare"
    GENERAL_RIGHTS = "general_rights"
    UNKNOWN = "unknown"


class Jurisdiction(str, Enum):
    """Geographical / administrative jurisdiction."""
    CENTRAL = "Central"
    KARNATAKA = "Karnataka"
    LOCAL = "Local"
    UNKNOWN = "Unknown"


# ---------------------------------------------------------------------------
# Document-level metadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """
    Metadata attached to every ingested legal document.

    Fields match the target schema defined in Section 4 of the project spec,
    plus the fields already in use by the existing ingestion pipeline.
    """

    # --- Identity ---
    document_id: str = Field(..., description="Stable unique identifier for this document")
    title: str = Field(..., description="Human-readable document title")

    # --- Classification ---
    document_type: DocumentType = Field(
        default=DocumentType.UNKNOWN,
        description="Type of legal document"
    )
    legal_domain: LegalDomain = Field(
        default=LegalDomain.UNKNOWN,
        description="Subject-matter domain of this document"
    )

    # --- Jurisdiction ---
    jurisdiction: Jurisdiction = Field(
        default=Jurisdiction.KARNATAKA,
        description="Primary jurisdiction"
    )
    state: str = Field(
        default="Karnataka",
        description="State name when jurisdiction is state-level"
    )

    # --- Authoring body ---
    department: str = Field(
        default="",
        description="Issuing government department or court"
    )
    court_name: Optional[str] = Field(
        default=None,
        description="Court name for judgments and orders"
    )
    act_name: Optional[str] = Field(
        default=None,
        description="Parent Act name when document is a rule or amendment"
    )

    # --- Case / section identifiers ---
    case_number: Optional[str] = Field(
        default=None,
        description="Court case number if applicable"
    )
    section_number: Optional[str] = Field(
        default=None,
        description="Section number when document covers a specific provision"
    )

    # --- Source ---
    source_url: str = Field(default="", description="Canonical URL of the official source")

    # --- Dates ---
    publication_date: Optional[date] = Field(
        default=None,
        description="Date the document was officially published"
    )
    commencement_date: Optional[date] = Field(
        default=None,
        description="Date the law or rule came into force"
    )
    effective_from: Optional[date] = Field(
        default=None,
        description="Start of the period during which this version is effective"
    )
    effective_until: Optional[date] = Field(
        default=None,
        description="End of the period (None = still current)"
    )
    retrieved_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 timestamp when this document was downloaded"
    )

    # --- Versioning ---
    version: str = Field(
        default="1.0",
        description="Document version string (e.g. '1.0', '2026-01')"
    )
    amendment_status: Optional[str] = Field(
        default=None,
        description="Description of pending or applied amendments"
    )

    # --- Quality ---
    language: str = Field(default="en", description="ISO 639-1 language code")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Whether the document has been verified as authoritative"
    )

    # Convenience method used by the ingestion pipeline
    def to_flat_dict(self) -> dict:
        """Return a flat dict suitable for embedding in chunk metadata."""
        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Chunk-level metadata
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """
    Metadata attached to every individual text chunk stored in FAISS.

    Each chunk must carry enough information to reconstruct a verifiable
    citation and to filter results by jurisdiction, domain, and date.
    """

    document: DocumentMetadata

    # --- Position ---
    page_number: Optional[int] = Field(
        default=None,
        description="Source page number within the original document"
    )
    chunk_index: int = Field(ge=0, description="Zero-based position of this chunk in the document")
    total_chunks: int = Field(gt=0, description="Total number of chunks in the document")

    # --- Legal structure ---
    section_reference: Optional[str] = Field(
        default=None,
        description="Section/rule/article label extracted from the text (e.g. 'Section 14')"
    )
    chapter_reference: Optional[str] = Field(
        default=None,
        description="Chapter label if the chunk belongs to a named chapter"
    )

    def to_flat_dict(self) -> dict:
        """
        Returns a flat dict for storage in FAISS metadata.json.
        The document sub-object is expanded under the 'metadata' key.
        """
        return {
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "section_reference": self.section_reference,
            "chapter_reference": self.chapter_reference,
            "document": self.document.to_flat_dict(),
        }
