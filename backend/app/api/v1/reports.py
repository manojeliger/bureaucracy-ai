"""
Report generation API — structured legal assistance reports.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.schemas.legal import ReportRequest, ReportResponse
from backend.app.services.database.repositories import (
    ConversationRepository,
    MessageRepository,
    ReportRepository,
    CitationRepository,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_conv_repo = ConversationRepository()
_msg_repo = MessageRepository()
_report_repo = ReportRepository()
_cit_repo = CitationRepository()

REPORTS_DIR = Path(settings.UPLOADS_DIR).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/reports/generate", response_model=ReportResponse)
def generate_report(request: ReportRequest):
    """
    Generate a structured legal assistance report from a conversation.
    """
    conv = _conv_repo.get(request.conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = _msg_repo.list_for_conversation(request.conversation_id)
    if not messages:
        raise HTTPException(status_code=400, detail="Conversation has no messages yet")

    title = request.title or f"Legal Assistance Report — {conv['title']}"

    # Try PDF generation
    try:
        file_path = _generate_pdf_report(title, conv, messages)
        status = "completed"
    except Exception as exc:
        logger.warning(f"PDF generation failed: {exc}. Falling back to text report.")
        try:
            file_path = _generate_text_report(title, conv, messages)
            status = "completed"
        except Exception as exc2:
            logger.error(f"Text report also failed: {exc2}")
            file_path = None
            status = "failed"

    report_id = _report_repo.create(
        title=title,
        conversation_id=request.conversation_id,
        file_path=str(file_path) if file_path else None,
    )

    return ReportResponse(
        report_id=report_id,
        title=title,
        conversation_id=request.conversation_id,
        file_path=str(file_path) if file_path else None,
        created_at=datetime.now(timezone.utc).isoformat(),
        download_url=f"/api/v1/reports/{report_id}/download" if file_path else None,
    )


@router.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: str):
    report = _report_repo.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return ReportResponse(**report, download_url=f"/api/v1/reports/{report_id}/download")


@router.get("/reports/{report_id}/download")
def download_report(report_id: str):
    report = _report_repo.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    file_path = Path(report.get("file_path") or "")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk")
    suffix = file_path.suffix.lower()
    media_type = "application/pdf" if suffix == ".pdf" else "text/plain"
    return FileResponse(str(file_path), media_type=media_type, filename=file_path.name)


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def _generate_pdf_report(title: str, conv: dict, messages: list) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title[:90], ln=True, align="C")
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 0, 0)
    pdf.multi_cell(0, 5,
        "IMPORTANT: This report provides legal information and preliminary analysis only. "
        "It is NOT formal legal advice and does not constitute legal representation. "
        "For consequential legal decisions, consult a qualified lawyer.")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Metadata
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}", ln=True)
    pdf.cell(0, 6, f"Jurisdiction: {conv.get('jurisdiction', 'Karnataka')}", ln=True)
    pdf.cell(0, 6, f"Language: {conv.get('language', 'English')}", ln=True)
    pdf.ln(4)

    # Conversation
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Conversation Transcript", ln=True)
    pdf.set_font("Helvetica", "", 10)

    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, f"{role}:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        # Handle multi-byte chars safely
        safe_content = content.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(0, 6, safe_content[:2000])
        pdf.ln(2)

    out_path = REPORTS_DIR / f"{conv['conversation_id']}.pdf"
    pdf.output(str(out_path))
    return out_path


def _generate_text_report(title: str, conv: dict, messages: list) -> Path:
    lines = [
        title,
        "=" * 70,
        "DISCLAIMER: This report provides legal information only, not formal legal advice.",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}",
        f"Jurisdiction: {conv.get('jurisdiction', 'Karnataka')}",
        f"Language: {conv.get('language', 'English')}",
        "",
        "CONVERSATION TRANSCRIPT",
        "-" * 70,
    ]
    for msg in messages:
        lines.append(f"\n{msg['role'].upper()}:")
        lines.append(msg["content"])

    out_path = REPORTS_DIR / f"{conv['conversation_id']}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path
