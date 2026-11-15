"""
Bureaucracy AI — Streamlit frontend application.

A polished, ChatGPT-style legal intelligence interface for Karnataka citizens.

Features:
- Multi-session conversation sidebar with persistence
- Bottom composer with + attachment button
- Chat-first, document-first, and combined workflows
- Source citation display with verification badges
- Suggested follow-up questions as clickable chips
- Legal assistance report generation
- Light professional design system
"""

import os
import io
import json
import time
import base64
import requests
import streamlit as st
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bureaucracy AI — Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Bureaucracy AI — Karnataka Legal Intelligence System"},
)

# ─────────────────────────────────────────────────────────────────────────────
# Design system CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Main layout ───────────────────────────────────────────────────── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ───────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f1117 !important;
    border-right: 1px solid #2d2d2d !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #2d3748 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-size: 0.85rem !important;
    padding: 8px 12px !important;
    margin-bottom: 2px !important;
    transition: background 0.15s !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1a202c !important;
    border-color: #4a5568 !important;
}

/* ── App header ────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 20px 24px 16px;
    border-bottom: 1px solid #e2e8f0;
    background: #ffffff;
    position: sticky;
    top: 0;
    z-index: 100;
}
.app-logo { font-size: 2rem; }
.app-title { font-size: 1.4rem; font-weight: 700; color: #1a202c; }
.app-subtitle { font-size: 0.8rem; color: #718096; margin-top: 1px; }
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-left: auto;
}
.status-healthy { background: #c6f6d5; color: #22543d; }
.status-degraded { background: #fed7d7; color: #742a2a; }

/* ── Chat area ─────────────────────────────────────────────────────── */
.chat-area {
    max-width: 820px;
    margin: 0 auto;
    padding: 20px 20px 160px;
}

/* ── Message bubbles ───────────────────────────────────────────────── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.msg-user-bubble {
    background: #2d3748;
    color: #f7fafc;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.95rem;
    line-height: 1.6;
    white-space: pre-wrap;
}
.msg-assistant {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
}
.msg-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea, #764ba2);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.9rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-assistant-bubble {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #2d3748;
}

/* ── Source citations ──────────────────────────────────────────────── */
.source-card {
    background: #ebf8ff;
    border-left: 3px solid #3182ce;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.82rem;
    color: #2b4c7e;
}
.source-verified { border-left-color: #38a169 !important; background: #f0fff4 !important; }
.source-badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 6px;
}
.badge-verified { background: #c6f6d5; color: #22543d; }
.badge-unverified { background: #fef3c7; color: #92400e; }

/* ── Follow-up chips ───────────────────────────────────────────────── */
.followup-chip {
    display: inline-block;
    padding: 6px 14px;
    background: #ebf4ff;
    border: 1px solid #bee3f8;
    border-radius: 20px;
    color: #2b6cb0;
    font-size: 0.82rem;
    cursor: pointer;
    margin: 4px 4px 4px 0;
    transition: background 0.15s;
}

/* ── Composer ──────────────────────────────────────────────────────── */
.composer-wrapper {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #ffffff;
    border-top: 1px solid #e2e8f0;
    padding: 12px 24px 20px;
    z-index: 200;
}
.composer-inner {
    max-width: 820px;
    margin: 0 auto;
}
.attachment-preview {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: #ebf8ff;
    border-radius: 8px;
    font-size: 0.82rem;
    color: #2b4c7e;
    margin-bottom: 8px;
    border: 1px solid #bee3f8;
}
.disclaimer-text {
    text-align: center;
    font-size: 0.72rem;
    color: #a0aec0;
    margin-top: 8px;
}

/* ── Empty state ───────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #718096;
}
.empty-icon { font-size: 3rem; margin-bottom: 16px; }
.empty-title { font-size: 1.5rem; font-weight: 600; color: #2d3748; margin-bottom: 8px; }
.empty-sub { font-size: 0.95rem; color: #718096; margin-bottom: 32px; }

/* ── Starter suggestions ───────────────────────────────────────────── */
.starter-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 600px;
    margin: 0 auto;
}
.starter-card {
    background: #f7fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 16px;
    text-align: left;
    cursor: pointer;
    font-size: 0.85rem;
    color: #2d3748;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.starter-card:hover {
    border-color: #667eea;
    box-shadow: 0 2px 8px rgba(102,126,234,0.15);
}
.starter-card-icon { font-size: 1.2rem; margin-bottom: 6px; }
.starter-card-text { font-weight: 500; }

/* ── Domain badge ──────────────────────────────────────────────────── */
.domain-badge {
    display: inline-block;
    padding: 2px 10px;
    background: #e9d8fd;
    color: #553c9a;
    border-radius: 10px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Loading ───────────────────────────────────────────────────────── */
.loading-dots { display: inline-flex; gap: 4px; }
.loading-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #667eea;
    animation: bounce 1.2s ease-in-out infinite;
}
.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
    40% { transform: scale(1.1); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 5):
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_post(path: str, json_data: dict = None, files=None, timeout: int = 90):
    try:
        if files:
            r = requests.post(f"{BACKEND_URL}{path}", files=files, timeout=timeout)
        else:
            r = requests.post(f"{BACKEND_URL}{path}", json=json_data, timeout=timeout)
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return {"error": "Backend server not reachable. Please ensure the FastAPI server is running."}, 503
    except Exception as exc:
        return {"error": str(exc)}, 500


# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "conversation_id": None,
    "messages": [],             # [{role, content, sources, follow_up}]
    "pending_message": None,    # Message to send on next rerun
    "attachment_bytes": None,
    "attachment_name": None,
    "attachment_type": None,
    "attachment_extracted": None,
    "attachment_doc_id": None,
    "conversations_cache": [],
    "health_cache": None,
    "health_ts": 0,
    "language": "English",
    "state": "Karnataka",
    "show_upload": False,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_health():
    now = time.time()
    if now - st.session_state.health_ts > 30:  # Cache 30s
        st.session_state.health_cache = api_get("/health")
        st.session_state.health_ts = now
    return st.session_state.health_cache


def load_conversations():
    data = api_get("/api/v1/conversations")
    st.session_state.conversations_cache = data or []
    return st.session_state.conversations_cache


def start_new_chat():
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.attachment_bytes = None
    st.session_state.attachment_name = None
    st.session_state.attachment_extracted = None
    st.session_state.attachment_doc_id = None
    st.session_state.pending_message = None


def select_conversation(conv_id: str):
    data = api_get(f"/api/v1/conversations/{conv_id}")
    if data:
        st.session_state.conversation_id = conv_id
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"], "sources": [], "follow_up": None}
            for m in data.get("messages", [])
        ]


def upload_and_extract(file_bytes: bytes, file_name: str, file_type: str) -> dict:
    files = {"file": (file_name, io.BytesIO(file_bytes), f"application/{file_type}")}
    result, status = api_post("/api/v1/extract-document", files=files, timeout=30)
    return result if status == 200 else {}


def send_chat_message(message: str, doc_text: str = None) -> dict:
    # Build history from session state (last 10 exchanges)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-12:]
    ]
    payload = {
        "message": message,
        "conversation_id": st.session_state.conversation_id,
        "history": history,
        "language": st.session_state.language,
        "state": st.session_state.state,
        "document_text": doc_text,
    }
    result, status = api_post("/api/v1/chat", json_data=payload, timeout=120)
    return result if status == 200 else {"error": result.get("detail", "Unknown error")}


def generate_report() -> dict:
    if not st.session_state.conversation_id:
        return {"error": "No active conversation to generate report from."}
    payload = {"conversation_id": st.session_state.conversation_id}
    result, status = api_post("/api/v1/reports/generate", json_data=payload, timeout=60)
    return result if status == 200 else {"error": result.get("detail", "Report generation failed")}


def format_domain(domain: str) -> str:
    mapping = {
        "employment_salary": "Employment & Salary",
        "rental_property": "Rental & Property",
        "consumer_dispute": "Consumer Rights",
        "land_revenue": "Land & Revenue",
        "government_service": "Government Services",
        "family_civil": "Family & Civil",
        "criminal": "Criminal",
        "taxation": "Taxation",
        "pension_welfare": "Pension & Welfare",
        "general_rights": "Citizen Rights",
    }
    return mapping.get(domain, domain.replace("_", " ").title())


def render_sources(sources: list):
    if not sources:
        return
    with st.expander("📚 Sources & References", expanded=False):
        for s in sources:
            verified = s.get("is_verified", False)
            card_class = "source-card source-verified" if verified else "source-card"
            badge_class = "badge-verified" if verified else "badge-unverified"
            badge_text = "✓ Official" if verified else "Unverified"
            url_html = f' &nbsp;<a href="{s["url"]}" target="_blank" style="color:#3182ce;">Visit →</a>' if s.get("url") else ""
            section_html = f'<br><small>Section: {s["section"]}</small>' if s.get("section") else ""
            conf = int(s.get("confidence", 0) * 100)
            st.markdown(f"""
<div class="{card_class}">
  <strong>{s.get("title", "Legal Source")}</strong>
  <span class="source-badge {badge_class}">{badge_text}</span>
  {url_html}
  <br><small>{s.get("department", "")} · Relevance: {conf}%</small>
  {section_html}
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Branding
    st.markdown("""
<div style="padding: 8px 0 20px;">
  <div style="font-size:1.6rem; font-weight:800; color:#e2e8f0; letter-spacing:-0.5px;">
    ⚖️ Bureaucracy AI
  </div>
  <div style="font-size:0.75rem; color:#718096; margin-top:4px;">
    Karnataka Legal Intelligence System
  </div>
</div>
""", unsafe_allow_html=True)

    # New chat button
    if st.button("✏️  New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("---")

    # Settings
    st.caption("SETTINGS")
    st.session_state.language = st.selectbox(
        "Language", ["English", "Kannada", "Hindi"],
        index=["English", "Kannada", "Hindi"].index(st.session_state.language),
        label_visibility="collapsed",
    )
    st.session_state.state = st.selectbox(
        "Jurisdiction", ["Karnataka", "Central Government"],
        index=0, label_visibility="collapsed",
    )

    st.markdown("---")

    # Recent conversations
    st.caption("RECENT CONVERSATIONS")
    convs = load_conversations()
    if convs:
        for conv in convs[:20]:
            title = conv.get("title", "Untitled")[:38]
            cid = conv["conversation_id"]
            is_active = cid == st.session_state.conversation_id
            btn_label = f"{'▶ ' if is_active else ''}{title}"
            if st.button(btn_label, key=f"conv_{cid}", use_container_width=True):
                select_conversation(cid)
                st.rerun()
    else:
        st.markdown('<div style="color:#4a5568;font-size:0.8rem;">No conversations yet.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # Health status
    health = get_health()
    if health and health.get("status") == "healthy":
        vectors = health.get("components", {}).get("indexed_vectors", 0)
        st.markdown(
            f'<div style="color:#68d391;font-size:0.78rem;">● System Ready · {vectors} legal records indexed</div>',
            unsafe_allow_html=True,
        )
    elif health:
        comps = health.get("components", {})
        issues = []
        if comps.get("ollama_api") != "connected":
            issues.append("Ollama offline")
        if comps.get("faiss_index") != "loaded":
            issues.append("Index not loaded")
        st.markdown(
            f'<div style="color:#fc8181;font-size:0.78rem;">● Degraded · {", ".join(issues) or "Check backend"}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="color:#fc8181;font-size:0.78rem;">● Backend unreachable</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Report generation
    if st.session_state.conversation_id and len(st.session_state.messages) >= 2:
        if st.button("📄 Generate Legal Report", use_container_width=True):
            with st.spinner("Generating report..."):
                report = generate_report()
            if "error" in report:
                st.error(report["error"])
            elif report.get("download_url"):
                st.success("Report ready!")
                st.markdown(f"[📥 Download Report]({BACKEND_URL}{report['download_url']})")
            else:
                st.info(f"Report created: {report.get('report_id', '')}")

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.72rem; color:#4a5568; line-height:1.6;">
<strong>Legal Information Disclaimer</strong><br>
Bureaucracy AI provides general legal information and preliminary assistance only.<br>
It is NOT formal legal advice and does not constitute legal representation.<br>
For important legal decisions, consult a qualified advocate.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main content area
# ─────────────────────────────────────────────────────────────────────────────

main_col = st.container()

with main_col:
    # ── Header ──────────────────────────────────────────────────────────────
    health = get_health()
    health_status = health.get("status") if health else "unknown"
    status_class = "status-healthy" if health_status == "healthy" else "status-degraded"
    status_text = "● System Ready" if health_status == "healthy" else "● Degraded"

    st.markdown(f"""
<div class="app-header">
  <span class="app-logo">⚖️</span>
  <div>
    <div class="app-title">Bureaucracy AI</div>
    <div class="app-subtitle">Karnataka Legal Intelligence & Citizen Rights Assistant</div>
  </div>
  <span class="status-pill {status_class}">{status_text}</span>
</div>
""", unsafe_allow_html=True)

    # ── Chat messages area ───────────────────────────────────────────────────
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    messages = st.session_state.messages

    if not messages:
        # Empty state with starter questions
        st.markdown("""
<div class="empty-state">
  <div class="empty-icon">⚖️</div>
  <div class="empty-title">How can I help you today?</div>
  <div class="empty-sub">Describe your legal problem, upload a document, or choose a common situation below.</div>
</div>
""", unsafe_allow_html=True)

        STARTERS = [
            ("💰", "My employer hasn't paid salary for 2 months. What can I do?"),
            ("🏠", "My landlord is refusing to return my security deposit."),
            ("📋", "My income certificate application was rejected on Seva Sindhu."),
            ("👨‍👩‍👧", "I received a legal notice. Can you help me understand it?"),
            ("🌾", "How do I check the status of my land mutation on Bhoomi?"),
            ("🎓", "My Karnataka SSP scholarship payment is delayed. Help?"),
        ]

        cols = st.columns(2)
        for i, (icon, text) in enumerate(STARTERS):
            col = cols[i % 2]
            with col:
                if st.button(f"{icon}  {text}", key=f"starter_{i}", use_container_width=True):
                    st.session_state.pending_message = text
                    st.rerun()
    else:
        # Render messages
        for idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]
            sources = msg.get("sources", [])
            follow_up = msg.get("follow_up")
            domain = msg.get("legal_domain")

            if role == "user":
                st.markdown(f"""
<div class="msg-user">
  <div class="msg-user-bubble">{content}</div>
</div>""", unsafe_allow_html=True)

            elif role == "assistant":
                st.markdown("""
<div class="msg-assistant">
  <div class="msg-avatar">⚖️</div>
  <div class="msg-assistant-bubble">
""", unsafe_allow_html=True)

                # Domain badge
                if domain and domain != "unknown":
                    st.markdown(
                        f'<span class="domain-badge">{format_domain(domain)}</span><br><br>',
                        unsafe_allow_html=True,
                    )

                st.markdown(content)
                st.markdown("</div></div>", unsafe_allow_html=True)

                # Sources
                if sources:
                    render_sources(sources)

                # Follow-up question as clickable chip
                if follow_up and idx == len(messages) - 1:
                    st.markdown('<div style="margin-top:8px;">', unsafe_allow_html=True)
                    if st.button(f"💬 {follow_up}", key=f"followup_{idx}"):
                        st.session_state.pending_message = follow_up
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close chat-area


# ─────────────────────────────────────────────────────────────────────────────
# Message composer (fixed bottom)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="composer-wrapper"><div class="composer-inner">', unsafe_allow_html=True)

# Attachment preview (shown above input if file attached)
if st.session_state.attachment_name:
    col_prev, col_remove = st.columns([9, 1])
    with col_prev:
        st.markdown(f"""
<div class="attachment-preview">
  📎 <strong>{st.session_state.attachment_name}</strong>
  &nbsp;·&nbsp; Ready to include with your message
</div>""", unsafe_allow_html=True)
    with col_remove:
        if st.button("✕", key="remove_attachment", help="Remove attachment"):
            st.session_state.attachment_bytes = None
            st.session_state.attachment_name = None
            st.session_state.attachment_type = None
            st.session_state.attachment_extracted = None
            st.session_state.attachment_doc_id = None
            st.rerun()

# Composer row: [+] [text input] [Send]
col_attach, col_input, col_send = st.columns([1, 10, 1])

with col_attach:
    # + button opens a file uploader
    if st.button("➕", help="Attach a legal document (PDF, PNG, JPG)", key="attach_btn"):
        st.session_state.show_upload = not st.session_state.show_upload
        st.rerun()

with col_input:
    user_input = st.chat_input(
        placeholder="Describe your legal situation, ask a question, or attach a document...",
        key="chat_input",
    )

with col_send:
    # Visual placeholder — st.chat_input already has Enter to send
    st.markdown("&nbsp;")

# File uploader (shown when + is clicked)
if st.session_state.show_upload:
    uploaded = st.file_uploader(
        "Upload a legal document",
        type=["pdf", "png", "jpg", "jpeg"],
        key="file_uploader",
        help="PDF, PNG, or JPG up to 15 MB",
        label_visibility="collapsed",
    )
    if uploaded and uploaded.name != st.session_state.attachment_name:
        with st.spinner(f"Processing {uploaded.name}..."):
            file_bytes = uploaded.read()
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            extracted_info = upload_and_extract(file_bytes, uploaded.name, ext)

        if extracted_info.get("extracted_text"):
            st.session_state.attachment_bytes = file_bytes
            st.session_state.attachment_name = uploaded.name
            st.session_state.attachment_type = ext
            st.session_state.attachment_extracted = extracted_info["extracted_text"]
            st.session_state.show_upload = False
            st.success(f"✓ Extracted {extracted_info.get('page_count', 1)} page(s)")

            # If no message yet, trigger an AI analysis of the document
            if not st.session_state.messages:
                st.session_state.pending_message = (
                    f"I have uploaded a document: {uploaded.name}. "
                    "Please analyse it and explain what it means in plain language. "
                    "Also suggest what questions I should ask about it."
                )
            st.rerun()
        else:
            st.error("Could not extract text from document. Please check the file.")

st.markdown("""
<div class="disclaimer-text">
Bureaucracy AI provides legal information only, not formal legal advice.
For important decisions, consult a qualified advocate.
</div>
""", unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)  # close composer


# ─────────────────────────────────────────────────────────────────────────────
# Message processing
# ─────────────────────────────────────────────────────────────────────────────

# Collect the message to send (from input OR pending)
message_to_send = user_input or st.session_state.pending_message
if st.session_state.pending_message:
    st.session_state.pending_message = None

if message_to_send and message_to_send.strip():
    # Add user message to display immediately
    st.session_state.messages.append({
        "role": "user",
        "content": message_to_send,
        "sources": [],
        "follow_up": None,
    })

    # Send to backend
    with st.spinner("Analysing your situation..."):
        response = send_chat_message(
            message=message_to_send,
            doc_text=st.session_state.attachment_extracted,
        )

    if "error" in response:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {response['error']}",
            "sources": [],
            "follow_up": None,
        })
    else:
        # Persist conversation ID
        if response.get("conversation_id"):
            st.session_state.conversation_id = response["conversation_id"]

        # Build sources list (merge new schema fields)
        raw_sources = response.get("sources", [])
        sources = []
        for s in raw_sources:
            sources.append({
                "title": s.get("title", ""),
                "department": s.get("department", ""),
                "url": s.get("url", ""),
                "confidence": s.get("confidence", 0.0),
                "is_verified": s.get("is_verified", False),
                "section": s.get("section"),
            })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.get("response", ""),
            "sources": sources,
            "follow_up": response.get("follow_up_question"),
            "legal_domain": response.get("legal_domain"),
        })

        # Clear attachment after first use
        st.session_state.attachment_extracted = None
        st.session_state.attachment_name = None

    st.rerun()
