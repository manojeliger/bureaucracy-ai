# Bureaucracy AI (Karnataka, India)
## Intelligent Citizen-Service Investigation & Administrative Guidance Assistant

Bureaucracy AI is an academic AI prototype designed to help citizens understand and navigate complex government-service problems in Karnataka, India. Operating like an investigative administrative partner, it diagnoses application rejections, uncovers causes of delayed services or duplicate tax demands, retrieves verified official rules, and guides citizens to appropriate grievance escalation routes without hallucinating.

---

## 🏛️ System Architecture

```text
                           ┌────────────────────────┐
                           │   CITIZEN / USER UI    │
                           │  (Streamlit Frontend)  │
                           └───────────┬────────────┘
                                       │ HTTP POST /chat & /extract-document
                                       ▼
                           ┌────────────────────────┐
                           │    FASTAPI BACKEND     │
                           │   (app/main.py & API)  │
                           └───────────┬────────────┘
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
      ┌──────────────────────┐                   ┌──────────────────────┐
      │ EMBEDDING SERVICE    │                   │ DOCUMENT PARSER      │
      │ Multilingual E5      │                   │ PyMuPDF (PDF/Challan)│
      │ 384-dim CPU Vectors  │                   └──────────────────────┘
      └──────────┬───────────┘
                 │ Normalized Query Embedding
                 ▼
      ┌──────────────────────┐
      │   FAISS VECTOR DB    │
      │  Cosine / IP Search  │
      │ (30 Service Chunks)  │
      └──────────┬───────────┘
                 │ Top-K Filtered Context & Verified Links
                 ▼
      ┌──────────────────────────────────────────────┐
      │       INVESTIGATION & RAG ENGINE             │
      │      (app/services/rag/rag_service.py)       │
      │ - Problem Classification                     │
      │ - Requirement & Eligibility Comparison       │
      │ - Next-Step Formulation                      │
      │ - Non-Intrusive Follow-Up Inquiry            │
      └──────────────────────┬───────────────────────┘
                             │ Grounded Prompt Context
                             ▼
      ┌──────────────────────────────────────────────┐
      │              LOCAL LLM ENGINE                │
      │            Ollama (Qwen 2.5 3B)              │
      │        Zero Paid External API Calls          │
      └──────────────────────┬───────────────────────┘
                             │ Verified & Grounded Investigation
                             ▼
                   [Delivered to Citizen]
```

---

## 🛠️ Technology Stack

| Layer | Component | Specification |
|---|---|---|
| **Frontend** | Streamlit | Responsive, accessible citizen interface with document extraction previews, official portal badges, and status pills |
| **Backend API** | FastAPI / Uvicorn | RESTful endpoints: `GET /health`, `POST /api/v1/chat`, `POST /api/v1/extract-document` |
| **Retrieval Engine** | FAISS (`faiss-cpu`) | IndexFlatIP index storing 30 processed chunks covering 10 Karnataka government domains |
| **Embeddings** | `intfloat/multilingual-e5-small` | 384-dimensional dense vectors running locally on CPU in offline mode |
| **Local LLM** | `qwen2.5:3b` via Ollama | Local inference on consumer hardware (tested on AMD Ryzen 5 + NVIDIA RTX 3050 Laptop GPU) |
| **Document Parsing** | PyMuPDF (`pymupdf`) | Instant extraction of text from notices, challans, and rejection slips |
| **Language Support** | Multilingual Prompting | English, Kannada, and Hindi prompt instructions |

---

## 📁 Repository Structure

```text
IND_Bureaucracy-ai/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── chat.py             # Chat & document extraction endpoints
│   │   │       └── __init__.py
│   │   ├── core/
│   │   │   └── config.py               # Paths, model parameters & thresholds
│   │   ├── services/
│   │   │   ├── llm/
│   │   │   │   └── ollama_client.py    # Local Ollama REST client
│   │   │   ├── rag/
│   │   │   │   ├── chunker.py          # Word-aware text chunking
│   │   │   │   ├── embedding_service.py# Multilingual E5 embedding generator
│   │   │   │   ├── metadata.py         # Pydantic schemas for document metadata
│   │   │   │   ├── vector_store.py     # FAISS storage, load, search, & save
│   │   │   │   └── rag_service.py      # Core RAG investigation orchestrator
│   │   │   └── ingestion/
│   │   │       ├── pdf_extractor.py    # PyMuPDF extractor
│   │   │       ├── document_processor.py# Page-to-chunk mapper
│   │   │       └── ingest_knowledge.py # Ingestion pipeline for PDF, JSON, & CSV
│   │   └── main.py                     # FastAPI application & health check
│   └── tests/
│       ├── test_chunker.py             # Chunking unit tests
│       ├── test_document_processor.py  # Processing tests
│       ├── test_embeddings.py          # Embedding dimension & normalization tests
│       ├── test_pdf_extractor.py       # PDF parsing tests
│       ├── test_rag_pipeline.py        # Metadata tests
│       ├── test_rag_service.py         # FAISS retrieval & grounding tests
│       ├── test_chat_api.py            # API contract & memory tests
│       └── test_rag_scenarios.py       # 10 benchmark citizen scenario tests
│
├── frontend/
│   └── app.py                          # Streamlit application
│
├── data/
│   ├── raw/                            # Source documents (PDF, JSON, CSV)
│   ├── processed/faiss/                # Serialized FAISS index & metadata
│   └── uploads/                        # Temporary uploaded citizen attachments
│
├── .env                                # Local environment settings
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.11 installed
- [Ollama](https://ollama.ai) installed and running locally
- Pull the local Qwen model:
  ```powershell
  ollama pull qwen2.5:3b
  ```

### 2. Activate Virtual Environment
```powershell
cd C:\projects\IND_Bureaucracy-ai
.\.venv\Scripts\Activate.ps1
```

### 3. Knowledge Base Ingestion (Already Pre-Built)
If you wish to re-index or update raw files in `data/raw/`:
```powershell
$env:PYTHONPATH = "C:\projects\IND_Bureaucracy-ai"
python backend/app/services/ingestion/ingest_knowledge.py
```

### 4. Start Backend Server
```powershell
$env:PYTHONPATH = "C:\projects\IND_Bureaucracy-ai"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Verify health status:
```powershell
curl http://127.0.0.1:8000/health
```

### 5. Start Frontend Application
In a new terminal window:
```powershell
cd C:\projects\IND_Bureaucracy-ai
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "C:\projects\IND_Bureaucracy-ai"
streamlit run frontend/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing and Validation

Run the complete test suite:
```powershell
$env:PYTHONPATH = "C:\projects\IND_Bureaucracy-ai"
python -m unittest discover -s backend/tests
```

### Benchmark Scenario Validation (10/10 Passed)
The 10 key citizen queries specified in the project benchmark passed verification:
1. **Income Certificate Procedure:** Grounds to Seva Sindhu / Nadakacheri; outlines VA Mahazar verification.
2. **Rejected Certificate Application:** Investigates eligibility mismatch, missing documents, and Sakala appeal route.
3. **Bhoomi Portal Purpose:** Details rural land records, RTC (Pahani), and mutation procedures under the Revenue Department.
4. **Ration Card Grievance:** Grounds to Ahara portal, e-KYC, and Food & Civil Supplies Department.
5. **Delayed Scholarship:** Pinpoints SSP portal, NPCI Aadhaar bank mapping, and RD certificate mismatches.
6. **Duplicate Property Tax Notice:** Distinguishes municipal vs. rural tax; advises verification via payment receipt & PID.
7. **Sakala Act:** Clarifies guaranteed timeline and grievance redressal under DPAR.
8. **Land Mutation Authority:** Identifies Village Accountant and Tahsildar through Bhoomi.
9. **Pension Scheme Documents:** Outlines income proof, Aadhaar, and age documentation.
10. **Unknown / Out-of-Domain Topic:** Refuses to hallucinate on Martian/fictional queries and honestly reports absence from database.

---

## 🔒 Responsible AI & Privacy Guidelines
- **Zero Hallucination of Credentials:** Never asks for citizen OTPs, passwords, bank account passwords, or personal mobile numbers.
- **Source Grounding:** Only cites verified URLs registered in `ksr.csv` (e.g. Seva Sindhu, Bhoomi, Sakala, Ahara, SSP).
- **Academic Scope:** Communicates clear disclaimers that advice is administrative assistance, not legal or official administrative decisions.
