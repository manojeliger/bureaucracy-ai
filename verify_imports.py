import sys
sys.path.insert(0, r'c:\projects\IND_Bureaucracy-ai')

import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print('=== BUREAUCRACY AI — IMPORT VERIFICATION ===\n')

tests = [
    ('Config', 'from backend.app.core.config import settings; x=settings.DATABASE_URL[-35:]; print(f"  DB tail: {x}")'),
    ('Metadata (full schema)', 'from backend.app.services.rag.metadata import DocumentMetadata, ChunkMetadata, DocumentType, Jurisdiction, VerificationStatus; m=DocumentMetadata(document_id="x",title="y"); print(f"  Fields: {len(DocumentMetadata.model_fields)}")'),
    ('VectorStore', 'from backend.app.services.rag.vector_store import FAISSVectorStore; v=FAISSVectorStore(); print(f"  Dim={v.dimension}")'),
    ('EmbeddingService', 'from backend.app.services.rag.embedding_service import EmbeddingService; print("  OK (model load skipped)")'),
    ('DocumentChunker', 'from backend.app.services.rag.chunker import DocumentChunker; c=DocumentChunker(); print(f"  chunk_size={c.chunk_size}")'),
    ('KeywordSearch', 'from backend.app.services.rag.keyword_search import KeywordSearchService; print("  OK")'),
    ('RetrievalService', 'from backend.app.services.rag.retrieval_service import RetrievalService, RetrievedChunk; print("  OK")'),
    ('CitationService', 'from backend.app.services.rag.citation_service import CitationService; print("  OK")'),
    ('DB Connection', 'from backend.app.services.database.connection import init_db, get_connection; print("  OK")'),
    ('Repositories', 'from backend.app.services.database.repositories import ConversationRepository, MessageRepository, DocumentRepository, LegalSourceRepository, LegalChunkRepository, CitationRepository; print("  OK")'),
    ('PDFExtractor', 'from backend.app.services.ingestion.pdf_extractor import PDFExtractor; print("  OK")'),
    ('DocumentProcessor', 'from backend.app.services.ingestion.document_processor import DocumentProcessor, classify_document_type; print("  OK")'),
    ('OCRService', 'from backend.app.services.ingestion.ocr_service import OCRService; print("  OK")'),
    ('OllamaClient', 'from backend.app.services.llm.ollama_client import OllamaClient; print("  OK")'),
    ('QueryAnalyzer', 'from backend.app.services.legal.query_analyzer import QueryAnalyzer, classify_legal_domain; print("  OK")'),
    ('FactExtractor', 'from backend.app.services.legal.fact_extractor import FactExtractor; print("  OK")'),
    ('InvestigationEngine', 'from backend.app.services.legal.investigation_engine import InvestigationEngine, RemedyEngine; print("  OK")'),
    ('Schemas', 'from backend.app.schemas.chat import ChatRequest, ChatResponse; from backend.app.schemas.document import DocumentUploadResponse; from backend.app.schemas.legal import LegalSearchRequest, ReportRequest; print("  OK")'),
    ('API chat module', 'import backend.app.api.v1.chat; print("  OK")'),
    ('API conversations', 'import backend.app.api.v1.conversations; print("  OK")'),
    ('API documents', 'import backend.app.api.v1.documents; print("  OK")'),
    ('API legal_search', 'import backend.app.api.v1.legal_search; print("  OK")'),
    ('API reports', 'import backend.app.api.v1.reports; print("  OK")'),
    ('API admin', 'import backend.app.api.v1.admin; print("  OK")'),
    ('classify_document_type', 'from backend.app.services.ingestion.document_processor import classify_document_type; r=classify_document_type("My landlord is refusing to return deposit rental agreement", "agreement.pdf"); print(f"  Result: {r}")'),
    ('classify_legal_domain', 'from backend.app.services.legal.query_analyzer import classify_legal_domain; d,c=classify_legal_domain("My employer has not paid salary for 2 months"); print(f"  Domain: {d} ({c})")'),
    ('DB init', 'from backend.app.services.database.connection import init_db; init_db(); print("  Tables created OK")'),
    ('ConvRepo CRUD', '''
from backend.app.services.database.repositories import ConversationRepository
r = ConversationRepository()
c = r.create(title="Test conversation")
assert c["conversation_id"]
found = r.get(c["conversation_id"])
assert found["title"] == "Test conversation"
r.delete(c["conversation_id"])
print("  Create/Get/Delete OK")
'''),
]

passed = 0
failed = 0
for name, code in tests:
    try:
        exec(code)
        print(f'[PASS] {name}')
        passed += 1
    except Exception as e:
        print(f'[FAIL] {name}: {e}')
        failed += 1

print(f'\n{"="*50}')
print(f'RESULT: {passed}/{passed+failed} tests passed')
if failed == 0:
    print('ALL IMPORTS AND CORE LOGIC OK')
else:
    print(f'{failed} failures need attention')
