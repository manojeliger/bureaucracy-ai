"""
Request and response schemas for the chat API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Citizen's current message")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Recent message history")
    language: str = Field(default="English", description="Response language")
    state: str = Field(default="Karnataka", description="Jurisdiction state")
    document_text: Optional[str] = Field(default=None, description="Extracted text from uploaded document")
    document_id: Optional[str] = Field(default=None, description="ID of an already-processed document")
    legal_domain: Optional[str] = Field(default=None, description="Legal domain hint")


class SourceItem(BaseModel):
    title: str
    department: str
    url: Optional[str] = None
    confidence: float
    is_verified: bool = False
    section: Optional[str] = None
    page: Optional[int] = None
    jurisdiction: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    sources: List[SourceItem] = []
    follow_up_question: Optional[str] = None
    has_grounded_context: bool = True
    legal_domain: Optional[str] = None


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    legal_domain: Optional[str] = None
    jurisdiction: str = "Karnataka"
    language: str = "English"
    created_at: str
    updated_at: str


class ConversationDetail(ConversationSummary):
    messages: List[Dict[str, Any]] = []


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
