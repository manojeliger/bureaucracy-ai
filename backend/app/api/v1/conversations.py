"""
Conversation management API — persistent multi-session chat history.
"""

from typing import List
from fastapi import APIRouter, HTTPException

from backend.app.schemas.chat import ConversationDetail, ConversationSummary, RenameRequest
from backend.app.services.database.repositories import ConversationRepository, MessageRepository

router = APIRouter()
_conv_repo = ConversationRepository()
_msg_repo = MessageRepository()


@router.get("/conversations", response_model=List[ConversationSummary])
def list_conversations():
    """Return all conversations, most recent first."""
    rows = _conv_repo.list_all(limit=100)
    return [ConversationSummary(**r) for r in rows]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str):
    """Return a conversation and its full message history."""
    conv = _conv_repo.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = _msg_repo.list_for_conversation(conversation_id)
    return ConversationDetail(**conv, messages=messages)


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(conversation_id: str, body: RenameRequest):
    """Rename a conversation."""
    conv = _conv_repo.get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    _conv_repo.update_title(conversation_id, body.title)
    updated = _conv_repo.get(conversation_id)
    return ConversationSummary(**updated)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    deleted = _conv_repo.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True, "conversation_id": conversation_id}
