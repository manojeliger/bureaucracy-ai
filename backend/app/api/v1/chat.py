from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.services.llm.ollama_client import OllamaClient

router = APIRouter()

llm_client = OllamaClient()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = llm_client.generate(request.message)

        return ChatResponse(response=answer)

    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error)
        )