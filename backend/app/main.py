from fastapi import FastAPI

from backend.app.api.v1.chat import router as chat_router

app = FastAPI(
    title="Bureaucracy AI",
    description="AI-powered Indian Government Services Assistant",
    version="0.1.0"
)


@app.get("/")
def home():
    return {
        "message": "Bureaucracy AI Backend is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(
    chat_router,
    prefix="/api/v1",
    tags=["Chat"]
)