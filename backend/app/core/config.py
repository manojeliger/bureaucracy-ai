"""
Central configuration for Bureaucracy AI.

All paths, model names, and tunable parameters live here.
Values can be overridden via the .env file without touching source code.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base project root directory (IND_Bureaucracy-ai/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # --- Application ---
    APP_NAME: str = "Bureaucracy_AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # --- Database ---
    # Empty string in .env falls back to the default path below.
    DATABASE_URL: str = ""

    @property
    def resolved_db_url(self) -> str:
        """Return the effective DB path, falling back to the project-relative default."""
        if self.DATABASE_URL and self.DATABASE_URL.strip():
            return self.DATABASE_URL
        return str(PROJECT_ROOT / "data" / "database" / "bureaucracy_ai.db")

    # --- LLM (Ollama) ---
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "qwen2.5:3b"

    # --- Embeddings ---
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
    EMBEDDING_DIMENSION: int = 384

    # --- RAG ---
    FAISS_INDEX_DIR: str = str(PROJECT_ROOT / "data" / "processed" / "faiss")
    RAW_DATA_DIR: str = str(PROJECT_ROOT / "data" / "raw")
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.60
    RAG_KEYWORD_TOP_K: int = 5
    RAG_MAX_CONTEXT_CHARS: int = 6000

    # --- File uploads ---
    UPLOADS_DIR: str = str(PROJECT_ROOT / "data" / "uploads")
    MAX_UPLOAD_SIZE_MB: int = 15
    ALLOWED_UPLOAD_EXTENSIONS: list = ["pdf", "png", "jpg", "jpeg"]

    # --- Security ---
    SECRET_KEY: str = "change_me_in_production"
    ADMIN_API_KEY: str = "admin_change_me"

    # --- Scraping ---
    SCRAPING_RATE_LIMIT_SECONDS: float = 2.0
    SCRAPING_MAX_RETRIES: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
