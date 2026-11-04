from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Bureaucracy_AI"
    APP_ENV: str = "development"
    DEBUG: bool = True

    DATABASE_URL: str = ""

    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:3b"

    EMBEDDING_MODEL: str = ""

    SECRET_KEY: str = "change_me"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )


settings = Settings()