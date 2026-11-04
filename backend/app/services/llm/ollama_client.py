import requests

from backend.app.core.config import settings


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.model = settings.LLM_MODEL

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            return response.json()["response"]

        except requests.RequestException as error:
            raise RuntimeError(
                f"Could not communicate with Ollama: {error}"
            ) from error