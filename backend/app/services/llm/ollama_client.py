import requests


class OllamaClient:
    def __init__(
        self,
        base_url="http://localhost:11434",
        model="qwen2.5:3b"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model

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