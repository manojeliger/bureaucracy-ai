from backend.app.services.llm.ollama_client import OllamaClient


def main():
    client = OllamaClient()

    prompt = "Explain RAG in AI in exactly 3 simple sentences."

    print("Sending request to Qwen 2.5 3B...\n")

    answer = client.generate(prompt)

    print("MODEL RESPONSE:")
    print(answer)


if __name__ == "__main__":
    main()