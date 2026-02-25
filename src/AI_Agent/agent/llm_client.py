import requests


class OllamaClient:
    """
    Lightweight wrapper for Ollama local model server.

    Assumes Ollama is running at:
        http://localhost:11434

    Model must be available locally (e.g., llama3.1:8b).
    """

    def __init__(self, model_name="llama3.1:8b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat-style request to Ollama and return raw text output.
        """

        url = f"{self.host}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>",
            "stream": False
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        return response.json()["response"]