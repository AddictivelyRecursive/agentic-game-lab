from typing import Optional

import requests


class OllamaClient:
    """
    Lightweight wrapper for Ollama local model server.
    Assumes Ollama is running at http://localhost:11434
    """

    def __init__(self, model_name: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>",
            "stream": False,
            "options": {
                "temperature": float(temperature) if temperature is not None else 0.2,
            },
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"]