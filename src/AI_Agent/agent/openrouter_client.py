from __future__ import annotations

import os
import requests
from typing import Optional


class OpenRouterClient:
    """
    OpenRouter chat-completions client (OpenAI-compatible).

    Reads API key from env:
      OPENROUTER_API_KEY
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: int = 60,
        site_url: Optional[str] = None,   # optional: helps OpenRouter analytics
        app_name: Optional[str] = None,   # optional: helps OpenRouter analytics
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        self.api_key = key

        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL")
        self.app_name = app_name or os.getenv("OPENROUTER_APP_NAME")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Optional headers recommended by OpenRouter (non-fatal if omitted)
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        # If unauthorized, raise (so your node writes N6_error / N8_error)
        resp.raise_for_status()
        data = resp.json()

        # OpenAI-compatible response
        return data["choices"][0]["message"]["content"]