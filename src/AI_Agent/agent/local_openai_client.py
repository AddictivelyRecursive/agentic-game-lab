"""Key-free, loopback-only OpenAI chat transport for a Slurm-managed vLLM."""
from __future__ import annotations

import math
import os
from urllib.parse import urlsplit

import requests


class LocalOpenAIClient:
    def __init__(self, model_name=None, base_url=None, *, generation_config=None, timeout_s=None):
        self.model_name = model_name or os.getenv("LLM_MODEL")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        url = urlsplit(self.base_url)
        if (url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1")
                or url.username or url.password or url.query or url.fragment or url.path != "/v1"):
            raise ValueError("Local backend requires a loopback http://localhost:<port>/v1 URL")
        if not self.model_name:
            raise ValueError("Local backend requires model_name or LLM_MODEL")
        self.timeout_s = float(timeout_s if timeout_s is not None else os.getenv("LOCAL_TIMEOUT", "600"))
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("Local timeout must be positive and finite")
        self.generation_config = {"temperature": 0.2, "top_p": 1.0, "max_tokens": 2048}
        supplied = dict(generation_config or {})
        if supplied.keys() - {"temperature", "top_p", "max_tokens", "seed"}:
            raise ValueError("Supported generation parameters: temperature, top_p, max_tokens, seed")
        self.generation_config.update(supplied)
        for key, low, high in (("temperature", 0, 2), ("top_p", 0, 1)):
            value = self.generation_config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"Invalid generation parameter {key}")
        if self.generation_config["top_p"] == 0:
            raise ValueError("top_p must be greater than zero")
        tokens = self.generation_config["max_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        seed = self.generation_config.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
            raise ValueError("generation seed must be a nonnegative integer or null")
        # Ignore HTTP_PROXY and .netrc: requests must stay on this node, without credentials.
        self.session = requests.Session()
        self.session.trust_env = False

    def generate(self, system_prompt, user_prompt, temperature=None):
        generation = dict(self.generation_config)
        if temperature is not None:
            generation["temperature"] = float(temperature)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **{key: value for key, value in generation.items() if value is not None},
        }
        response = self.session.post(
            self.base_url + "/chat/completions", json=payload,
            timeout=self.timeout_s, allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise requests.HTTPError("Local endpoint redirects are forbidden", response=response)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Local model returned no text content; inspect its chat template/parser configuration")
        return content
