"""LLM client wrappers (offline-friendly)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

import requests
from requests import RequestException


class BaseLLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


@dataclass
class OllamaClient:
    model: str
    endpoint: str
    temperature: float = 0.0
    max_tokens: int | None = None

    def generate(self, prompt: str) -> str:
        url = f"{self.endpoint.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = resp.read().decode("utf-8")
        except (urllib.error.URLError, ConnectionError) as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.endpoint}. Ensure the service is running."
            ) from exc

        try:
            parsed = json.loads(resp_data)
            return parsed.get("response", "")
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid response from Ollama") from exc


@dataclass
class OpenAICompatibleClient:
    endpoint: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 350
    timeout_seconds: int = 30

    def generate(self, prompt: str) -> str:
        url = f"{self.endpoint.rstrip('/')}/chat/completions"
        system_prompt = "You are a helpful assistant that follows the output format exactly."
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(
                "LM Studio server not reachable. Start Local Server at http://127.0.0.1:1234"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid response from LM Studio server.") from exc

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""
