"""
ManiBench Evaluation — Inference.net API Client
==================================================
Handles LLM code generation via Inference.net's OpenAI-compatible API.
Supports retries, rate limiting, token tracking, and error handling.

Inference.net endpoint:  https://api.inference.net/v1
Auth:  Bearer <INFERENCE_API_KEY>

The API is OpenAI-compatible, so the request/response schema is the same
as OpenRouter (chat completions).
"""

import time
import json
import re
from typing import Any

import requests

from evaluation.config import (
    INFERENCE_API_KEY,
    INFERENCE_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    ModelSpec,
)


class InferenceNetError(Exception):
    """Raised on unrecoverable Inference.net API errors."""
    pass


class InferenceNetClient:
    """
    Stateless client for Inference.net chat completions.

    Usage:
        client = InferenceNetClient()
        result = client.generate(model_spec, messages)
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or INFERENCE_API_KEY
        if not self.api_key:
            raise InferenceNetError(
                "INFERENCE_API_KEY not set. "
                "Export it or add it to your .env file. "
                "Get a key at https://inference.net"
            )
        self.base_url = INFERENCE_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def generate(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request and return parsed result.

        Returns:
            {
                "content": str,          # Generated text
                "code": str,             # Extracted Python code block
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
                "latency_ms": float,
                "model_id": str,
                "finish_reason": str,
            }
        """
        payload = {
            "model": model.id,
            "messages": messages,
            "temperature": temperature if temperature is not None else model.temperature,
            "max_tokens": max_tokens or model.max_tokens,
            "top_p": model.top_p,
            "stream": False,  # We collect the full response (no streaming)
        }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                if resp.status_code == 429:
                    # Rate limited — back-off and retry
                    wait = RETRY_DELAY * attempt
                    time.sleep(wait)
                    continue

                if resp.status_code != 200:
                    error_body = resp.text[:500]
                    raise InferenceNetError(
                        f"HTTP {resp.status_code}: {error_body}"
                    )

                data = resp.json()

                # Parse response (OpenAI-compatible schema)
                choice = data["choices"][0]
                content = choice["message"]["content"]
                usage = data.get("usage", {})

                return {
                    "content": content,
                    "code": self._extract_code(content),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "latency_ms": latency_ms,
                    "model_id": data.get("model", model.id),
                    "finish_reason": choice.get("finish_reason", "unknown"),
                }

            except (requests.ConnectionError, requests.Timeout) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                continue

        raise InferenceNetError(
            f"Failed after {MAX_RETRIES} attempts: {last_error}"
        )

    @staticmethod
    def _extract_code(content: str) -> str:
        """
        Extract Python code from LLM response.

        Handles:
          - ```python ... ``` blocks
          - ``` ... ``` blocks
          - Raw code (if no code fence found)
        """
        # Try ```python block first
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        # Try generic ``` block
        pattern = r"```\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        # Fallback: look for 'from manim import' as code start
        lines = content.split("\n")
        code_start = None
        for i, line in enumerate(lines):
            if "from manim import" in line or "import manim" in line:
                code_start = i
                break

        if code_start is not None:
            return "\n".join(lines[code_start:]).strip()

        # Last resort: return full content
        return content.strip()

    def list_models(self) -> list[dict]:
        """Fetch available models from Inference.net."""
        resp = self.session.get(
            f"{self.base_url}/models",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])
