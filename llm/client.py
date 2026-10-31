"""Single, local, OpenAI-agnostic LLM client for Ollama.

Everything in the app talks to the model through here, so there is exactly one
place that knows about HTTP, retries, timeouts and token accounting -- and no
`subprocess.run(["ollama", ...])` anywhere (that approach spawns a process per
call and cannot stream or report timing).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_BASE = "http://127.0.0.1:11434"


def _normalise_base(url: str | None) -> str:
    """Accept OLLAMA_URL=http://host:11434/api/generate and turn it into a base."""
    raw = (url or os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_URL") or DEFAULT_BASE).strip()
    for suffix in ("/api/generate", "/api/chat", "/api/tags"):
        if raw.endswith(suffix):
            raw = raw[: -len(suffix)]
    return raw.rstrip("/")


class LLMError(RuntimeError):
    """A model call failed in a way the caller should surface to the user."""


@dataclass
class LLMResult:
    text: str
    model: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds else 0.0

    def __str__(self) -> str:
        return (
            f"{self.model} {self.seconds:.1f}s "
            f"{self.output_tokens} tok ({self.tokens_per_second:.0f} tok/s)"
        )


class OllamaClient:
    def __init__(self, base_url: str | None = None, timeout: float = 300.0, retries: int = 2):
        self.base_url = _normalise_base(base_url)
        self.timeout = float(os.environ.get("LLM_TIMEOUT", timeout))
        self.retries = retries
        self.session = requests.Session()

    # -- low level ---------------------------------------------------------

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(url, json=payload, timeout=timeout or self.timeout)
                if response.status_code == 404:
                    model = payload.get("model", "?")
                    raise LLMError(
                        f"Model '{model}' is not installed in Ollama. "
                        f"Run: ollama pull {model}"
                    )
                response.raise_for_status()
                return response.json()
            except LLMError:
                raise
            except requests.exceptions.ConnectionError as exc:
                last_error = LLMError(
                    f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`."
                )
                if attempt == self.retries:
                    raise last_error from exc
            except requests.exceptions.Timeout as exc:
                last_error = LLMError(f"Ollama timed out after {timeout or self.timeout:.0f}s.")
                if attempt == self.retries:
                    raise last_error from exc
            except requests.exceptions.HTTPError as exc:
                body = ""
                try:
                    body = exc.response.text[:300]
                except Exception:
                    pass
                last_error = LLMError(f"Ollama returned HTTP {exc.response.status_code}: {body}")
                if attempt == self.retries:
                    raise last_error from exc

            time.sleep(1.5 * (attempt + 1))

        raise last_error or LLMError("unknown Ollama failure")

    # -- public api --------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        *,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 1200,
        json_schema: dict | None = None,
        think: bool = False,
        keep_alive: str | int = "30m",
        timeout: float | None = None,
    ) -> LLMResult:
        model = model or os.environ.get("SQL_MODEL") or "qwen3.6:latest"

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
        }

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think,
            "keep_alive": keep_alive,
            "options": options,
        }
        if json_schema:
            payload["format"] = json_schema

        started = time.time()
        data = self._post("/api/chat", payload, timeout=timeout)
        elapsed = time.time() - started

        message = data.get("message") or {}
        text = message.get("content", "") or ""

        return LLMResult(
            text=text,
            model=data.get("model", model),
            seconds=elapsed,
            prompt_tokens=data.get("prompt_eval_count", 0) or 0,
            output_tokens=data.get("eval_count", 0) or 0,
            raw=data,
        )

    def generate(self, prompt: str, model: str | None = None, **kwargs) -> LLMResult:
        return self.chat([{"role": "user", "content": prompt}], model=model, **kwargs)

    def stream_chat(self, messages: list[dict], model: str | None = None, **kwargs):
        """Yield text deltas. Used by the UI; not used by batch jobs."""
        model = model or os.environ.get("SQL_MODEL") or "qwen3.6:latest"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": kwargs.get("think", False),
            "keep_alive": kwargs.get("keep_alive", "30m"),
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_ctx": kwargs.get("num_ctx", 8192),
                "num_predict": kwargs.get("num_predict", 1200),
            },
        }
        with self.session.post(
            f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("message") or {}).get("content")
                if delta:
                    yield delta

    # -- introspection -----------------------------------------------------

    def models(self) -> list[dict]:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception:
            return []

    def is_up(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            return response.ok
        except Exception:
            return False

    def warm(self, model: str | None = None) -> None:
        """Load the model into memory so the first real request isn't cold."""
        model = model or os.environ.get("SQL_MODEL") or "qwen3.6:latest"
        try:
            self.chat([{"role": "user", "content": "ok"}], model=model, num_predict=1)
        except LLMError:
            pass


_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
