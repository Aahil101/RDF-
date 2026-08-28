"""LLM providers — all free.

Four backends behind one interface, resolved by ``VERIRAG_LLM_PROVIDER``:

=============  ===========================================================
``groq``       Free tier, Llama 3.3 70B, extremely fast. Key: console.groq.com
``gemini``     Free tier, Gemini 2.0 Flash. Key: aistudio.google.com/apikey
``ollama``     Fully local, no key, no network. ollama pull llama3.2
``auto``       Try the above in order, then fall back to extractive mode
=============  ===========================================================

If none is reachable, :func:`get_llm` returns ``None`` and the answerer switches
to **extractive mode**, which composes the answer directly from retrieved lines.
That guarantees the demo always works — no key, no quota, no internet.

Only ``requests`` is used, so there is no vendor SDK lock-in and no paid
dependency anywhere in the stack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import requests

from ..config import Settings, get_settings

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(slots=True)
class LLMResponse:
    """Normalised completion result."""

    text: str
    provider: str
    model: str
    latency_ms: int = 0
    error: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and not self.error


@runtime_checkable
class LLM(Protocol):
    provider: str
    model: str

    def available(self) -> bool: ...

    def complete(self, system: str, user: str) -> LLMResponse: ...


# ---------------------------------------------------------------------------
class _HttpLLM:
    """Shared HTTP plumbing: timeouts, bounded retries, error normalisation."""

    provider = "http"

    def __init__(self, model: str, settings: Settings) -> None:
        self.model = model
        self.settings = settings
        self._session = requests.Session()

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str], attempts: int = 3) -> dict[str, Any]:
        last_error = ""
        for attempt in range(attempts):
            try:
                response = self._session.post(
                    url, json=payload, headers=headers, timeout=self.settings.request_timeout
                )
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return response.json()
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < attempts - 1:
                time.sleep(1.5 * (attempt + 1))  # linear backoff, free tiers are rate-limited
        raise RuntimeError(last_error or "request failed")


class GroqLLM(_HttpLLM):
    """Groq free tier via its OpenAI-compatible endpoint."""

    provider = "groq"
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.groq_model, settings)
        self.api_key = settings.groq_api_key

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        try:
            data = self._post(
                self.URL,
                {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": self.settings.temperature,
                    "max_tokens": self.settings.max_tokens,
                },
                {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            )
        except RuntimeError as exc:
            return LLMResponse("", self.provider, self.model, _ms(started), error=str(exc))
        usage = data.get("usage") or {}
        return LLMResponse(
            text=(data["choices"][0]["message"]["content"] or "").strip(),
            provider=self.provider,
            model=self.model,
            latency_ms=_ms(started),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class GeminiLLM(_HttpLLM):
    """Google AI Studio free tier."""

    provider = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.gemini_model, settings)
        self.api_key = settings.gemini_api_key

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        url = f"{self.BASE}/{self.model}:generateContent?key={self.api_key}"
        try:
            data = self._post(
                url,
                {
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {
                        "temperature": self.settings.temperature,
                        "maxOutputTokens": self.settings.max_tokens,
                    },
                },
                {"Content-Type": "application/json"},
            )
        except RuntimeError as exc:
            return LLMResponse("", self.provider, self.model, _ms(started), error=str(exc))
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError):
            return LLMResponse("", self.provider, self.model, _ms(started), error="empty gemini response")
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_ms=_ms(started),
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )


class OllamaLLM(_HttpLLM):
    """Local Ollama server — zero cost, zero network egress."""

    provider = "ollama"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.ollama_model, settings)
        self.base_url = settings.ollama_base_url.rstrip("/")

    def available(self) -> bool:
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=2.5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def complete(self, system: str, user: str) -> LLMResponse:
        started = time.perf_counter()
        try:
            data = self._post(
                f"{self.base_url}/api/chat",
                {
                    "model": self.model,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "options": {
                        "temperature": self.settings.temperature,
                        "num_predict": self.settings.max_tokens,
                    },
                },
                {"Content-Type": "application/json"},
                attempts=1,
            )
        except RuntimeError as exc:
            return LLMResponse("", self.provider, self.model, _ms(started), error=str(exc))
        return LLMResponse(
            text=(data.get("message", {}).get("content") or "").strip(),
            provider=self.provider,
            model=self.model,
            latency_ms=_ms(started),
        )


# ---------------------------------------------------------------------------
def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


_BUILDERS = {"groq": GroqLLM, "gemini": GeminiLLM, "ollama": OllamaLLM}
_AUTO_ORDER = ("groq", "gemini", "ollama")


def get_llm(settings: Settings | None = None, *, probe: bool = True) -> LLM | None:
    """Resolve a usable LLM, or ``None`` to signal extractive mode."""
    settings = settings or get_settings()
    requested = (settings.llm_provider or "auto").strip().lower()

    if requested in {"extractive", "none", "off"}:
        return None

    if requested in _BUILDERS:
        client = _BUILDERS[requested](settings)
        if not probe or client.available():
            return client
        return None

    for name in _AUTO_ORDER:  # auto
        client = _BUILDERS[name](settings)
        if client.available():
            return client
    return None


def describe_provider(llm: LLM | None) -> str:
    return f"{llm.provider}:{llm.model}" if llm is not None else "extractive:no-llm"
