"""
Ollama client for embedding and generation model calls.

The class intentionally exposes the tiny protocol surface used by the rest
of the app:
  - embed(text) for QueryService / EmbeddingObserver
  - embed_batch(texts) for batch chunk embedding
  - generate(prompt) for QueryService answer generation
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from services.shared.exceptions import RAGException

OllamaTransport = Callable[[str, Mapping[str, object], float], Mapping[str, object]]

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_GENERATION_MODEL = "qwen2.5:3b"
DEFAULT_TIMEOUT = 120.0


class OllamaClient:
    """
    Small synchronous HTTP client for Ollama's local API.

    A custom transport can be injected in tests so unit tests do not need
    a running Ollama process.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        generation_model: str = DEFAULT_GENERATION_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: OllamaTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.timeout = timeout
        self._transport = transport or self._default_transport

    def embed(self, text: str) -> list[float]:
        """Return one embedding vector for *text*."""
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector for each text in *texts*."""
        if not texts:
            return []

        payload = {
            "model": self.embedding_model,
            "input": texts,
        }
        response = self._post("/api/embed", payload)
        return self._parse_embeddings(response, expected_count=len(texts))

    def generate(self, prompt: str) -> str:
        """Return a generated answer for *prompt* using the generation model."""
        payload = {
            "model": self.generation_model,
            "prompt": prompt,
            "stream": False,
        }
        response = self._post("/api/generate", payload)
        answer = response.get("response")
        if not isinstance(answer, str):
            raise RAGException(f"Ollama generation response missing text: {response!r}")
        return answer

    def _post(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        url = f"{self.base_url}{path}"
        try:
            return self._transport(url, payload, self.timeout)
        except RAGException:
            raise
        except Exception as exc:
            raise RAGException(f"Ollama request failed for {url}: {exc}") from exc

    @staticmethod
    def _parse_embeddings(
        response: Mapping[str, object],
        *,
        expected_count: int,
    ) -> list[list[float]]:
        raw = response.get("embeddings")

        # Some Ollama-compatible endpoints historically returned a single
        # "embedding" field. Accept it for a one-text request so the client is
        # tolerant while still preferring the modern /api/embed shape.
        if raw is None and expected_count == 1:
            raw = [response.get("embedding")]

        if not isinstance(raw, list):
            raise RAGException(f"Ollama embedding response missing vectors: {response!r}")
        if len(raw) != expected_count:
            raise RAGException(
                f"Ollama returned {len(raw)} embedding(s) for {expected_count} input(s)"
            )

        vectors: list[list[float]] = []
        for vector in raw:
            if not isinstance(vector, list):
                raise RAGException(f"Ollama embedding is not a list: {vector!r}")
            parsed: list[float] = []
            for value in vector:
                if not isinstance(value, (int, float)):
                    raise RAGException(
                        f"Ollama embedding value is not numeric: {value!r}"
                    )
                parsed.append(float(value))
            vectors.append(parsed)

        return vectors

    @staticmethod
    def _default_transport(
        url: str,
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RAGException(f"Could not reach Ollama at {url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise RAGException(f"Ollama returned invalid JSON from {url}: {exc}") from exc

        if not isinstance(decoded, dict):
            raise RAGException(f"Ollama returned non-object JSON: {decoded!r}")
        return decoded
