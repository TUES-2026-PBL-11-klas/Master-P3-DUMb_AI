from __future__ import annotations


class PlatformEmbeddingClient:
    """Stub — replace with the real llama.cpp HTTP client implementation."""

    def __init__(self, base_url: str, *, model: str = "bge-m3") -> None:
        self._base_url = base_url
        self._model = model

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
