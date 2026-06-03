from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LlamaCppClient(Protocol):
    """Structural interface for the llama.cpp HTTP embedding client."""

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per entry in *texts*."""
        ...


@runtime_checkable
class GenerationClient(Protocol):
    """Structural interface for a text generation model client."""

    def generate(self, prompt: str) -> str:
        """Return a generated answer for *prompt*."""
        ...
