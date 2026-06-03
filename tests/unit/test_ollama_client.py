from __future__ import annotations

from collections.abc import Mapping

import pytest

from services.shared.exceptions import RAGException
from services.shared.ollama_client import OllamaClient


class FakeTransport:
    def __init__(self, response: Mapping[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, object], float]] = []

    def __call__(
        self,
        url: str,
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        self.calls.append((url, payload, timeout))
        return self.response


def test_embed_posts_to_ollama_embed_endpoint() -> None:
    transport = FakeTransport({"embeddings": [[0.1, 0.2, 0.3]]})
    client = OllamaClient(
        base_url="http://ollama.local/",
        embedding_model="bge-m3",
        timeout=10.0,
        transport=transport,
    )

    vector = client.embed("What is TCP?")

    assert vector == [0.1, 0.2, 0.3]
    assert transport.calls == [
        (
            "http://ollama.local/api/embed",
            {"model": "bge-m3", "input": ["What is TCP?"]},
            10.0,
        )
    ]


def test_embed_batch_returns_one_vector_per_text() -> None:
    transport = FakeTransport({"embeddings": [[0.1], [0.2]]})
    client = OllamaClient(transport=transport)

    vectors = client.embed_batch(["one", "two"])

    assert vectors == [[0.1], [0.2]]


def test_embed_accepts_legacy_single_embedding_field() -> None:
    transport = FakeTransport({"embedding": [0.1, 0.2]})
    client = OllamaClient(transport=transport)

    assert client.embed("single") == [0.1, 0.2]


def test_embed_batch_rejects_wrong_vector_count() -> None:
    transport = FakeTransport({"embeddings": [[0.1]]})
    client = OllamaClient(transport=transport)

    with pytest.raises(RAGException, match="returned 1 embedding"):
        client.embed_batch(["one", "two"])


def test_embed_rejects_non_numeric_values() -> None:
    transport = FakeTransport({"embeddings": [["not-a-number"]]})
    client = OllamaClient(transport=transport)

    with pytest.raises(RAGException, match="not numeric"):
        client.embed("bad")


def test_generate_posts_to_ollama_generate_endpoint() -> None:
    transport = FakeTransport({"response": "TCP is reliable."})
    client = OllamaClient(
        base_url="http://ollama.local",
        generation_model="qwen2.5:3b",
        timeout=20.0,
        transport=transport,
    )

    answer = client.generate("Use these sources...")

    assert answer == "TCP is reliable."
    assert transport.calls == [
        (
            "http://ollama.local/api/generate",
            {
                "model": "qwen2.5:3b",
                "prompt": "Use these sources...",
                "stream": False,
            },
            20.0,
        )
    ]


def test_generate_rejects_missing_response_text() -> None:
    transport = FakeTransport({"done": True})
    client = OllamaClient(transport=transport)

    with pytest.raises(RAGException, match="missing text"):
        client.generate("prompt")


def test_transport_errors_are_wrapped() -> None:
    def broken_transport(
        url: str,
        payload: Mapping[str, object],
        timeout: float,
    ) -> Mapping[str, object]:
        raise RuntimeError("boom")

    client = OllamaClient(transport=broken_transport)

    with pytest.raises(RAGException, match="Ollama request failed"):
        client.generate("prompt")
