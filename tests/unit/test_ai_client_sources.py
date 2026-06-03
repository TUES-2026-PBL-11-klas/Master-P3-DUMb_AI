from __future__ import annotations

from typing import Any

import pytest

from client.ai_client import AIClient, AIClientError


def _client_with_reply(reply: dict[str, Any]) -> AIClient:
    client = AIClient()
    client._username = "alice"
    client._password = "secret"

    def fake_round_trip(
        message: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        assert message == {"type": "query", "text": "what is tcp?"}
        assert timeout is None
        return reply

    client._authed_round_trip = fake_round_trip  # type: ignore[method-assign]
    return client


def test_ask_with_sources_returns_structured_answer() -> None:
    client = _client_with_reply(
        {
            "type": "answer",
            "text": "TCP is a transport protocol.",
            "sources": [
                {
                    "doc_id": "doc-1",
                    "position": 2,
                    "similarity": 0.91,
                    "metadata": {"page": 3},
                }
            ],
        }
    )

    result = client.ask_with_sources("what is tcp?")

    assert result.answer == "TCP is a transport protocol."
    assert len(result.sources) == 1
    assert result.sources[0].doc_id == "doc-1"
    assert result.sources[0].position == 2
    assert result.sources[0].similarity == 0.91
    assert result.sources[0].metadata == {"page": 3}


def test_ask_stays_backwards_compatible_and_returns_only_text() -> None:
    client = _client_with_reply(
        {
            "type": "answer",
            "text": "TCP is a transport protocol.",
            "sources": [{"doc_id": "doc-1", "position": 2}],
        }
    )

    assert client.ask("what is tcp?") == "TCP is a transport protocol."


def test_ask_with_sources_accepts_missing_sources() -> None:
    client = _client_with_reply(
        {
            "type": "answer",
            "text": "No citations yet.",
        }
    )

    result = client.ask_with_sources("what is tcp?")

    assert result.answer == "No citations yet."
    assert result.sources == []


@pytest.mark.parametrize(
    "sources",
    [
        "not-a-list",
        [{"doc_id": 12, "position": 1}],
        [{"doc_id": "doc-1", "position": "first"}],
        [{"doc_id": "doc-1", "position": 1, "similarity": "high"}],
        [{"doc_id": "doc-1", "position": 1, "metadata": "page 3"}],
    ],
)
def test_ask_with_sources_rejects_invalid_sources(sources: object) -> None:
    client = _client_with_reply(
        {
            "type": "answer",
            "text": "Answer.",
            "sources": sources,
        }
    )

    with pytest.raises(AIClientError):
        client.ask_with_sources("what is tcp?")


def test_ask_with_sources_raises_server_error() -> None:
    client = _client_with_reply(
        {
            "type": "error",
            "message": "query service is unavailable",
        }
    )

    with pytest.raises(AIClientError, match="query service is unavailable"):
        client.ask_with_sources("what is tcp?")
