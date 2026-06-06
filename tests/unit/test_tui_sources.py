from __future__ import annotations

from collections.abc import Iterator

import pytest

from client.ai_client import AIClientError, AnswerResponse, AnswerSource
from client import tui


@pytest.fixture(autouse=True)
def reset_tui_state() -> Iterator[None]:
    original_client = tui.state.get("ai_client")
    original_status = tui.state.get("ai_status")
    try:
        tui.state["ai_client"] = None
        tui.state["ai_status"] = "offline"
        yield
    finally:
        tui.state["ai_client"] = original_client
        tui.state["ai_status"] = original_status


class FakeClient:
    host = "127.0.0.1"
    port = 5555

    def __init__(
        self,
        response: AnswerResponse | None = None,
        documents: list[dict[str, object]] | None = None,
    ) -> None:
        self.response = response or AnswerResponse(answer="Answer.", sources=[])
        self.documents = documents or []
        self.questions: list[str] = []

    def ask_with_sources(self, text: str) -> AnswerResponse:
        self.questions.append(text)
        return self.response

    def list_documents(self) -> list[dict[str, object]]:
        return self.documents


class FailingClient:
    def ask_with_sources(self, text: str) -> AnswerResponse:
        raise AIClientError("query service unavailable")


def test_format_source_uses_filename_page_chunk_and_score() -> None:
    source = AnswerSource(
        doc_id="doc-1",
        position=4,
        similarity=0.87654,
        metadata={"filename": "networking.pdf", "page": 3},
    )

    assert (
        tui.format_source(source, 1)
        == "[1] networking.pdf | chunk 4 | page 3 | score 0.877"
    )


def test_format_source_falls_back_to_doc_id() -> None:
    source = AnswerSource(doc_id="doc-1", position=0)

    assert tui.format_source(source, 2) == "[2] doc-1 | chunk 0"


def test_fetch_ai_response_returns_online_structured_response() -> None:
    response = AnswerResponse(
        answer="TCP is reliable.",
        sources=[AnswerSource(doc_id="doc-1", position=2)],
    )
    client = FakeClient(response)
    tui.state["ai_client"] = client

    result = tui.fetch_ai_response("What is TCP?")

    assert result == response
    assert client.questions == ["What is TCP?"]
    assert tui.state["ai_status"] == "online"


def test_fetch_ai_response_returns_offline_stub_response() -> None:
    tui.state["ai_client"] = None

    result = tui.fetch_ai_response("What is TCP?")

    assert result.answer.startswith("[offline]")
    assert result.sources == []
    assert tui.state["ai_status"] == "offline"


def test_fetch_ai_response_returns_error_response() -> None:
    tui.state["ai_client"] = FailingClient()

    result = tui.fetch_ai_response("What is TCP?")

    assert result.answer.startswith("[server error]")
    assert "query service unavailable" in result.answer
    assert "offline stub" not in result.answer
    assert result.sources == []
    assert tui.state["ai_status"] == "error: query service unavailable"


def test_refresh_documents_from_server_updates_tui_state() -> None:
    tui.state["ai_client"] = FakeClient(
        documents=[
            {
                "document_id": "doc-1",
                "filename": "networking.md",
                "uploaded_at": "2026-06-05T10:15:00",
                "status": "ready",
            }
        ]
    )

    ok, message = tui.refresh_documents_from_server()

    assert ok
    assert message == ""
    assert tui.state["documents"] == [
        {
            "id": "doc-1",
            "name": "networking.md",
            "size": 0,
            "uploaded_at": "2026-06-05 10:15",
            "status": "ready",
        }
    ]
    assert tui.state["ai_status"] == "online (127.0.0.1:5555)"


def test_refresh_documents_from_server_reports_offline() -> None:
    tui.state["ai_client"] = None

    ok, message = tui.refresh_documents_from_server()

    assert not ok
    assert message == "offline"
