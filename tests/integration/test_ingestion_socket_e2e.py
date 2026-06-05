"""
Socket integration tests for the WIRED upload path in dummy_server.

test_socket_integration.py exercises the validate-only stub path — the
branch taken when no ingestion service is wired. These tests cover the
other branch: when an IngestionService has been injected via
set_ingestion_service(), every accepted upload is fed through ingest()
and the ack/error reflects the pipeline result.

This is the project reference's "full ingestion end-to-end" integration
test (Section 5.2), driven over a real socket with a real AIClient.

Covered:
  - Successful ingestion -> upload_ack carries chunks + doc_id, and the
    authenticated user id reaches ingest() (server never trusts the client)
  - ingest() returning a FAILED event -> server replies with an error
  - ingest() raising RAGException -> server replies with an error
  - the extension allow-list is derived from the WIRED service's registry,
    not the hard-coded stub set — both directions:
      * an extension the wired registry supports is accepted even if it is
        absent from the stub set (no false rejection)
      * an extension the wired registry does NOT support is rejected even
        if it is present in the stub set (no drift / over-acceptance)

A fake IngestionService (returns real IngestionEvent objects, no MongoDB
or llama.cpp) is injected so the socket round-trip is exercised without
the heavy backends. The fixture resets set_ingestion_service(None) on
teardown so this module never leaks the wired state into other suites.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import Callable, Generator
from datetime import datetime, timezone

import pytest

from client.ai_client import AIClient, AIClientError
from services.dummy_server import (
    _MemoryUserStore,
    set_ingestion_service,
    set_user_store,
    start_in_background,
)
from services.shared.domain import (
    Chunk,
    Document,
    DocumentStatus,
    IngestionEvent,
    IngestionStatus,
)
from services.shared.exceptions import RAGException


# Fake pipeline


class _FakeIngestionService:
    """
    Minimal stand-in for IngestionService that dummy_server can drive.

    Exposes exactly the two members _handle_upload touches:
    ``supported_extensions`` and ``ingest()``. The mode controls the
    outcome so each branch of the handler can be tested:

        mode="ok"     -> COMPLETED event with *chunks* chunks
        mode="fail"   -> event.fail(...) (FAILED status, no raise)
        mode="raise"  -> raises RAGException
    """

    def __init__(
        self,
        *,
        extensions: list[str] | None = None,
        mode: str = "ok",
        chunks: int = 3,
    ) -> None:
        self._extensions = extensions if extensions is not None else ["txt", "md"]
        self._mode = mode
        self._chunks = chunks
        self.calls: list[tuple[str, int, uuid.UUID]] = []

    @property
    def supported_extensions(self) -> list[str]:
        return list(self._extensions)

    def ingest(self, filename: str, raw: bytes, user_id: uuid.UUID) -> IngestionEvent:
        self.calls.append((filename, len(raw), user_id))

        if self._mode == "raise":
            raise RAGException("boom in pipeline")

        doc = Document(
            id=uuid.uuid4(),
            user_id=user_id,
            content=raw.decode("utf-8", errors="replace"),
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
        )
        event = IngestionEvent(document=doc)

        if self._mode == "fail":
            event.fail("synthetic failure")
            return event

        event.chunks = [
            Chunk(
                text=f"chunk {i}",
                doc_id=doc.id,
                user_id=user_id,
                position=i,
                embedding=[0.0] * 1024,
            )
            for i in range(self._chunks)
        ]
        event.status = IngestionStatus.COMPLETED
        doc.mark_status(DocumentStatus.READY)
        return event


# Fixtures


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def make_server() -> Generator[
    Callable[[_FakeIngestionService], tuple[str, int]], None, None
]:
    """
    Factory fixture: boot a server wired to a given fake ingestion service.

    Resets BOTH the user store and the ingestion service on teardown so
    the global wiring never leaks into other test modules.
    """
    servers = []

    def _make(service: _FakeIngestionService) -> tuple[str, int]:
        set_user_store(_MemoryUserStore())
        set_ingestion_service(service)
        port = _free_port()
        srv, _thread = start_in_background(host="127.0.0.1", port=port)
        servers.append(srv)
        return "127.0.0.1", port

    try:
        yield _make
    finally:
        set_ingestion_service(None)  # critical — do not leak into other suites
        for srv in servers:
            srv.shutdown()
            srv.server_close()


def _authed_client(host: str, port: int, username: str = "alice") -> AIClient:
    client = AIClient(host=host, port=port, timeout=10.0)
    client.signup(username, "pw")
    return client


# Happy path


class TestWiredUploadSuccess:
    def test_ack_includes_chunk_count(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        svc = _FakeIngestionService(extensions=["md"], chunks=4)
        host, port = make_server(svc)
        with _authed_client(host, port) as c:
            ack = c.upload("notes.md", b"# title\n\nsentence one. sentence two.\n")
        assert ack["type"] == "upload_ack"
        assert ack["chunks"] == 4

    def test_ack_includes_doc_id(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        svc = _FakeIngestionService(extensions=["txt"], chunks=2)
        host, port = make_server(svc)
        with _authed_client(host, port) as c:
            ack = c.upload("notes.txt", b"hello\n")
        assert "doc_id" in ack
        # doc_id is stringified UUID
        uuid.UUID(ack["doc_id"])

    def test_authenticated_user_id_reaches_ingest(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        """The server stamps the session's user id — not anything from the wire."""
        svc = _FakeIngestionService(extensions=["txt"])
        host, port = make_server(svc)
        with _authed_client(host, port, "bob") as c:
            c.upload("notes.txt", b"hello\n")
        assert len(svc.calls) == 1
        _filename, _size, user_id = svc.calls[0]
        assert isinstance(user_id, uuid.UUID)


# Failure paths


class TestWiredUploadFailure:
    def test_failed_event_returns_error(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        svc = _FakeIngestionService(extensions=["txt"], mode="fail")
        host, port = make_server(svc)
        with _authed_client(host, port) as c:
            with pytest.raises(AIClientError, match="ingestion failed"):
                c.upload("notes.txt", b"hello\n")

    def test_raised_exception_returns_error(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        svc = _FakeIngestionService(extensions=["txt"], mode="raise")
        host, port = make_server(svc)
        with _authed_client(host, port) as c:
            with pytest.raises(AIClientError, match="ingestion failed"):
                c.upload("notes.txt", b"hello\n")


# Allow-list derives from the wired registry (drift prevention)


class TestAllowListFromRegistry:
    def test_extension_only_in_wired_registry_is_accepted(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        """
        'rst' is absent from the hard-coded stub set but present in the
        wired registry — the upload must be accepted.
        """
        svc = _FakeIngestionService(extensions=["rst"])
        host, port = make_server(svc)
        with _authed_client(host, port) as c:
            ack = c.upload("notes.rst", b"hello\n")
        assert ack["type"] == "upload_ack"
        assert len(svc.calls) == 1

    def test_extension_not_in_wired_registry_is_rejected(
        self, make_server: Callable[[_FakeIngestionService], tuple[str, int]]
    ) -> None:
        """
        The wired registry supports only 'txt', so a '.md' upload is
        rejected even though 'md' is in the stub set — proving the
        allow-list follows the live registry, not the stale constant.
        """
        svc = _FakeIngestionService(extensions=["txt"])
        host, port = make_server(svc)
        with _authed_client(host, port) as c:
            with pytest.raises(AIClientError, match="unsupported extension"):
                c.upload("notes.md", b"# hi\n")
        # ingest() must never be reached for a rejected extension.
        assert svc.calls == []
