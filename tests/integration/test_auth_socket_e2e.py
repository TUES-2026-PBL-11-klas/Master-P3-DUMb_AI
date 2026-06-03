"""
End-to-end socket test for the auth flow with per-connection sessions.

Boots the real dummy_server on a random port, talks to it with a real
AIClient, and checks the full round-trip:
  - signup binds a session; ask() works without re-supplying credentials.
  - login binds a session; same.
  - signup of the same username again fails with "already taken".
  - login with the wrong password fails with "invalid credentials".
  - login of an unknown user fails with "invalid credentials" — same message.
  - query / upload before authentication fails with "not authenticated".
  - logout clears the session; subsequent query fails with "not authenticated".
  - Re-login after a forced socket reconnect succeeds transparently.
"""

from __future__ import annotations

import socket
from collections.abc import Generator

import pytest

from client.ai_client import AIClient, AIClientError
from services.dummy_server import (
    _MemoryUserStore,
    set_query_service,
    set_user_store,
    start_in_background,
)
from services.shared.domain import QueryResult


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeQueryService:
    def ask(self, user_id: object, question: str) -> QueryResult:
        return QueryResult(answer=f"[test-rag] {question}")


@pytest.fixture
def server() -> Generator[tuple[str, int], None, None]:
    # Fresh in-memory user store per test so tests don't see each other.
    set_user_store(_MemoryUserStore())
    set_query_service(_FakeQueryService())

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port
    finally:
        srv.shutdown()
        srv.server_close()
        set_query_service(None)


# Happy paths


def test_signup_binds_session(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        assert client.signup("alice", "hunter2") == "alice"
        assert client.is_authenticated
        assert client.username == "alice"
        # No username on the wire — server stamps from the session.
        answer = client.ask("any thoughts?")
        assert isinstance(answer, str) and answer


def test_login_binds_session(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        client.signup("alice", "hunter2")
        client.logout()
    with AIClient(host=host, port=port, timeout=5.0) as client:
        assert client.login("alice", "hunter2") == "alice"
        assert client.is_authenticated


# Auth failures


def test_signup_rejects_existing_username(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        client.signup("alice", "hunter2")
    with AIClient(host=host, port=port, timeout=5.0) as client2:
        with pytest.raises(AIClientError, match="already taken"):
            client2.signup("alice", "different")


def test_login_invalid_credentials_does_not_leak_existence(
    server: tuple[str, int],
) -> None:
    """Both unknown-user and wrong-password emit byte-identical errors."""
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        client.signup("alice", "hunter2")

    with AIClient(host=host, port=port, timeout=5.0) as c1:
        with pytest.raises(AIClientError) as wrong_pw:
            c1.login("alice", "wrong-one")

    with AIClient(host=host, port=port, timeout=5.0) as c2:
        with pytest.raises(AIClientError) as unknown:
            c2.login("ghost", "hunter2")

    # Same message, no enumeration channel.
    assert str(wrong_pw.value) == str(unknown.value)
    assert "invalid credentials" in str(wrong_pw.value)


def test_login_rejects_empty_credentials(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        with pytest.raises(AIClientError, match="username"):
            client.login("", "hunter2")
        with pytest.raises(AIClientError, match="password"):
            client.login("alice", "")


def test_signup_rejects_empty_credentials(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        with pytest.raises(AIClientError, match="username"):
            client.signup("", "hunter2")
        with pytest.raises(AIClientError, match="password"):
            client.signup("alice", "")


# Session enforcement


def test_query_requires_authentication(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        # Don't call login/signup — client.ask() guards locally.
        with pytest.raises(AIClientError, match="not authenticated"):
            client.ask("hello?")


def test_upload_requires_authentication(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        with pytest.raises(AIClientError, match="not authenticated"):
            client.upload("notes.txt", b"hi\n")


def test_server_rejects_unauthenticated_query_directly(server: tuple[str, int]) -> None:
    """
    Bypass the client's local guard and send a raw query — the server's
    own session check must catch it.
    """
    host, port = server
    import json

    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.sendall((json.dumps({"type": "query", "text": "hi"}) + "\n").encode())
        reply = json.loads(sock.makefile("rb").readline())
        assert reply["type"] == "error"
        assert "not authenticated" in reply["message"]


def test_logout_clears_session(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        client.signup("alice", "hunter2")
        client.ask("ok?")  # works
        client.logout()
        assert not client.is_authenticated
        with pytest.raises(AIClientError, match="not authenticated"):
            client.ask("after logout?")


# Transparent re-auth on reconnect


def test_reconnect_re_runs_login_transparently(server: tuple[str, int]) -> None:
    """
    If the underlying socket drops between two ask() calls, the next
    ask() should still succeed — the client re-runs the cached login
    on the new socket.
    """
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        client.signup("alice", "hunter2")
        client.ask("first?")

        # Force a "stale socket" condition. closes the current socket
        # while keeping cached credentials in place.
        client._close_locked()  # type: ignore[attr-defined]

        # This should silently reconnect, re-bind the session, and answer.
        answer = client.ask("after reconnect?")
        assert isinstance(answer, str) and answer
        assert client.is_authenticated


# Ping is always allowed


def test_ping_works_without_auth(server: tuple[str, int]) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        assert client.ping() is True
