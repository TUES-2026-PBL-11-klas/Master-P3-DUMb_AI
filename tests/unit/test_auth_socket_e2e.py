"""
End-to-end socket test for the auth flow

Boots the real dummy_server on a random port, talks to it with a real
AIClient, and checks the full round-trip:
  - First auth creates the account.
  - Second auth with the same password succeeds (no recreation).
  - Auth with a wrong password raises AIClientError("incorrect password").
  - The pre-existing ping/query channels still work.
"""

from __future__ import annotations

import socket

import pytest

from client.ai_client import AIClient, AIClientError
from services.dummy_server import (
    _MemoryUserStore,
    set_user_store,
    start_in_background,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    # Fresh in-memory user store per test so tests don't see each other.
    set_user_store(_MemoryUserStore())

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port
    finally:
        srv.shutdown()
        srv.server_close()


def test_auth_creates_then_signs_in(server) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        assert client.authenticate("alice", "hunter2") is True   # created
        assert client.authenticate("alice", "hunter2") is False  # signed in


def test_auth_rejects_wrong_password(server) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        client.authenticate("alice", "hunter2")
        with pytest.raises(AIClientError, match="incorrect password"):
            client.authenticate("alice", "wrong-one")


def test_auth_rejects_empty_credentials(server) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        with pytest.raises(AIClientError, match="username"):
            client.authenticate("", "hunter2")
        with pytest.raises(AIClientError, match="password"):
            client.authenticate("alice", "")


def test_ping_and_query_still_work(server) -> None:
    host, port = server
    with AIClient(host=host, port=port, timeout=5.0) as client:
        assert client.ping() is True
        answer = client.ask("anything?", username="alice")
        assert isinstance(answer, str) and answer