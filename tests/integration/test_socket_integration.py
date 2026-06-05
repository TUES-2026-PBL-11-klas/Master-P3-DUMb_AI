"""
Socket integration tests for dummy_server.

These tests boot a real dummy_server on a random port and drive it with
real AIClient connections. They exercise the parts that unit tests with
mocks can't cover:

  - The upload happy path and every rejection branch over a real socket.
  - Multi-user isolation: two users on two connections see their own
    sessions; one cannot affect the other.
  - Concurrent stress: N users sign up, authenticate, and chat in
    parallel without errors or cross-contamination.
  - Server-side enforcement of the session check (bypassing the
    client's local guard to prove the server doesn't trust the client).

The test file is named ``test_socket_integration.py`` rather than mixed
into ``test_auth_socket_e2e.py`` so the suites can be run independently
when iterating on one concern or the other.
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from client.ai_client import AIClient, AIClientError, MAX_UPLOAD_BYTES
from services.dummy_server import (
    SUPPORTED_EXTENSIONS,
    _MemoryUserStore,
    set_document_store,
    set_ingestion_service,
    set_query_service,
    set_user_store,
    start_in_background,
)
from services.shared.domain import QueryResult
from services.shared.exceptions import RAGException


# Fixtures


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeQueryService:
    def ask(self, user_id: object, question: str) -> QueryResult:
        return QueryResult(answer=f"[test-rag] {question}")


class _FakeIngestionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, object]] = []
        self.document_id = uuid.UUID(int=123)

    def ingest(self, filename: str, raw: bytes, user_id: object) -> object:
        self.calls.append((filename, raw, user_id))
        return SimpleNamespace(
            document=SimpleNamespace(id=self.document_id),
            chunks=[object(), object()],
        )


class _FailingIngestionService:
    def ingest(self, filename: str, raw: bytes, user_id: object) -> object:
        raise RAGException("parser exploded")


class _FakeDocumentStore:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def list_by_user(self, user_id: object) -> list[dict[str, object]]:
        self.calls.append(user_id)
        return [
            {
                "document_id": str(uuid.UUID(int=456)),
                "filename": "networking.md",
                "uploaded_at": "2026-06-05T10:15:00",
                "status": "ready",
            }
        ]


@pytest.fixture
def server() -> Generator[tuple[str, int], None, None]:
    # Fresh in-memory user store per test so tests don't see each other.
    set_user_store(_MemoryUserStore())
    set_document_store(None)
    set_query_service(_FakeQueryService())

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_query_service(None)
        set_ingestion_service(None)


@pytest.fixture
def server_with_ingestion() -> Generator[
    tuple[str, int, _FakeIngestionService],
    None,
    None,
]:
    set_user_store(_MemoryUserStore())
    set_document_store(None)
    set_query_service(_FakeQueryService())
    ingestion = _FakeIngestionService()
    set_ingestion_service(ingestion)

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port, ingestion
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_query_service(None)
        set_ingestion_service(None)


@pytest.fixture
def server_with_failing_ingestion() -> Generator[tuple[str, int], None, None]:
    set_user_store(_MemoryUserStore())
    set_document_store(None)
    set_query_service(_FakeQueryService())
    set_ingestion_service(_FailingIngestionService())

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_query_service(None)
        set_ingestion_service(None)


@pytest.fixture
def server_without_query_service() -> Generator[tuple[str, int], None, None]:
    set_user_store(_MemoryUserStore())
    set_document_store(None)
    set_query_service(None)

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_ingestion_service(None)


@pytest.fixture
def server_with_document_store() -> Generator[
    tuple[str, int, _FakeDocumentStore],
    None,
    None,
]:
    set_user_store(_MemoryUserStore())
    store = _FakeDocumentStore()
    set_document_store(store)
    set_query_service(_FakeQueryService())

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port, store
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_query_service(None)
        set_ingestion_service(None)


def _authed_client(
    host: str, port: int, username: str, password: str = "pw"
) -> AIClient:
    """Open a client, sign up, return it ready to send authenticated requests."""
    client = AIClient(host=host, port=port, timeout=10.0)
    client.signup(username, password)
    return client


def _raw_send(host: str, port: int, message: dict[str, object]) -> dict[str, object]:
    """
    Bypass AIClient and send a raw message over a fresh socket.

    Used to prove that the server enforces its own checks, independent
    of any client-side guard.
    """
    with socket.create_connection((host, port), timeout=10.0) as sock:
        sock.sendall((json.dumps(message) + "\n").encode("utf-8"))
        line = sock.makefile("rb").readline()
    decoded = json.loads(line)
    assert isinstance(decoded, dict)
    return decoded


# Document listing


class TestDocumentListing:
    def test_authenticated_user_can_list_documents(
        self,
        server_with_document_store: tuple[str, int, _FakeDocumentStore],
    ) -> None:
        host, port, store = server_with_document_store

        with _authed_client(host, port, "alice") as c:
            documents = c.list_documents()

        assert documents == [
            {
                "document_id": str(uuid.UUID(int=456)),
                "filename": "networking.md",
                "uploaded_at": "2026-06-05T10:15:00",
                "status": "ready",
            }
        ]
        assert len(store.calls) == 1

    def test_raw_document_list_without_auth_rejected(
        self,
        server_with_document_store: tuple[str, int, _FakeDocumentStore],
    ) -> None:
        host, port, _store = server_with_document_store

        reply = _raw_send(host, port, {"type": "documents"})

        assert reply == {"type": "error", "message": "not authenticated"}


# Upload — happy path


class TestUploadHappyPath:
    def test_txt_upload_returns_ack(self, server: tuple[str, int]) -> None:
        host, port = server
        with _authed_client(host, port, "alice") as c:
            ack = c.upload("notes.txt", b"hello over the wire\n")
            assert ack["type"] == "upload_ack"
            assert ack["filename"] == "notes.txt"
            assert ack["size"] == len(b"hello over the wire\n")

    def test_upload_invokes_ingestion_service(
        self,
        server_with_ingestion: tuple[str, int, _FakeIngestionService],
    ) -> None:
        host, port, ingestion = server_with_ingestion
        payload = b"hello over the wire\n"
        with _authed_client(host, port, "alice") as c:
            ack = c.upload("subdir/notes.txt", payload)

            assert ack["type"] == "upload_ack"
            assert ack["filename"] == "notes.txt"
            assert ack["size"] == len(payload)
            assert ack["document_id"] == str(ingestion.document_id)
            assert ack["chunks"] == 2

        assert len(ingestion.calls) == 1
        filename, raw, user_id = ingestion.calls[0]
        assert filename == "notes.txt"
        assert raw == payload
        assert user_id is not None

    def test_upload_surfaces_ingestion_failure(
        self,
        server_with_failing_ingestion: tuple[str, int],
    ) -> None:
        host, port = server_with_failing_ingestion
        with _authed_client(host, port, "alice") as c:
            with pytest.raises(AIClientError, match="ingestion failed"):
                c.upload("notes.txt", b"hello\n")

    def test_md_upload_returns_ack(self, server: tuple[str, int]) -> None:
        host, port = server
        payload = b"# Title\n\nSome **markdown** content.\n"
        with _authed_client(host, port, "alice") as c:
            ack = c.upload("readme.md", payload)
            assert ack["type"] == "upload_ack"
            assert ack["size"] == len(payload)

    def test_basename_is_stripped(self, server: tuple[str, int]) -> None:
        """Client may include a directory; server should only see the basename."""
        host, port = server
        with _authed_client(host, port, "alice") as c:
            ack = c.upload("subdir/notes.txt", b"hi\n")
            assert ack["filename"] == "notes.txt"

    def test_each_supported_extension_accepted(self, server: tuple[str, int]) -> None:
        """Smoke test every alias the server advertises in SUPPORTED_EXTENSIONS."""
        host, port = server
        with _authed_client(host, port, "alice") as c:
            for ext in SUPPORTED_EXTENSIONS:
                ack = c.upload(f"file.{ext}", b"content\n")
                assert ack["type"] == "upload_ack", f"failed for .{ext}"

    def test_many_uploads_on_same_connection(self, server: tuple[str, int]) -> None:
        """A single authenticated client can drive a batch sequentially."""
        host, port = server
        with _authed_client(host, port, "alice") as c:
            for i in range(10):
                ack = c.upload(f"note{i}.txt", f"content {i}\n".encode())
                assert ack["type"] == "upload_ack"
                assert ack["size"] == len(f"content {i}\n")


# Upload — rejections from the server


class TestUploadRejections:
    def test_unsupported_extension(self, server: tuple[str, int]) -> None:
        host, port = server
        with _authed_client(host, port, "alice") as c:
            with pytest.raises(AIClientError, match="unsupported extension"):
                c.upload("virus.exe", b"\x00\x01\x02")

    def test_oversized_payload_rejected_client_side(
        self, server: tuple[str, int]
    ) -> None:
        """Client guards before transmitting — saves the round-trip."""
        host, port = server
        with _authed_client(host, port, "alice") as c:
            with pytest.raises(AIClientError, match="exceeds.*upload cap"):
                c.upload("big.txt", b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_oversized_payload_rejected_server_side(
        self, server: tuple[str, int]
    ) -> None:
        """
        Bypass the client guard with a raw socket and prove the server
        also rejects the payload itself — defence in depth.
        """
        host, port = server
        # Authenticate first on a separate socket so we can keep using it.
        # We need both auth AND oversized payload on the SAME socket.
        with socket.create_connection((host, port), timeout=10.0) as sock:
            f = sock.makefile("rb")
            sock.sendall(
                (
                    json.dumps(
                        {
                            "type": "signup",
                            "username": "alice",
                            "password": "pw",  # pragma: allowlist secret
                        }
                    )
                    + "\n"
                ).encode()
            )
            auth_reply = json.loads(f.readline())
            assert auth_reply["type"] == "auth_ok"

            payload_b64 = base64.b64encode(b"x" * (MAX_UPLOAD_BYTES + 100)).decode(
                "ascii"
            )
            sock.sendall(
                (
                    json.dumps(
                        {
                            "type": "upload",
                            "filename": "big.txt",
                            "bytes_b64": payload_b64,
                        }
                    )
                    + "\n"
                ).encode()
            )
            reply = json.loads(f.readline())
            assert reply["type"] == "error"
            assert "exceeds" in reply["message"]

    def test_malformed_base64(self, server: tuple[str, int]) -> None:
        host, port = server
        # Need to be on the same auth'd socket — _raw_send opens a new one.
        with socket.create_connection((host, port), timeout=10.0) as sock:
            f = sock.makefile("rb")
            sock.sendall(
                (
                    json.dumps(
                        {
                            "type": "signup",
                            "username": "alice",
                            "password": "pw",  # pragma: allowlist secret
                        }
                    )
                    + "\n"
                ).encode()
            )
            f.readline()

            sock.sendall(
                (
                    json.dumps(
                        {
                            "type": "upload",
                            "filename": "notes.txt",
                            "bytes_b64": "@@@not valid base64@@@",
                        }
                    )
                    + "\n"
                ).encode()
            )
            reply = json.loads(f.readline())
            assert reply["type"] == "error"
            assert "base64" in reply["message"]

    def test_missing_filename(self, server: tuple[str, int]) -> None:
        host, port = server
        with socket.create_connection((host, port), timeout=10.0) as sock:
            f = sock.makefile("rb")
            sock.sendall(
                (
                    json.dumps(
                        {
                            "type": "signup",
                            "username": "alice",
                            "password": "pw",  # pragma: allowlist secret
                        }
                    )
                    + "\n"
                ).encode()
            )
            f.readline()

            sock.sendall(
                (json.dumps({"type": "upload", "bytes_b64": "aGk="}) + "\n").encode()
            )
            reply = json.loads(f.readline())
            assert reply["type"] == "error"
            assert "filename" in reply["message"]

    def test_missing_bytes(self, server: tuple[str, int]) -> None:
        host, port = server
        with socket.create_connection((host, port), timeout=10.0) as sock:
            f = sock.makefile("rb")
            sock.sendall(
                (
                    json.dumps(
                        {
                            "type": "signup",
                            "username": "alice",
                            "password": "pw",  # pragma: allowlist secret
                        }
                    )
                    + "\n"
                ).encode()
            )
            f.readline()

            sock.sendall(
                (
                    json.dumps({"type": "upload", "filename": "notes.txt"}) + "\n"
                ).encode()
            )
            reply = json.loads(f.readline())
            assert reply["type"] == "error"
            assert "bytes_b64" in reply["message"]


# Server-side session enforcement


class TestSessionEnforcement:
    def test_raw_upload_without_auth_rejected(self, server: tuple[str, int]) -> None:
        """
        Client lies and sends an upload without auth. Server must still
        reject — the protection lives on both sides of the wire.
        """
        host, port = server
        reply = _raw_send(
            host,
            port,
            {
                "type": "upload",
                "filename": "notes.txt",
                "bytes_b64": "aGk=",
            },
        )
        assert reply["type"] == "error"
        assert "not authenticated" in str(reply["message"])

    def test_raw_query_without_auth_rejected(self, server: tuple[str, int]) -> None:
        host, port = server
        reply = _raw_send(host, port, {"type": "query", "text": "hi"})
        assert reply["type"] == "error"
        assert "not authenticated" in str(reply["message"])

    def test_query_without_query_service_returns_error(
        self, server_without_query_service: tuple[str, int]
    ) -> None:
        host, port = server_without_query_service
        with _authed_client(host, port, "alice") as c:
            with pytest.raises(AIClientError, match="query service"):
                c.ask("hi")

    def test_session_does_not_leak_across_connections(
        self, server: tuple[str, int]
    ) -> None:
        """
        Alice authenticates on connection A. A second, unauthenticated
        connection B must NOT inherit Alice's session.
        """
        host, port = server
        with _authed_client(host, port, "alice"):
            # Connection B is fresh and unauthenticated.
            reply = _raw_send(host, port, {"type": "query", "text": "hi"})
            assert reply["type"] == "error"
            assert "not authenticated" in str(reply["message"])


# Multi-user isolation


class TestMultiUser:
    def test_two_users_separate_sessions(self, server: tuple[str, int]) -> None:
        """Two clients, two users — each sees its own username."""
        host, port = server
        with _authed_client(host, port, "alice") as a:
            with _authed_client(host, port, "bob") as b:
                assert a.username == "alice"
                assert b.username == "bob"
                # Each can ask without contaminating the other.
                a.ask("hi from alice")
                b.ask("hi from bob")
                # Each is still bound to its own user.
                assert a.username == "alice"
                assert b.username == "bob"

    def test_user_cannot_upload_as_another(self, server: tuple[str, int]) -> None:
        """
        Alice and Bob both upload — uploads must be processed by their
        own session, not bleed across.

        Verified indirectly: each upload's ack carries the filename the
        sender chose. If sessions were shared, two parallel uploads
        could ack each other's filenames.
        """
        host, port = server
        with _authed_client(host, port, "alice") as a:
            with _authed_client(host, port, "bob") as b:
                ack_a = a.upload("alice_notes.txt", b"alice's content\n")
                ack_b = b.upload("bob_notes.md", b"# bob's content\n")
                assert ack_a["filename"] == "alice_notes.txt"
                assert ack_b["filename"] == "bob_notes.md"

    def test_duplicate_signup_across_connections(self, server: tuple[str, int]) -> None:
        """
        Two clients race to claim the same username. Exactly one wins.

        This exercises the server's pre-check + fallback-to-duplicate-key
        path in _signup().
        """
        host, port = server

        results: list[tuple[bool, str]] = []
        barrier = threading.Barrier(2)

        def try_signup() -> None:
            barrier.wait()
            client = AIClient(host=host, port=port, timeout=10.0)
            try:
                client.signup("contested", "pw")
                results.append((True, "ok"))
            except AIClientError as exc:
                results.append((False, str(exc)))
            finally:
                client.close()

        t1 = threading.Thread(target=try_signup)
        t2 = threading.Thread(target=try_signup)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        successes = [r for r in results if r[0]]
        failures = [r for r in results if not r[0]]
        assert len(successes) == 1, f"expected 1 winner, got {results}"
        assert len(failures) == 1
        assert "already taken" in failures[0][1]


# Concurrent stress


class TestConcurrentStress:
    def test_many_users_sign_up_and_chat_in_parallel(
        self, server: tuple[str, int]
    ) -> None:
        """
        N concurrent users, each signing up + asking a question. No
        errors, no cross-contamination of replies.

        N=20 is enough to surface races in _MemoryUserStore's lock and
        in the handler's session binding without making the suite slow.
        """
        host, port = server
        N = 20

        def session(i: int) -> tuple[int, str]:
            with AIClient(host=host, port=port, timeout=10.0) as client:
                client.signup(f"user{i}", f"pw{i}")
                # The reply text isn't deterministic, but it should be a
                # non-empty string from the dummy responses pool.
                ans = client.ask(f"I am user {i}")
                return i, client.username, ans  # type: ignore[return-value]

        with ThreadPoolExecutor(max_workers=N) as pool:
            results = list(pool.map(session, range(N)))

        # Every session ran.
        assert len(results) == N

        # Every session is bound to its own user — no cross-contamination.
        for i, username, ans in results:  # type: ignore[misc]
            assert username == f"user{i}", f"session {i} ended up as {username!r}"
            assert isinstance(ans, str) and ans

    def test_many_users_upload_in_parallel(self, server: tuple[str, int]) -> None:
        """
        N concurrent users each uploading a small file. Acks should be
        well-formed and match what each sender claimed.
        """
        host, port = server
        N = 10

        def session(i: int) -> tuple[int, dict]:
            with AIClient(host=host, port=port, timeout=10.0) as client:
                client.signup(f"user{i}", f"pw{i}")
                payload = f"content from user {i}\n".encode()
                ack = client.upload(f"user{i}.txt", payload)
                return i, ack

        with ThreadPoolExecutor(max_workers=N) as pool:
            results = list(pool.map(session, range(N)))

        for i, ack in results:
            assert ack["type"] == "upload_ack"
            assert ack["filename"] == f"user{i}.txt", (
                f"upload from user{i} got ack for {ack['filename']!r}"
            )
            assert ack["size"] == len(f"content from user {i}\n")


# Ping is always allowed


class TestPing:
    def test_ping_does_not_require_auth(self, server: tuple[str, int]) -> None:
        host, port = server
        with AIClient(host=host, port=port, timeout=5.0) as client:
            # No login/signup.
            assert client.ping() is True

    def test_ping_works_after_auth(self, server: tuple[str, int]) -> None:
        host, port = server
        with _authed_client(host, port, "alice") as client:
            assert client.ping() is True
