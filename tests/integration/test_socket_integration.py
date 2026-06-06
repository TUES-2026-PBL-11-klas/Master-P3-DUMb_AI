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

import pytest

from client.ai_client import AIClient, AIClientError, MAX_UPLOAD_BYTES
from services.dummy_server import (
    SUPPORTED_EXTENSIONS,
    _MemoryUserStore,
    set_document_store,
    set_vector_store,
    set_user_store,
    start_in_background,
)


# Fixtures


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.doc_owners: dict[str, str] = {}
        self.delete_calls: list[tuple[str, str | None]] = []

    def seed(self, doc_id: str, user_id: object) -> None:
        self.doc_owners[doc_id] = str(user_id)

    def delete_document(self, doc_id: object, *, user_id: object | None = None) -> int:
        owner = str(user_id) if user_id is not None else None
        doc_key = str(doc_id)
        self.delete_calls.append((doc_key, owner))
        if self.doc_owners.get(doc_key) != owner:
            return 0
        del self.doc_owners[doc_key]
        return 2


class _FakeDocumentStore:
    def __init__(self, vector_store: _FakeVectorStore | None = None) -> None:
        self.calls: list[object] = []
        self.delete_calls: list[tuple[str, str]] = []
        self._vector_store = vector_store
        self._docs_by_user: dict[str, list[dict[str, object]]] = {}

    def upsert(self, document: object) -> None:
        pass

    def list_by_user(self, user_id: object) -> list[dict[str, object]]:
        self.calls.append(user_id)
        user_key = str(user_id)
        if user_key not in self._docs_by_user:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc:{user_key}"))
            self._docs_by_user[user_key] = [
                {
                    "document_id": doc_id,
                    "filename": "networking.md",
                    "uploaded_at": "2026-06-05T10:15:00",
                    "status": "ready",
                }
            ]
            if self._vector_store is not None:
                self._vector_store.seed(doc_id, user_key)
        return list(self._docs_by_user[user_key])

    def delete(self, user_id: object, doc_id: object) -> bool:
        user_key = str(user_id)
        doc_key = str(doc_id)
        self.delete_calls.append((user_key, doc_key))
        docs = self._docs_by_user.get(user_key, [])
        remaining = [doc for doc in docs if doc.get("document_id") != doc_key]
        if len(remaining) == len(docs):
            return False
        self._docs_by_user[user_key] = remaining
        return True


@pytest.fixture
def server() -> Generator[tuple[str, int], None, None]:
    # Fresh in-memory user store per test so tests don't see each other.
    set_user_store(_MemoryUserStore())
    set_document_store(None)
    set_vector_store(None)

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_vector_store(None)


@pytest.fixture
def server_with_document_store() -> Generator[
    tuple[str, int, _FakeDocumentStore, _FakeVectorStore],
    None,
    None,
]:
    set_user_store(_MemoryUserStore())
    vector_store = _FakeVectorStore()
    store = _FakeDocumentStore(vector_store)
    set_document_store(store)
    set_vector_store(vector_store)

    port = _free_port()
    srv, _thread = start_in_background(host="127.0.0.1", port=port)
    try:
        yield "127.0.0.1", port, store, vector_store
    finally:
        srv.shutdown()
        srv.server_close()
        set_document_store(None)
        set_vector_store(None)


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
        server_with_document_store: tuple[
            str, int, _FakeDocumentStore, _FakeVectorStore
        ],
    ) -> None:
        host, port, store, _vector_store = server_with_document_store

        with _authed_client(host, port, "alice") as c:
            documents = c.list_documents()

        assert len(documents) == 1
        assert uuid.UUID(str(documents[0]["document_id"]))
        assert documents[0]["filename"] == "networking.md"
        assert documents[0]["uploaded_at"] == "2026-06-05T10:15:00"
        assert documents[0]["status"] == "ready"
        assert len(store.calls) == 1

    def test_raw_document_list_without_auth_rejected(
        self,
        server_with_document_store: tuple[
            str, int, _FakeDocumentStore, _FakeVectorStore
        ],
    ) -> None:
        host, port, _store, _vector_store = server_with_document_store

        reply = _raw_send(host, port, {"type": "documents"})

        assert reply == {"type": "error", "message": "not authenticated"}


# Document deletion


class TestDocumentDeletion:
    def test_raw_delete_without_auth_rejected(
        self,
        server_with_document_store: tuple[
            str, int, _FakeDocumentStore, _FakeVectorStore
        ],
    ) -> None:
        host, port, _store, _vector_store = server_with_document_store

        reply = _raw_send(host, port, {"type": "delete_document", "doc_id": "doc-1"})

        assert reply == {"type": "error", "message": "not authenticated"}

    def test_authenticated_user_deletes_own_document_and_chunks(
        self,
        server_with_document_store: tuple[
            str, int, _FakeDocumentStore, _FakeVectorStore
        ],
    ) -> None:
        host, port, store, vector_store = server_with_document_store

        with _authed_client(host, port, "alice") as client:
            documents = client.list_documents()
            doc_id = str(documents[0]["document_id"])

            reply = client.delete_document(doc_id)
            remaining = client.list_documents()

        assert reply == {"type": "delete_ok", "doc_id": doc_id, "deleted_chunks": 2}
        assert remaining == []
        assert store.delete_calls[-1][1] == doc_id
        assert vector_store.delete_calls[-1][0] == doc_id

    def test_user_cannot_delete_another_users_document(
        self,
        server_with_document_store: tuple[
            str, int, _FakeDocumentStore, _FakeVectorStore
        ],
    ) -> None:
        host, port, store, vector_store = server_with_document_store

        with _authed_client(host, port, "alice") as alice:
            alice_doc_id = str(alice.list_documents()[0]["document_id"])
            alice_user_id = store.calls[-1]

        with _authed_client(host, port, "bob") as bob:
            with pytest.raises(AIClientError, match="document not found"):
                bob.delete_document(alice_doc_id)

        assert store.delete_calls[-1][1] == alice_doc_id
        assert vector_store.delete_calls[-1] == (alice_doc_id, store.delete_calls[-1][0])

        alice_docs = store.list_by_user(alice_user_id)
        assert alice_docs[0]["document_id"] == alice_doc_id


# Upload — happy path


class TestUploadHappyPath:
    def test_txt_upload_returns_ack(self, server: tuple[str, int]) -> None:
        host, port = server
        with _authed_client(host, port, "alice") as c:
            ack = c.upload("notes.txt", b"hello over the wire\n")
            assert ack["type"] == "upload_ack"
            assert ack["filename"] == "notes.txt"
            assert ack["size"] == len(b"hello over the wire\n")

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
