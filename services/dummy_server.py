#!/usr/bin/env python3
"""
DocChat socket AI server.

A tiny TCP socket server that exposes the backend protocol used by the TUI.
It speaks newline-delimited JSON and delegates real queries to an injected
QueryService.

Protocol
--------
Every message is a single JSON object terminated by a newline (``\\n``).

Client -> Server:
    {"type": "ping"}
    {"type": "login",  "username": "<str>", "password": "<str>"}
    {"type": "signup", "username": "<str>", "password": "<str>"}
    {"type": "logout"}                              # optional; ends the session
    {"type": "query",  "text": "<str>"}             # session-authenticated
    {"type": "upload", "filename": "<str>",         # session-authenticated
                       "bytes_b64": "<base64 str>"}

Server -> Client:
    {"type": "pong"}
    {"type": "auth_ok",    "username": "<str>"}     # reply to login / signup
    {"type": "logout_ok"}
    {"type": "answer",     "text": "<str>"}         # reply to query
    {"type": "upload_ack", "filename": "<str>", "size": <int>}
    {"type": "error",      "message": "<str>"}      # malformed request,
                                                    # auth failure, missing
                                                    # session, or upload
                                                    # rejection.

Session model:
    The connection IS the session. login or signup binds a UserAcc to the
    handler instance; query and upload check that binding and refuse
    otherwise ("not authenticated"). When the socket closes, the session
    evaporates. There are no tokens — the client only needs to know it
    must call login or signup before query / upload.

    Multiple logins on the same connection are allowed; the most recent
    wins (useful for "switch user" without dropping the connection).

Auth design notes:
    - login and signup are separate intents on the wire. A find-or-create
      endpoint silently turns a typo on the login screen into a permanent
      new account; the split makes intent explicit.
    - login does NOT distinguish "user not found" from "incorrect
      password" on the wire — both surface as "invalid credentials" so an
      attacker can't enumerate accounts via the error channel. The login
      path also always runs scrypt (against a dummy hash if the user is
      missing) so the timing channel doesn't leak either.
    - signup necessarily reveals whether a username is taken — there is
      no way to let the user pick a different name otherwise. The
      mitigation for that is rate limiting at the transport layer, not
      message wording.

Upload design note:
    The upload handler is intentionally a *stub* — it validates the
    message shape, base64-decodes the payload, checks the size cap and
    the allow-list of extensions, and returns an ack. It does NOT yet
    feed the bytes into IngestionService (parser → chunk → embed →
    store). Wiring that up is the next step; the protocol is stable.

Run it
------
    python -m services.dummy_server                # 127.0.0.1:5555 by default
    python -m services.dummy_server --port 6000
    python -m services.dummy_server --host 0.0.0.0 --port 5555
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import secrets
import socket
import socketserver
import threading
from typing import Any, Protocol

from services.shared.domain import QueryResult, UserAcc
from services.shared.exceptions import (
    AuthError,
    QueryError,
    RAGException,
    StorageError,
)

# Configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

# Hard upload size cap
# ("Hard upload size cap (default: 5 MB per file)").
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# Extensions the server will accept. In the wired-up server this comes from
# ParserRegistry.supported_extensions; the stub hard-codes the same set so
# the TUI gets the same early-rejection behaviour.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {"txt", "md", "markdown", "mkd", "mkdn", "mdown"}
)

# Single message the wire returns for any login failure. Never include the
# specific reason here — see the auth design note in the module docstring.
_INVALID_CREDENTIALS = "invalid credentials"

logger = logging.getLogger("dummy_server")


# User store plumbing
#
# The handler talks to "something that looks like a user store" via a tiny
# structural Protocol. Production code injects a MongoUserStore; tests can
# inject a MagicMock; if nothing is injected (the default), we fall back to
# the in-memory _MemoryUserStore so the server still boots and is demoable
# without Mongo — matching the offline-stub philosophy in client/tui.py.


class _UserStore(Protocol):
    """Minimal repository surface the auth handler needs."""

    def find_by_username(self, username: str) -> UserAcc | None: ...

    def create(self, username: str, password_hash: str) -> UserAcc: ...


class _QueryService(Protocol):
    """Minimal query service surface the socket handler needs."""

    def ask(self, user_id: object, question: str) -> QueryResult: ...


class _MemoryUserStore:
    """In-memory fallback used when no real store has been injected."""

    def __init__(self) -> None:
        from datetime import datetime, timezone
        from uuid import uuid4

        self._uuid4 = uuid4
        self._now = lambda: datetime.now(timezone.utc)
        self._by_name: dict[str, UserAcc] = {}
        self._lock = threading.Lock()

    def find_by_username(self, username: str) -> UserAcc | None:
        with self._lock:
            return self._by_name.get(username)

    def create(self, username: str, password_hash: str) -> UserAcc:
        with self._lock:
            if username in self._by_name:
                # Mirror what MongoUserStore raises on the unique-index hit
                # so the handler's except-branch behaves the same way.
                raise StorageError(f"username {username!r} is already taken")
            user = UserAcc(
                id=self._uuid4(),
                username=username,
                password_hash=password_hash,
                created_at=self._now(),
            )
            self._by_name[username] = user
            return user


# Module-level hook — replace from main() / tests via set_user_store().
_user_store: _UserStore = _MemoryUserStore()
_query_service: _QueryService | None = None


def set_user_store(store: _UserStore) -> None:
    """Swap the active user store (used by main() and by tests)."""
    global _user_store
    _user_store = store


def set_query_service(service: _QueryService | None) -> None:
    """Swap the active query service, or disable query handling with None."""
    global _query_service
    _query_service = service


# Password hashing
#
# scrypt is in the stdlib (since 3.6) and is a memory-hard KDF designed for
# password storage. We store a self-describing string so we can change the
# parameters later without breaking existing accounts:
#
#     scrypt$<n>$<r>$<p>$<salt-hex>$<hash-hex>
#
# Verification uses hmac.compare_digest to avoid timing leaks.

_SCRYPT_N = 2**14  # CPU/memory cost — ~16 MB of work per hash
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


def _hash_password(password: str) -> str:
    """Return a self-describing scrypt hash string for *password*."""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
        maxmem=64 * 1024 * 1024,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Constant-time check of *password* against a stored scrypt string."""
    try:
        scheme, n_s, r_s, p_s, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    try:
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except ValueError:
        return False
    try:
        dk = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            dklen=len(expected),
            maxmem=64 * 1024 * 1024,
        )
    except ValueError:
        return False
    return hmac.compare_digest(dk, expected)


# Dummy hash used when the requested username does not exist. We still
# run _verify_password against it so the login code path spends the same
# scrypt time whether the user exists or not — closing the timing
# side-channel that would otherwise let an attacker enumerate accounts.
#
# Lazy-built once on first miss; the value is intentionally NOT bound to
# any real password (it's the hash of 32 random bytes that nobody knows).
_DUMMY_HASH: str | None = None
_DUMMY_HASH_LOCK = threading.Lock()


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        with _DUMMY_HASH_LOCK:
            if _DUMMY_HASH is None:
                _DUMMY_HASH = _hash_password(secrets.token_hex(32))
    return _DUMMY_HASH


def _login(username: str, password: str) -> UserAcc:
    """
    Look up *username* and verify *password*.

    On the unhappy paths (unknown user OR wrong password) the function
    raises AuthError("invalid credentials") — a single message — and
    always runs one scrypt verification, so the wire response and the
    handler's response time both reveal nothing about whether the
    username exists.

    Raises:
        AuthError: on any auth failure or empty input. Input-validation
                   errors ("username must not be empty") are kept
                   specific because they're about the request shape, not
                   the user record.
        StorageError: surfaced as-is from the user store.
    """
    if not username:
        raise AuthError("username must not be empty")
    if not password:
        raise AuthError("password must not be empty")

    existing = _user_store.find_by_username(username)
    if existing is None:
        # Burn the same scrypt cycles we'd spend on a real verify so the
        # response time doesn't distinguish "user not found" from "wrong
        # password". The result is discarded.
        _verify_password(password, _dummy_hash())
        raise AuthError(_INVALID_CREDENTIALS)

    if not _verify_password(password, existing.password_hash):
        raise AuthError(_INVALID_CREDENTIALS)

    return existing


def _signup(username: str, password: str) -> UserAcc:
    """
    Create a brand-new account for *username* with *password*.

    Note that signup necessarily reveals whether a username is taken —
    we cannot ask the user to pick a different name otherwise. See the
    module docstring for the rationale.

    Raises:
        AuthError: if the input is empty or the username already exists.
        StorageError: surfaced as-is from the user store on any other
                      backend failure.
    """
    if not username:
        raise AuthError("username must not be empty")
    if not password:
        raise AuthError("password must not be empty")

    # Pre-check returns a cleaner error and avoids a wasted scrypt hash
    # on the common "username taken" case.
    if _user_store.find_by_username(username) is not None:
        raise AuthError("username already taken")

    try:
        return _user_store.create(username, _hash_password(password))
    except StorageError as exc:
        # Race: another connection inserted the same username between our
        # find_by_username() and create() calls. Surface as AuthError so
        # the client sees a consistent message.
        if "already taken" in str(exc):
            raise AuthError("username already taken") from exc
        raise


# Handler
class _DummyAIHandler(socketserver.StreamRequestHandler):
    """
    One instance per client connection. Reads NDJSON, writes NDJSON.

    Per-connection session: ``self._user`` is None until login or signup
    succeeds, then holds the authenticated UserAcc for the rest of the
    connection. query and upload require this; everything else does not.
    """

    # Make sure a slow client doesn't hang the server forever.
    timeout = 300  # seconds

    # Buffered reader's default line length cap is generous, but with base64
    # payloads up to 5 MB our messages can hit ~7 MB. Bump rbufsize so
    # rfile.readline() doesn't truncate.
    rbufsize = 8 * 1024 * 1024  # 8 MB

    def setup(self) -> None:
        super().setup()
        # Session state — see class docstring.
        self._user: UserAcc | None = None

    def handle(self) -> None:
        peer = self.client_address
        logger.info("client connected: %s:%s", *peer)
        try:
            # readline() with a generous max so we can swallow a full 5 MB
            # base64 payload (≈ 7 MB on the wire including JSON overhead).
            max_line = 16 * 1024 * 1024  # 16 MB
            while True:
                raw = self.rfile.readline(max_line)
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                response = self._handle_line(line)
                self._send(response)
        except (ConnectionResetError, BrokenPipeError):
            logger.info("client %s:%s disconnected abruptly", *peer)
        except socket.timeout:
            logger.info("client %s:%s timed out", *peer)
        finally:
            logger.info(
                "client closed: %s:%s (session=%s)",
                *peer,
                self._user.username if self._user else "<none>",
            )

    # -- helpers -------------------------------------------------------------
    def _handle_line(self, line: str) -> dict[str, Any]:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"type": "error", "message": f"invalid JSON: {exc.msg}"}

        if not isinstance(msg, dict):
            return {"type": "error", "message": "message must be a JSON object"}

        kind = msg.get("type")
        if kind == "ping":
            return {"type": "pong"}
        if kind == "login":
            return self._auth("login", msg)
        if kind == "signup":
            return self._auth("signup", msg)
        if kind == "logout":
            return self._logout()
        if kind == "query":
            user = self._require_session()
            if user is None:
                return {"type": "error", "message": "not authenticated"}
            text = str(msg.get("text", "")).strip()
            if not text:
                return {"type": "error", "message": "empty 'text' field"}
            return self._answer(user, text)
        if kind == "upload":
            user = self._require_session()
            if user is None:
                return {"type": "error", "message": "not authenticated"}
            return self._handle_upload(user, msg)
        return {"type": "error", "message": f"unknown type: {kind!r}"}

    def _require_session(self) -> UserAcc | None:
        """Return the bound user, or None if the connection hasn't authenticated."""
        return self._user

    def _auth(self, mode: str, msg: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a login or signup request and bind the result to the
        connection on success.
        """
        username = str(msg.get("username", "")).strip()
        password = str(msg.get("password", ""))
        try:
            if mode == "signup":
                user = _signup(username, password)
            else:
                user = _login(username, password)
        except AuthError as exc:
            # Log the real reason locally for ops, but only surface the
            # safe message to the wire. Note we log the *requested*
            # username, which on login may be an attacker probe — that's
            # fine, the log is internal.
            logger.info("%s denied for %r: %s", mode, username, exc)
            return {"type": "error", "message": str(exc)}
        except StorageError as exc:
            logger.warning("%s storage error for %r: %s", mode, username, exc)
            return {"type": "error", "message": "internal storage error"}

        # Bind the session to the connection.
        self._user = user
        logger.info("%s ok for %r (session bound)", mode, user.username)
        return {"type": "auth_ok", "username": user.username}

    def _logout(self) -> dict[str, Any]:
        if self._user is not None:
            logger.info("logout for %r", self._user.username)
            self._user = None
        return {"type": "logout_ok"}

    def _answer(self, user: UserAcc, text: str) -> dict[str, Any]:
        if _query_service is None:
            logger.warning(
                "query from %s rejected: query service is not configured",
                user.username,
            )
            return {"type": "error", "message": "query service is not configured"}

        try:
            result = _query_service.ask(user.id, text)
        except QueryError as exc:
            logger.warning("query from %s failed: %s", user.username, exc)
            return {"type": "error", "message": str(exc)}
        except Exception as exc:
            logger.exception("query from %s failed unexpectedly", user.username)
            return {"type": "error", "message": f"query failed: {exc}"}

        logger.info(
            "query from %s: %r -> reply len=%d",
            user.username,
            text[:60],
            len(result.answer),
        )
        return {
            "type": "answer",
            "text": result.answer,
            "sources": self._serialize_sources(result),
        }

    @staticmethod
    def _serialize_sources(result: QueryResult) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": str(chunk.doc_id),
                "position": chunk.position,
                "similarity": chunk.similarity,
                "metadata": chunk.metadata,
            }
            for chunk in result.sources
        ]

    def _handle_upload(self, user: UserAcc, msg: dict[str, Any]) -> dict[str, Any]:
        """
        Validate an upload message and return an ack.

        Steps:
          1. Shape validation (required string fields present, non-empty).
          2. Extension allow-list check against SUPPORTED_EXTENSIONS.
          3. Base64 decode (rejects malformed payload).
          4. Size cap check against MAX_UPLOAD_BYTES.

        This is a stub — once IngestionService is wired in, the decoded
        bytes will be passed to it for the parse → chunk → embed → store
        pipeline. The authenticated user.id will be stamped onto the
        resulting Document at that point — see the TODO below.
        """
        filename = msg.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return {"type": "error", "message": "missing or empty 'filename'"}

        bytes_b64 = msg.get("bytes_b64")
        if not isinstance(bytes_b64, str):
            return {"type": "error", "message": "missing 'bytes_b64' field"}

        # Strip any directory component the client may have included.
        filename = os.path.basename(filename)

        # Extension allow-list. Catching unknown extensions here saves the
        # base64 decode for files we'd reject anyway.
        ext = os.path.splitext(filename)[1].lstrip(".").lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return {
                "type": "error",
                "message": (
                    f"unsupported extension '.{ext}' for '{filename}' "
                    f"(supported: {sorted(SUPPORTED_EXTENSIONS)})"
                ),
            }

        # Base64 decode. Use validate=True to catch malformed input fast.
        try:
            raw = base64.b64decode(bytes_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            return {"type": "error", "message": f"invalid base64 payload: {exc}"}

        if len(raw) > MAX_UPLOAD_BYTES:
            return {
                "type": "error",
                "message": (
                    f"upload '{filename}' is {len(raw)} bytes — "
                    f"exceeds the {MAX_UPLOAD_BYTES}-byte cap"
                ),
            }

        # TODO: hand `raw`, `filename`, and `user.id` to
        # IngestionService.ingest(...) here.
        logger.info(
            "upload from %s: '%s' accepted (%d bytes, ext=.%s) [stub: not ingested]",
            user.username,
            filename,
            len(raw),
            ext,
        )

        return {
            "type": "upload_ack",
            "filename": filename,
            "size": len(raw),
        }

    def _send(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()


class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """One thread per client; reuse the address on quick restarts."""

    allow_reuse_address = True
    daemon_threads = True


# Public entry points
def serve_forever(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Run the server in the current thread until Ctrl+C."""
    with _ThreadedServer((host, port), _DummyAIHandler) as server:
        logger.info("dummy AI server listening on %s:%d (Ctrl+C to stop)", host, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("shutting down")
        finally:
            server.shutdown()


def start_in_background(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> tuple[_ThreadedServer, threading.Thread]:
    """
    Start the server on a daemon thread and return (server, thread).

    Useful in tests or when launching alongside the TUI. Call
    ``server.shutdown()`` and ``server.server_close()`` to stop it cleanly.
    """
    server = _ThreadedServer((host, port), _DummyAIHandler)
    thread = threading.Thread(
        target=server.serve_forever, name="dummy-ai-server", daemon=True
    )
    thread.start()
    logger.info("dummy AI server started on %s:%d (background)", host, port)
    return server, thread


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DocChat dummy AI server")
    p.add_argument(
        "--host", default=DEFAULT_HOST, help="bind host (default %(default)s)"
    )
    p.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="bind port (default %(default)s)"
    )
    p.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # If MONGODB_URI is set, wire a real MongoUserStore. Otherwise keep
    # the in-memory fallback so the server still boots for demos.
    mongo_uri = os.environ.get("MONGODB_URI")
    if mongo_uri:
        try:
            from services.db.mongo_user_store import MongoUserStore

            set_user_store(MongoUserStore.from_uri(mongo_uri))
            logger.info("user store: MongoUserStore (%s)", mongo_uri)
        except StorageError as exc:
            logger.warning(
                "could not connect to %s — falling back to in-memory user store: %s",
                mongo_uri,
                exc,
            )
    else:
        logger.info("user store: in-memory (set MONGODB_URI to use Mongo)")

    try:
        from services.query.wiring import build_query_service_from_env

        query_service = build_query_service_from_env()
        if query_service is None:
            set_query_service(None)
            logger.info(
                "query service: disabled "
                "(set RAG_MONGODB_URI or MONGODB_URI to enable)"
            )
        else:
            set_query_service(query_service)
            logger.info(
                "query service: QueryService wired with MongoVectorStore + Ollama"
            )
    except RAGException as exc:
        set_query_service(None)
        logger.warning("query service disabled: %s", exc)

    serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
