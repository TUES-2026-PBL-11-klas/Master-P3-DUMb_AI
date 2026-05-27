#!/usr/bin/env python3
"""
DocChat dummy AI server.

A tiny TCP socket server that pretends to be the AI backend. It speaks a
newline-delimited JSON protocol so the TUI (or any other client) can be
developed and demoed before the real RAG pipeline is wired up.

Protocol
--------
Every message is a single JSON object terminated by a newline (``\\n``).

Client -> Server:
    {"type": "auth",   "username": "<str>", "password": "<str>"}
    {"type": "query",  "username": "<str>", "text": "<str>"}
    {"type": "ping"}

Server -> Client:
    {"type": "auth_ok",    "username": "<str>", "created": <bool>}
                                                   # created=True if the
                                                   # account was just
                                                   # added; False if it
                                                   # already existed and
                                                   # the password matched.
    {"type": "answer", "text": "<str>"}            # reply to "query"
    {"type": "pong"}                               # reply to "ping"
    {"type": "error",  "message": "<str>"}         # malformed request OR
                                                   # auth failure (wrong
                                                   # password)

Run it
------
    python -m services.dummy_server                # 127.0.0.1:5555 by default
    python -m services.dummy_server --port 6000
    python -m services.dummy_server --host 0.0.0.0 --port 5555
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import random
import secrets
import socket
import socketserver
import threading
import time
from typing import Any, Protocol

from services.shared.domain import UserAcc
from services.shared.exceptions import AuthError, StorageError

# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

# Simulated "thinking" delay so the UI feels like it's talking to a real model.
MIN_THINK_SECONDS = 0.2
MAX_THINK_SECONDS = 0.8

DUMMY_RESPONSES = [
    "Based on the documents you uploaded, the short answer is: yes, but with caveats.",
    "I scanned the relevant chunks and the most likely answer is somewhere on page 3.",
    "Good question. The documents suggest three possible interpretations -- want me to enumerate them?",
    "I couldn't find a definitive answer, but the closest match talks about exactly this topic.",
    "According to your uploaded files, the relevant section explicitly addresses this.",
    "That's outside what your documents cover. Try uploading more material on the topic.",
    "The documents mention this in passing -- I'd recommend re-checking the source for nuance.",
    "Yes. The evidence in your corpus points clearly in that direction.",
    "No, your documents actually argue the opposite, citing a specific case study.",
    "I'd summarize the answer in three bullet points if you'd like a quick overview.",
]

logger = logging.getLogger("dummy_server")


# ── User store plumbing ──────────────────────────────────────────────────────
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


def set_user_store(store: _UserStore) -> None:
    """Swap the active user store (used by main() and by tests)."""
    global _user_store
    _user_store = store


# ── Password hashing ─────────────────────────────────────────────────────────
#
# scrypt is in the stdlib (since 3.6) and is a memory-hard KDF designed for
# password storage. We store a self-describing string so we can change the
# parameters later without breaking existing accounts:
#
#     scrypt$<n>$<r>$<p>$<salt-hex>$<hash-hex>
#
# Verification uses hmac.compare_digest to avoid timing leaks.

_SCRYPT_N = 2 ** 14   # CPU/memory cost — ~16 MB of work per hash
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
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
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
            salt=salt, n=n, r=r, p=p, dklen=len(expected),
            maxmem=64 * 1024 * 1024,
        )
    except ValueError:
        return False
    return hmac.compare_digest(dk, expected)


def _authenticate(username: str, password: str) -> tuple[UserAcc, bool]:
    """
    Find-or-create the user and verify the password.

    Returns (user, created) where ``created`` is True if a new account was
    just inserted, False if the user already existed and the password
    matched.

    Raises:
        AuthError: if the username exists but the password is wrong, or
                   if the input is empty.
        StorageError: surfaced as-is from the user store.
    """
    if not username:
        raise AuthError("username must not be empty")
    if not password:
        raise AuthError("password must not be empty")

    existing = _user_store.find_by_username(username)
    if existing is not None:
        if not _verify_password(password, existing.password_hash):
            raise AuthError("incorrect password")
        return existing, False

    # New account — hash the password before it ever touches the store.
    user = _user_store.create(username, _hash_password(password))
    return user, True


# ── Handler ──────────────────────────────────────────────────────────────────
class _DummyAIHandler(socketserver.StreamRequestHandler):
    """One instance per client connection. Reads NDJSON, writes NDJSON."""

    # Make sure a slow client doesn't hang the server forever.
    timeout = 300  # seconds

    def handle(self) -> None:
        peer = self.client_address
        logger.info("client connected: %s:%s", *peer)
        try:
            for raw in self.rfile:
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
            logger.info("client closed: %s:%s", *peer)

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
        if kind == "auth":
            username = str(msg.get("username", "")).strip()
            password = str(msg.get("password", ""))
            return self._auth(username, password)
        if kind == "query":
            text = str(msg.get("text", "")).strip()
            username = str(msg.get("username", "anonymous"))
            if not text:
                return {"type": "error", "message": "empty 'text' field"}
            return self._answer(username, text)
        return {"type": "error", "message": f"unknown type: {kind!r}"}

    def _auth(self, username: str, password: str) -> dict[str, Any]:
        try:
            user, created = _authenticate(username, password)
        except AuthError as exc:
            logger.info("auth denied for %r: %s", username, exc)
            return {"type": "error", "message": str(exc)}
        except StorageError as exc:
            logger.warning("auth storage error for %r: %s", username, exc)
            return {"type": "error", "message": "internal storage error"}
        logger.info("auth ok for %r (created=%s)", user.username, created)
        return {"type": "auth_ok", "username": user.username, "created": created}

    def _answer(self, username: str, text: str) -> dict[str, Any]:
        # Pretend to think.
        time.sleep(random.uniform(MIN_THINK_SECONDS, MAX_THINK_SECONDS))
        body = random.choice(DUMMY_RESPONSES)
        # Echo a hint of the question so it feels less robotic.
        preview = text if len(text) <= 60 else text[:57] + "..."
        answer = f"[dummy] {body}  (re: \"{preview}\")"
        logger.info("query from %s: %r -> reply len=%d", username, preview, len(answer))
        return {"type": "answer", "text": answer}

    def _send(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()


class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """One thread per client; reuse the address on quick restarts."""

    allow_reuse_address = True
    daemon_threads = True


# ── Public entry points ──────────────────────────────────────────────────────
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
    p.add_argument("--host", default=DEFAULT_HOST, help="bind host (default %(default)s)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help="bind port (default %(default)s)")
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
                mongo_uri, exc,
            )
    else:
        logger.info("user store: in-memory (set MONGODB_URI to use Mongo)")

    serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()