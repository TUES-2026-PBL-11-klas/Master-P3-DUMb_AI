"""
Socket client for talking to the DocChat dummy AI server.

Encapsulates the newline-delimited JSON protocol so the TUI stays focused on
rendering. Reconnects automatically on the next call after a dropped socket.
"""

from __future__ import annotations

import json
import socket
import threading
from typing import Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT = 10.0  # seconds for a single request/response round-trip


class AIClientError(RuntimeError):
    """Raised when the client cannot complete a request."""


class AIClient:
    """
    Thin synchronous client for the dummy AI server.

    Thread-safe at the request level (one request at a time per client),
    which is all the TUI needs since it sends a message and then waits.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._rfile = None  # buffered reader for line-based reads
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise AIClientError(
                f"cannot connect to AI server at {self.host}:{self.port} ({exc})"
            ) from exc
        sock.settimeout(self.timeout)
        self._sock = sock
        self._rfile = sock.makefile("rb")

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._rfile is not None:
            try:
                self._rfile.close()
            except OSError:
                pass
            self._rfile = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def __enter__(self) -> "AIClient":
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- public API ----------------------------------------------------------
    def ping(self) -> bool:
        """Return True if the server responds with a pong."""
        try:
            reply = self._round_trip({"type": "ping"})
        except AIClientError:
            return False
        return reply.get("type") == "pong"

    def ask(self, text: str, username: str = "anonymous") -> str:
        """
        Send a query and return the AI's answer text.

        Raises ``AIClientError`` on any transport or protocol failure.
        """
        reply = self._round_trip({"type": "query", "username": username, "text": text})
        if reply.get("type") == "answer":
            answer = reply.get("text", "")
            if not isinstance(answer, str):
                raise AIClientError(f"non-string answer field: {answer!r}")
            return answer
        if reply.get("type") == "error":
            raise AIClientError(f"server error: {reply.get('message', 'unknown')}")
        raise AIClientError(f"unexpected reply type: {reply.get('type')!r}")

    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate (or register) *username* with *password*.

        Semantics mirror the server's ``auth`` handler:
          - If the username does not exist, it is created and stored.
          - If it does exist, the password must match the stored hash.

        Returns:
            True if the account was just created, False if an existing
            account's password matched.

        Raises:
            AIClientError: on transport failure, malformed reply, OR on
                           a failed auth (e.g. wrong password). Callers
                           that want to distinguish "wrong password" from
                           "server unreachable" should inspect the
                           exception message — the server returns
                           ``"incorrect password"`` verbatim.
        """
        reply = self._round_trip({
            "type": "auth",
            "username": username,
            "password": password,
        })
        if reply.get("type") == "auth_ok":
            return bool(reply.get("created", False))
        if reply.get("type") == "error":
            raise AIClientError(str(reply.get("message", "auth failed")))
        raise AIClientError(f"unexpected reply type: {reply.get('type')!r}")

    # -- internals -----------------------------------------------------------
    def _round_trip(self, message: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            # First attempt; reconnect once if the socket is stale.
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        # connect() takes the lock too -- inline to avoid recursion.
                        self._connect_locked()
                    self._send_locked(message)
                    return self._recv_locked()
                except (OSError, AIClientError) as exc:
                    self._close_locked()
                    if attempt == 2:
                        raise AIClientError(str(exc)) from exc
            # Unreachable, but mypy can't tell.
            raise AIClientError("round-trip failed")

    def _connect_locked(self) -> None:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise AIClientError(
                f"cannot connect to AI server at {self.host}:{self.port} ({exc})"
            ) from exc
        sock.settimeout(self.timeout)
        self._sock = sock
        self._rfile = sock.makefile("rb")

    def _send_locked(self, message: dict[str, Any]) -> None:
        assert self._sock is not None
        payload = (json.dumps(message) + "\n").encode("utf-8")
        self._sock.sendall(payload)

    def _recv_locked(self) -> dict[str, Any]:
        assert self._rfile is not None
        line = self._rfile.readline()
        if not line:
            raise AIClientError("server closed the connection")
        try:
            obj = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise AIClientError(f"invalid JSON from server: {exc}") from exc
        if not isinstance(obj, dict):
            raise AIClientError(f"expected JSON object, got {type(obj).__name__}")
        return obj