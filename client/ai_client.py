"""
Socket client for talking to the DocChat dummy AI server.

Encapsulates the newline-delimited JSON protocol so the TUI stays focused on
rendering. Reconnects automatically on the next call after a dropped socket.

Wire protocol (see services.dummy_server for the server side):

    Client -> Server:
        {"type": "ping"}
        {"type": "query",  "username": <str>, "text": <str>}
        {"type": "upload", "username": <str>, "filename": <str>,
                           "bytes_b64": <base64 str>}

    Server -> Client:
        {"type": "pong"}
        {"type": "answer",     "text": <str>}
        {"type": "upload_ack", "filename": <str>, "size": <int>}
        {"type": "error",      "message": <str>}
"""

from __future__ import annotations

import base64
import json
import socket
import threading
from typing import IO, Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT = 10.0  # seconds for a single request/response round-trip

# Default upload timeout — larger files take longer to round-trip than queries.
UPLOAD_TIMEOUT = 60.0

# Per-upload size cap on the client side. Mirrors the server's cap
# "Hard upload size cap (default: 5 MB per file)").
# Catching this on the client saves a wasted base64-encode + transmit + reject.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


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
        self._rfile: IO[bytes] | None = None  # buffered reader for line-based reads
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def connect(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
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

    def upload(
        self,
        filename: str,
        raw: bytes,
        username: str = "anonymous",
    ) -> dict[str, Any]:
        """
        Send a file upload and return the server's acknowledgement.

        The TUI reads the file from local disk into *raw* and calls this
        method. Bytes are base64-encoded before being wrapped in JSON so
        the existing newline-delimited transport stays text-safe.

        Args:
            filename: Original filename to associate with the upload
                      (used by the server for parser dispatch + logging).
                      Only the basename is sent — the client's local
                      directory layout is irrelevant on the server.
            raw:      The raw file bytes.
            username: Authenticated user (currently stub-tracked).

        Returns:
            The decoded server reply, e.g. ``{"type": "upload_ack",
            "filename": "...", "size": 1234}``.

        Raises:
            AIClientError: on size-cap violation, transport failure, or
                           an error reply from the server.
        """
        if not isinstance(raw, (bytes, bytearray)):
            raise AIClientError(f"upload expected bytes, got {type(raw).__name__}")

        if len(raw) > MAX_UPLOAD_BYTES:
            raise AIClientError(
                f"file '{filename}' is {len(raw)} bytes — exceeds "
                f"the {MAX_UPLOAD_BYTES}-byte upload cap"
            )

        # Strip any directory component on the way out — what we send is
        # metadata, not a path.
        import os

        base = os.path.basename(filename)
        if not base:
            raise AIClientError(f"invalid filename: {filename!r}")

        payload_b64 = base64.b64encode(bytes(raw)).decode("ascii")
        message = {
            "type": "upload",
            "username": username,
            "filename": base,
            "bytes_b64": payload_b64,
        }

        reply = self._round_trip(message, timeout=UPLOAD_TIMEOUT)
        kind = reply.get("type")
        if kind == "upload_ack":
            return reply
        if kind == "error":
            raise AIClientError(f"server error: {reply.get('message', 'unknown')}")
        raise AIClientError(f"unexpected reply type: {kind!r}")

    # -- internals -----------------------------------------------------------
    def _round_trip(
        self,
        message: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Send *message*, read one reply line, return the decoded JSON.

        If *timeout* is given, it's applied to this one round-trip only;
        the socket's prior timeout is restored before we return.
        """
        with self._lock:
            # First attempt; reconnect once if the socket is stale.
            for attempt in (1, 2):
                try:
                    if self._sock is None:
                        # connect() takes the lock too -- inline to avoid recursion.
                        self._connect_locked()

                    prior_timeout: float | None = None
                    if timeout is not None and self._sock is not None:
                        prior_timeout = self._sock.gettimeout()
                        self._sock.settimeout(timeout)
                    try:
                        self._send_locked(message)
                        return self._recv_locked()
                    finally:
                        if (
                            timeout is not None
                            and self._sock is not None
                            and prior_timeout is not None
                        ):
                            self._sock.settimeout(prior_timeout)
                except (OSError, AIClientError) as exc:
                    self._close_locked()
                    if attempt == 2:
                        raise AIClientError(str(exc)) from exc
            # Unreachable, but mypy can't tell.
            raise AIClientError("round-trip failed")

    def _connect_locked(self) -> None:
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.timeout
            )
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
