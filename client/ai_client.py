"""
Socket client for talking to the DocChat dummy AI server.

Encapsulates the newline-delimited JSON protocol so the TUI stays focused on
rendering. Reconnects automatically on the next call after a dropped socket.

Wire protocol (see services.dummy_server for the server side):

    Client -> Server:
        {"type": "ping"}
        {"type": "login",  "username": <str>, "password": <str>}
        {"type": "signup", "username": <str>, "password": <str>}
        {"type": "logout"}
        {"type": "query",  "text": <str>}            # session-authenticated
        {"type": "upload", "filename": <str>,        # session-authenticated

                           "bytes_b64": <base64 str>}

    Server -> Client:
        {"type": "pong"}
        {"type": "auth_ok",    "username": <str>}
        {"type": "logout_ok"}
        {"type": "answer",     "text": <str>}
        {"type": "upload_ack", "filename": <str>, "size": <int>}
        {"type": "error",      "message": <str>}

Session model:
    Auth is bound to the connection, not to individual messages. After
    login() or signup() succeeds the server remembers the user for the
    rest of the socket — ask() and upload() carry no username.

    Two things follow:
      1. If the underlying socket reconnects (e.g. after a transport
         error), the session on the *new* socket is empty. AIClient
         re-runs the last successful login automatically so callers
         don't need to think about it. See _ensure_session().
      2. Closing the client and opening a new one is a clean logout —
         no token to revoke server-side.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import threading
from typing import IO, Any

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555
DEFAULT_TIMEOUT = 10.0  # seconds for a single request/response round-trip

# Default upload timeout — larger files take longer to round-trip than queries.
UPLOAD_TIMEOUT = 60.0

# Per-upload size cap on the client side. Mirrors the server's cap
# Hard upload size cap (default: 5 MB per file).
# Catching this on the client saves a wasted base64-encode + transmit + reject.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


class AIClientError(RuntimeError):
    """Raised when the client cannot complete a request."""


class AIClient:
    """
    Thin synchronous client for the dummy AI server.

    Thread-safe at the request level (one request at a time per client),
    which is all the TUI needs since it sends a message and then waits.

    Attributes:
        username: The currently authenticated username, or None if no
                  login has happened on this client yet. Read-only from
                  outside; updated by login(), signup(), and logout().
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

        # Session state — see class docstring. _password is held in memory
        # only so we can transparently re-authenticate after a socket
        # reconnect; it's wiped by logout() and by close().
        self._username: str | None = None
        self._password: str | None = None

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
            # Wipe the cached password on full close so it doesn't outlive
            # the client. The username is harmless and stays for the UI.
            self._password = None

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

    # -- session ------------------------------------------------------------
    @property
    def username(self) -> str | None:
        """The currently authenticated username, or None."""
        return self._username

    @property
    def is_authenticated(self) -> bool:
        """True after a successful login()/signup() that has not been undone."""
        return self._username is not None

    # -- public API ----------------------------------------------------------
    def ping(self) -> bool:
        """Return True if the server responds with a pong."""
        try:
            reply = self._round_trip({"type": "ping"})
        except AIClientError:
            return False
        return reply.get("type") == "pong"

    def ask(self, text: str) -> str:
        """
        Send a query and return the AI's answer text.

        The connection must be authenticated — call login() or signup()
        first. The server stamps the answer to the session's user
        automatically; no username travels with the message.

        Raises:
            AIClientError: on transport failure, "not authenticated"
                           from the server, or any other error reply.
        """
        if not self.is_authenticated:
            raise AIClientError("not authenticated — call login() or signup() first")

        reply = self._authed_round_trip({"type": "query", "text": text})
        if reply.get("type") == "answer":
            answer = reply.get("text", "")
            if not isinstance(answer, str):
                raise AIClientError(f"non-string answer field: {answer!r}")
            return answer
        if reply.get("type") == "error":
            raise AIClientError(f"server error: {reply.get('message', 'unknown')}")
        raise AIClientError(f"unexpected reply type: {reply.get('type')!r}")

    def login(self, username: str, password: str) -> str:
        """
        Authenticate an existing account and bind the session.

        Returns:
            The server-confirmed username on success.

        Raises:
            AIClientError: on transport failure or auth failure. The
                           server returns "invalid credentials" for both
                           wrong-password and unknown-user cases, so the
                           client cannot distinguish them either.
        """
        return self._auth("login", username, password)

    def signup(self, username: str, password: str) -> str:
        """
        Create a brand-new account and bind the session.

        Returns:
            The server-confirmed username on success.

        Raises:
            AIClientError: on transport failure or if the username is
                           already taken. (Signup necessarily leaks
                           whether a username exists — see the auth
                           design note in services.dummy_server.)
        """
        return self._auth("signup", username, password)

    def logout(self) -> None:
        """
        End the current session.

        Always succeeds locally — the session is wiped even if the
        server-side logout message fails (e.g. broken socket).
        """
        had_session = self.is_authenticated
        self._username = None
        self._password = None
        if not had_session:
            return
        try:
            self._round_trip({"type": "logout"})
        except AIClientError:
            # The local state is already cleared; the server will GC
            # the binding when the connection drops anyway.
            pass

    def upload(self, filename: str, raw: bytes) -> dict[str, Any]:
        """
        Send a file upload and return the server's acknowledgement.

        The connection must be authenticated. The server stamps the
        resulting Document with the session's user automatically; no
        username travels with the message.

        Args:
            filename: Original filename to associate with the upload
                      (used by the server for parser dispatch + logging).
                      Only the basename is sent — the client's local
                      directory layout is irrelevant on the server.
            raw:      The raw file bytes.

        Returns:
            The decoded server reply, e.g. ``{"type": "upload_ack",
            "filename": "...", "size": 1234}``.

        Raises:
            AIClientError: on missing session, size-cap violation,
                           transport failure, or an error reply from
                           the server.
        """
        if not self.is_authenticated:
            raise AIClientError("not authenticated — call login() or signup() first")

        if not isinstance(raw, (bytes, bytearray)):
            raise AIClientError(f"upload expected bytes, got {type(raw).__name__}")

        if len(raw) > MAX_UPLOAD_BYTES:
            raise AIClientError(
                f"file '{filename}' is {len(raw)} bytes — exceeds "
                f"the {MAX_UPLOAD_BYTES}-byte upload cap"
            )

        # Strip any directory component on the way out — what we send is
        # metadata, not a path.
        base = os.path.basename(filename)
        if not base:
            raise AIClientError(f"invalid filename: {filename!r}")

        payload_b64 = base64.b64encode(bytes(raw)).decode("ascii")
        message = {
            "type": "upload",
            "filename": base,
            "bytes_b64": payload_b64,
        }

        reply = self._authed_round_trip(message, timeout=UPLOAD_TIMEOUT)
        kind = reply.get("type")
        if kind == "upload_ack":
            return reply
        if kind == "error":
            raise AIClientError(f"server error: {reply.get('message', 'unknown')}")
        raise AIClientError(f"unexpected reply type: {kind!r}")

    # -- session-aware transport ---------------------------------------------
    def _auth(self, mode: str, username: str, password: str) -> str:
        """
        Send a login or signup, cache credentials on success.

        Credentials are cached only after auth_ok so that a failed login
        does not overwrite a working session. Cached password is used
        for transparent re-auth after a reconnect.
        """
        reply = self._round_trip(
            {
                "type": mode,
                "username": username,
                "password": password,
            }
        )
        if reply.get("type") == "auth_ok":
            confirmed = reply.get("username")
            if not isinstance(confirmed, str):
                raise AIClientError(f"auth_ok missing username field: {reply!r}")
            self._username = confirmed
            self._password = password
            return confirmed
        if reply.get("type") == "error":
            raise AIClientError(str(reply.get("message", "auth failed")))
        raise AIClientError(f"unexpected reply type: {reply.get('type')!r}")

    def _authed_round_trip(
        self,
        message: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Like _round_trip, but transparent to "not authenticated" responses.

        If the server rejects a message with "not authenticated" — which
        only happens when the socket reconnected and we lost the session
        — re-run the cached login once and retry the original message.
        Without this, every transient network blip would force the TUI
        to ask the user for credentials again.
        """
        reply = self._round_trip(message, timeout=timeout)
        if (
            reply.get("type") == "error"
            and "not authenticated" in str(reply.get("message", ""))
            and self._username is not None
            and self._password is not None
        ):
            # Re-bind the session on the (likely new) socket, then retry.
            self._round_trip(
                {
                    "type": "login",
                    "username": self._username,
                    "password": self._password,
                }
            )
            reply = self._round_trip(message, timeout=timeout)
        return reply

    # -- low-level transport -------------------------------------------------
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
