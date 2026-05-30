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
    {"type": "ping"}
    {"type": "query",  "username": "<str>", "text": "<str>"}
    {"type": "upload", "username": "<str>", "filename": "<str>",
                       "bytes_b64": "<base64 str>"}

Server -> Client:
    {"type": "pong"}
    {"type": "answer",     "text": "<str>"}             # reply to "query"
    {"type": "upload_ack", "filename": "<str>", "size": <int>}
    {"type": "error",      "message": "<str>"}          # malformed / oversized

The upload handler is intentionally a *stub* — it validates the message
shape, base64-decodes the payload, checks the size cap and the allow-list
of extensions, and returns an ack. It does NOT yet feed the bytes into
IngestionService (parser → chunk → embed → store). Wiring that up is the
next step; the protocol is stable and the TUI can be developed against
this stub.

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
import json
import logging
import random
import socket
import socketserver
import threading
import time
from typing import Any

# Configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5555

# Simulated "thinking" delay so the UI feels like it's talking to a real model.
MIN_THINK_SECONDS = 0.2
MAX_THINK_SECONDS = 0.8

# Hard upload size cap
# ("Hard upload size cap (default: 5 MB per file)").
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# Extensions the server will accept. In the wired-up server this comes from
# ParserRegistry.supported_extensions; the stub hard-codes the same set so
# the TUI gets the same early-rejection behaviour.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {"txt", "md", "markdown", "mkd", "mkdn", "mdown"}
)

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


# Handler
class _DummyAIHandler(socketserver.StreamRequestHandler):
    """One instance per client connection. Reads NDJSON, writes NDJSON."""

    # Make sure a slow client doesn't hang the server forever.
    timeout = 300  # seconds

    # Buffered reader's default line length cap is generous, but with base64
    # payloads up to 5 MB our messages can hit ~7 MB. Bump rbufsize so
    # rfile.readline() doesn't truncate.
    rbufsize = 8 * 1024 * 1024  # 8 MB

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
        if kind == "query":
            text = str(msg.get("text", "")).strip()
            username = str(msg.get("username", "anonymous"))
            if not text:
                return {"type": "error", "message": "empty 'text' field"}
            return self._answer(username, text)
        if kind == "upload":
            return self._handle_upload(msg)
        return {"type": "error", "message": f"unknown type: {kind!r}"}

    def _answer(self, username: str, text: str) -> dict[str, Any]:
        # Pretend to think.
        time.sleep(random.uniform(MIN_THINK_SECONDS, MAX_THINK_SECONDS))
        body = random.choice(DUMMY_RESPONSES)
        # Echo a hint of the question so it feels less robotic.
        preview = text if len(text) <= 60 else text[:57] + "..."
        answer = f'[dummy] {body}  (re: "{preview}")'
        logger.info("query from %s: %r -> reply len=%d", username, preview, len(answer))
        return {"type": "answer", "text": answer}

    def _handle_upload(self, msg: dict[str, Any]) -> dict[str, Any]:
        """
        Validate an upload message and return an ack.

        Steps:
          1. Shape validation (required string fields present, non-empty).
          2. Extension allow-list check against SUPPORTED_EXTENSIONS.
          3. Base64 decode (rejects malformed payload).
          4. Size cap check against MAX_UPLOAD_BYTES.

        This is a stub — once IngestionService is wired in, the decoded
        bytes will be passed to it for the parse → chunk → embed → store
        pipeline. For now we just log + acknowledge.
        """
        username = str(msg.get("username", "anonymous"))

        filename = msg.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return {"type": "error", "message": "missing or empty 'filename'"}

        bytes_b64 = msg.get("bytes_b64")
        if not isinstance(bytes_b64, str):
            return {"type": "error", "message": "missing 'bytes_b64' field"}

        # Strip any directory component the client may have included.
        import os

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

        # TODO: hand `raw` and `filename` to IngestionService.ingest(...) here.
        logger.info(
            "upload from %s: '%s' accepted (%d bytes, ext=.%s) [stub: not ingested]",
            username,
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
    serve_forever(args.host, args.port)


if __name__ == "__main__":
    main()
