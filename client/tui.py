#!/usr/bin/env python3
"""
DocChat TUI - AI-powered document chat interface.

Login and signup are real: credentials are sent to the dummy server over
a newline-delimited JSON socket, where they are hashed and stored. The
TUI falls back to local-only auth when running --offline so it stays
demoable without a backend.

Uploads read the chosen file from local disk into bytes and send them to
the server (the server stub validates the upload and returns an ack — no
real ingestion yet). AI replies use the same socket; if the server is
unreachable the chat screen falls back to a local stub.

Run the server in one terminal:
    python -m services.dummy_server

Then launch the TUI in another:
    python -m client.tui
    python -m client.tui --host 127.0.0.1 --port 5555    # custom server
    python -m client.tui --offline                       # force local stub
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import textwrap
from datetime import datetime
from typing import Any

try:
    import curses
except ModuleNotFoundError:
    curses = None  # type: ignore[assignment]

# Make ``python client/tui.py`` work as well as ``python -m client.tui``.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.ai_client import (
    AIClient,
    AIClientError,
    AnswerResponse,
    AnswerSource,
    MAX_UPLOAD_BYTES,
)

# In-memory state
state: dict[str, Any] = {
    "logged_in": False,
    "username": "",
    "documents": [],  # {"name", "size", "uploaded_at"}
    "messages": [],  # {"role": "user"|"ai", "text", "time", "sources"?}
    "current_screen": "login",
    "ai_client": None,  # AIClient or None when running --offline
    "ai_status": "offline",  # "online" | "offline" | "error: ..."
}

STUB_AI_RESPONSES = [
    "[offline] Based on your documents, here is a placeholder answer.",
    "[offline] The documents mention several relevant points. (Server not connected.)",
    "[offline] This will be answered by the real AI once the server is reachable.",
    "[offline] Interesting question! Start the dummy server to see real replies.",
]
_stub_index = 0


def stub_ai_response() -> str:
    """Local fallback used when the dummy server is unreachable."""
    global _stub_index
    r = STUB_AI_RESPONSES[_stub_index % len(STUB_AI_RESPONSES)]
    _stub_index += 1
    return r


def fetch_ai_response(text: str) -> AnswerResponse:
    """
    Ask the dummy server for a reply, with graceful fallback.

    Updates ``state["ai_status"]`` so the chat screen can show the
    current connection state. Always returns an AnswerResponse -- never raises.
    """
    client = state.get("ai_client")
    if client is None:
        state["ai_status"] = "offline"
        return AnswerResponse(answer=stub_ai_response())
    try:
        answer = client.ask_with_sources(text)
        state["ai_status"] = "online"
        return answer
    except AIClientError as exc:
        state["ai_status"] = f"error: {exc}"
        return AnswerResponse(answer=f"[server error] {exc}")


def format_source(source: AnswerSource, index: int) -> str:
    """Build a compact, readable source label for the chat transcript."""
    metadata = source.metadata
    name = (
        metadata.get("filename")
        or metadata.get("source_file")
        or metadata.get("document")
        or source.doc_id
    )
    parts = [f"[{index}] {name}", f"chunk {source.position}"]

    page = metadata.get("page")
    if page is not None:
        parts.append(f"page {page}")

    if source.similarity is not None:
        parts.append(f"score {source.similarity:.3f}")

    return " | ".join(str(part) for part in parts)


def refresh_documents_from_server() -> tuple[bool, str]:
    """Reload the authenticated user's documents from the server, if available."""
    client = state.get("ai_client")
    if client is None:
        return False, "offline"

    try:
        documents = client.list_documents()
    except AIClientError as exc:
        state["ai_status"] = f"error: {exc}"
        return False, str(exc)

    state["documents"] = [_document_summary_for_tui(doc) for doc in documents]
    state["ai_status"] = f"online ({client.host}:{client.port})"
    return True, ""


def _document_summary_for_tui(doc: dict[str, Any]) -> dict[str, Any]:
    uploaded_at = str(doc.get("uploaded_at") or "")
    if "T" in uploaded_at:
        uploaded_at = uploaded_at.replace("T", " ")[:16]
    elif not uploaded_at:
        uploaded_at = "unknown"

    size = doc.get("size", 0)
    if not isinstance(size, (int, float)):
        size = 0

    return {
        "id": str(doc.get("document_id") or doc.get("id") or ""),
        "name": str(doc.get("filename") or doc.get("name") or ""),
        "size": int(size),
        "uploaded_at": uploaded_at,
        "status": str(doc.get("status") or ""),
    }


def normalize_input_path(raw: str | None) -> str:
    """
    Clean up a path the way a user is likely to have typed or pasted it.

    Handles the common Windows + cross-platform footguns:
      - Surrounding double or single quotes (from File Explorer's
        "Copy as path" or PowerShell tab completion).
      - Leading / trailing whitespace.
      - A ``~`` for the home directory.
      - Mixed slashes (normpath collapses them).

    Returns the cleaned path. Does NOT check existence — that's
    upload_file_to_server's job.
    """
    if raw is None:
        return ""

    path = raw.strip()

    # Strip a single matching pair of surrounding quotes (single or double).
    if len(path) >= 2 and path[0] == path[-1] and path[0] in ('"', "'"):
        path = path[1:-1].strip()

    # ~/notes.txt -> $HOME/notes.txt (or %USERPROFILE% on Windows).
    if path.startswith("~"):
        path = os.path.expanduser(path)

    # Collapse "//" and ".." segments, normalise slash direction.
    if path:
        path = os.path.normpath(path)

    return path


def upload_file_to_server(path: str) -> tuple[bool, str]:
    """
    Read *path* from local disk, send the bytes to the server, and return
    (ok, message).

    Pre-flight checks happen before the file is even read so we don't
    waste a slurp on a file that's obviously too large. All failures are
    returned as ``(False, "...")``; the caller decides whether to surface
    them as status messages or treat them differently.

    When running --offline, the file is still read (to learn its size
    for the documents list) but nothing is sent.
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return False, f"Cannot read '{path}': {exc}"

    if size > MAX_UPLOAD_BYTES:
        return False, (
            f"'{os.path.basename(path)}' is {size} bytes — exceeds the "
            f"{MAX_UPLOAD_BYTES}-byte cap"
        )

    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        return False, f"Cannot read '{path}': {exc}"

    name = os.path.basename(path)
    client = state.get("ai_client")

    if client is None:
        # Offline: simulate a successful upload so the UI flow still works.
        state["documents"].append(
            {
                "name": name,
                "size": size,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        )
        return True, f"'{name}' added locally (offline -- not sent to server)"

    try:
        ack = client.upload(name, raw)
    except AIClientError as exc:
        state["ai_status"] = f"error: {exc}"
        return False, f"Upload failed: {exc}"

    state["ai_status"] = f"online ({client.host}:{client.port})"
    chunks = ack.get("chunks")  # absent when the server runs the validate-only stub
    refreshed, _ = refresh_documents_from_server()
    if not refreshed:
        state["documents"].append(
            {
                "name": ack.get("filename", name),
                "size": ack.get("size", size),
                "chunks": chunks,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        )
    detail = f"{size} bytes" if chunks is None else f"{size} bytes, {chunks} chunks"
    return True, f"'{name}' uploaded ({detail})"


def _remove_local_document(doc: dict[str, Any]) -> None:
    """Drop *doc* from the in-memory documents list (by id, else by identity)."""
    docs = state["documents"]
    doc_id = doc.get("id")
    if doc_id:
        state["documents"] = [d for d in docs if d.get("id") != doc_id]
    else:
        state["documents"] = [d for d in docs if d is not doc]


def delete_document_on_server(doc: dict[str, Any]) -> tuple[bool, str]:
    """
    Delete *doc* from the server (metadata + chunks), then refresh the list.

    Returns (ok, message). Offline, or for a document with no server id
    (e.g. one added during an --offline session), it just drops the entry
    locally so the UI stays consistent.
    """
    name = doc.get("name") or "document"
    client = state.get("ai_client")
    doc_id = doc.get("id")

    if client is None:
        _remove_local_document(doc)
        return True, f"'{name}' removed (offline -- not sent to server)"

    if not doc_id:
        _remove_local_document(doc)
        return True, f"'{name}' removed locally (no server id)"

    try:
        reply = client.delete_document(doc_id)
    except AIClientError as exc:
        state["ai_status"] = f"error: {exc}"
        return False, f"Delete failed: {exc}"

    state["ai_status"] = f"online ({client.host}:{client.port})"
    deleted_chunks = reply.get("deleted_chunks")

    # Reflect the server's new truth; fall back to a local drop if the
    # refresh round-trip fails for any reason.
    refreshed, _ = refresh_documents_from_server()
    if not refreshed:
        _remove_local_document(doc)

    if deleted_chunks:
        return True, f"'{name}' deleted from server ({deleted_chunks} chunks removed)"
    return True, f"'{name}' deleted from server"


# ── Color pair IDs ───────────────────────────────────────────────────────────
C_NORMAL = 1
C_HEADER = 2
C_ACCENT = 3
C_DIM = 4
C_ERROR = 5
C_SUCCESS = 6
C_USER_MSG = 7
C_AI_MSG = 8


# Safe wrappers
def safe_curs_set(v: int) -> None:
    try:
        curses.curs_set(v)
    except curses.error:
        pass


def safe_addstr(win: curses.window, y: int, x: int, s: str, attr: int = 0) -> None:
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        max_len = w - x - 1
        if max_len <= 0:
            return
        s = s[:max_len]
        if attr:
            win.addstr(y, x, s, attr)
        else:
            win.addstr(y, x, s)
    except curses.error:
        pass


def init_colors() -> None:
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK

    curses.init_pair(C_NORMAL, curses.COLOR_WHITE, bg)
    curses.init_pair(C_HEADER, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(C_ACCENT, curses.COLOR_CYAN, bg)
    curses.init_pair(C_DIM, curses.COLOR_WHITE, bg)
    curses.init_pair(C_ERROR, curses.COLOR_RED, bg)
    curses.init_pair(C_SUCCESS, curses.COLOR_GREEN, bg)
    curses.init_pair(C_USER_MSG, curses.COLOR_CYAN, bg)
    curses.init_pair(C_AI_MSG, curses.COLOR_YELLOW, bg)


# Drawing helpers
def draw_header(win: curses.window, title: str) -> None:
    h, w = win.getmaxyx()
    text = f"  DocChat  --  {title}  "
    safe_addstr(
        win, 0, 0, text.ljust(w - 1), curses.color_pair(C_HEADER) | curses.A_BOLD
    )


def draw_footer(win: curses.window, hints: list[str]) -> None:
    h, w = win.getmaxyx()
    text = "   ".join(hints)
    safe_addstr(win, h - 1, 0, text.ljust(w - 1), curses.color_pair(C_HEADER))


def center_text(win: curses.window, row: int, text: str, attr: int = 0) -> None:
    _, w = win.getmaxyx()
    col = max(0, (w - len(text)) // 2)
    safe_addstr(win, row, col, text, attr)


def draw_box(
    win: curses.window, y: int, x: int, h: int, w: int, title: str = ""
) -> None:
    """Plain ASCII box - works on every Windows terminal."""
    safe_addstr(win, y, x, "+")
    safe_addstr(win, y, x + w - 1, "+")
    safe_addstr(win, y + h - 1, x, "+")
    safe_addstr(win, y + h - 1, x + w - 1, "+")
    for i in range(1, w - 1):
        safe_addstr(win, y, x + i, "-")
        safe_addstr(win, y + h - 1, x + i, "-")
    for i in range(1, h - 1):
        safe_addstr(win, y + i, x, "|")
        safe_addstr(win, y + i, x + w - 1, "|")
    if title:
        safe_addstr(
            win, y, x + 2, f" {title} ", curses.color_pair(C_ACCENT) | curses.A_BOLD
        )


def read_line(
    win: curses.window, y: int, x: int, max_len: int, mask: bool = False
) -> str:
    curses.echo()
    safe_curs_set(1)
    try:
        win.move(y, x)
    except curses.error:
        pass
    win.refresh()
    buf: list[str] = []
    while True:
        ch = win.getch()
        if ch in (10, 13):
            break
        elif ch in (127, curses.KEY_BACKSPACE, 8):
            if buf:
                buf.pop()
                safe_addstr(win, y, x + len(buf), " ")
                try:
                    win.move(y, x + len(buf))
                except curses.error:
                    pass
        elif 32 <= ch < 127 and len(buf) < max_len:
            buf.append(chr(ch))
            safe_addstr(win, y, x + len(buf) - 1, "*" if mask else chr(ch))
        win.refresh()
    curses.noecho()
    safe_curs_set(0)
    return "".join(buf)


# ── Screens ──────────────────────────────────────────────────────────────────


def screen_auth_choice(stdscr: curses.window) -> str:
    """
    Entry screen: ask the user whether they want to log in or sign up.

    Returns "login" or "signup".
    """
    selected = 0
    options = ["Login", "Sign Up"]

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, "Welcome")
        draw_footer(
            stdscr, ["Up/Down to navigate", "Enter to select", "Ctrl+C to quit"]
        )

        center_text(
            stdscr, 2, "Welcome to DocChat", curses.color_pair(C_ACCENT) | curses.A_BOLD
        )
        center_text(
            stdscr,
            3,
            "AI-powered document Q&A",
            curses.color_pair(C_DIM) | curses.A_DIM,
        )

        bw, bh = 40, 10
        bx = max(0, (w - bw) // 2)
        by = max(1, (h - bh) // 2)
        draw_box(stdscr, by, bx, bh, bw, "Get Started")

        for i, label in enumerate(options):
            row = by + 2 + i * 3
            is_sel = i == selected
            arrow = ">>  " if is_sel else "    "
            attr = (
                (curses.color_pair(C_ACCENT) | curses.A_BOLD)
                if is_sel
                else curses.color_pair(C_NORMAL)
            )
            safe_addstr(stdscr, row, bx + 4, arrow + label, attr)

        stdscr.refresh()

        ch = stdscr.getch()
        if ch == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        elif ch == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        elif ch in (10, 13):
            return "login" if selected == 0 else "signup"
        elif ch in (ord("1"),):
            return "login"
        elif ch in (ord("2"),):
            return "signup"


def screen_login(stdscr: curses.window) -> None:
    """
    Login screen — verifies credentials against the DB.

    On wrong password the error is shown and the user is asked to try
    again. If the server is unreachable, credentials are accepted locally
    (offline stub) so the TUI is still demoable without the backend.
    """
    error_msg = ""

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, "Login")
        draw_footer(stdscr, ["Enter to confirm", "Ctrl+C to quit"])

        center_text(
            stdscr, 2, "Welcome back!", curses.color_pair(C_ACCENT) | curses.A_BOLD
        )
        center_text(
            stdscr,
            3,
            "Sign in to your existing account",
            curses.color_pair(C_DIM) | curses.A_DIM,
        )

        bw, bh = 52, 12
        bx = max(0, (w - bw) // 2)
        by = max(1, (h - bh) // 2)
        draw_box(stdscr, by, bx, bh, bw, "Login")

        safe_addstr(stdscr, by + 2, bx + 3, "Username:", curses.color_pair(C_ACCENT))
        safe_addstr(stdscr, by + 5, bx + 3, "Password:", curses.color_pair(C_ACCENT))

        client = state.get("ai_client")
        if client is None:
            safe_addstr(
                stdscr,
                by + 9,
                bx + 3,
                "(offline -- credentials accepted locally)",
                curses.color_pair(C_DIM) | curses.A_DIM,
            )

        if error_msg:
            safe_addstr(
                stdscr,
                by + 10,
                bx + 3,
                error_msg[: bw - 6],
                curses.color_pair(C_ERROR) | curses.A_BOLD,
            )

        stdscr.refresh()

        username = read_line(stdscr, by + 3, bx + 3, bw - 6).strip()
        password = read_line(stdscr, by + 6, bx + 3, bw - 6, mask=True)

        if not username or not password:
            error_msg = "Username and password are required."
            continue

        # Offline mode: accept locally so the TUI is demoable without a backend.
        if client is None:
            state["username"] = username
            state["logged_in"] = True
            safe_addstr(
                stdscr,
                by + 7,
                bx + 3,
                "Logged in (offline)!",
                curses.color_pair(C_SUCCESS) | curses.A_BOLD,
            )
            stdscr.refresh()
            time.sleep(0.7)
            state["current_screen"] = "main"
            return

        # Online: send a login request. The server looks the username up,
        # verifies the password hash, and returns auth_ok or an error.
        try:
            confirmed = client.login(username, password)
        except AIClientError as exc:
            msg = str(exc)
            if "incorrect password" in msg:
                error_msg = "Wrong password, try again."
            elif "user not found" in msg:
                error_msg = "Username not found. Did you mean to sign up?"
            else:
                error_msg = f"Server error: {msg}"
            continue

        state["username"] = confirmed
        state["logged_in"] = True
        refresh_documents_from_server()
        safe_addstr(
            stdscr,
            by + 7,
            bx + 3,
            "Signed in!",
            curses.color_pair(C_SUCCESS) | curses.A_BOLD,
        )
        stdscr.refresh()
        time.sleep(0.7)
        state["current_screen"] = "main"
        return


def screen_signup(stdscr: curses.window) -> None:
    """
    Sign-up screen — creates a new account in the DB.

    If the username already exists the error is shown and the user is
    asked to choose a different name. If the server is unreachable,
    the account is accepted locally so the TUI remains demoable.
    """
    error_msg = ""

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, "Sign Up")
        draw_footer(stdscr, ["Enter to confirm", "Ctrl+C to quit"])

        center_text(
            stdscr,
            2,
            "Create your account",
            curses.color_pair(C_ACCENT) | curses.A_BOLD,
        )
        center_text(
            stdscr,
            3,
            "Choose a username and password",
            curses.color_pair(C_DIM) | curses.A_DIM,
        )

        bw, bh = 52, 14
        bx = max(0, (w - bw) // 2)
        by = max(1, (h - bh) // 2)
        draw_box(stdscr, by, bx, bh, bw, "Sign Up")

        safe_addstr(stdscr, by + 2, bx + 3, "Username:", curses.color_pair(C_ACCENT))
        safe_addstr(stdscr, by + 5, bx + 3, "Password:", curses.color_pair(C_ACCENT))
        safe_addstr(
            stdscr, by + 8, bx + 3, "Confirm password:", curses.color_pair(C_ACCENT)
        )

        client = state.get("ai_client")
        if client is None:
            safe_addstr(
                stdscr,
                by + 11,
                bx + 3,
                "(offline -- account saved locally)",
                curses.color_pair(C_DIM) | curses.A_DIM,
            )

        if error_msg:
            safe_addstr(
                stdscr,
                by + 12,
                bx + 3,
                error_msg[: bw - 6],
                curses.color_pair(C_ERROR) | curses.A_BOLD,
            )

        stdscr.refresh()

        username = read_line(stdscr, by + 3, bx + 3, bw - 6).strip()
        password = read_line(stdscr, by + 6, bx + 3, bw - 6, mask=True)
        confirm = read_line(stdscr, by + 9, bx + 3, bw - 6, mask=True)

        if not username or not password:
            error_msg = "Username and password are required."
            continue

        if password != confirm:
            error_msg = "Passwords do not match. Try again."
            continue

        # Offline mode: accept locally.
        if client is None:
            state["username"] = username
            state["logged_in"] = True
            safe_addstr(
                stdscr,
                by + 10,
                bx + 3,
                "Account created (offline)!",
                curses.color_pair(C_SUCCESS) | curses.A_BOLD,
            )
            stdscr.refresh()
            time.sleep(0.7)
            state["current_screen"] = "main"
            return

        # Online: send a signup request. The server creates the account
        # if the username is free, or returns an error if it's taken.
        try:
            confirmed = client.signup(username, password)
        except AIClientError as exc:
            msg = str(exc)
            if "already taken" in msg or "already exists" in msg:
                error_msg = "Username already taken. Choose another."
            else:
                error_msg = f"Server error: {msg}"
            continue

        state["username"] = confirmed
        state["logged_in"] = True
        refresh_documents_from_server()
        safe_addstr(
            stdscr,
            by + 10,
            bx + 3,
            "Account created!",
            curses.color_pair(C_SUCCESS) | curses.A_BOLD,
        )
        stdscr.refresh()
        time.sleep(0.7)
        state["current_screen"] = "main"
        return


def screen_main(stdscr: curses.window) -> None:
    """Main menu with >> arrow on selected item."""
    menu_items = [
        ("1", "Chat with AI", "chat", "Ask questions about your documents"),
        ("2", "Manage Documents", "upload", "Upload and view your files"),
        ("Q", "Quit", "quit", ""),
    ]
    selected = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, f"Main Menu  [{state['username']}]")
        draw_footer(stdscr, ["Up/Down to navigate", "Enter to select", "Q to quit"])

        center_text(
            stdscr,
            2,
            "What would you like to do?",
            curses.color_pair(C_ACCENT) | curses.A_BOLD,
        )

        col = max(4, w // 2 - 18)
        for i, (key, label, _, desc) in enumerate(menu_items):
            row = 5 + i * 3
            is_sel = i == selected

            arrow = ">>  " if is_sel else "    "
            attr_arrow = curses.color_pair(C_ACCENT) | curses.A_BOLD
            attr_key = (
                (curses.color_pair(C_ACCENT) | curses.A_BOLD)
                if is_sel
                else curses.color_pair(C_DIM) | curses.A_DIM
            )
            attr_label = (
                (curses.color_pair(C_NORMAL) | curses.A_BOLD)
                if is_sel
                else curses.color_pair(C_NORMAL)
            )
            attr_desc = curses.color_pair(C_DIM) | curses.A_DIM

            safe_addstr(stdscr, row, col, arrow, attr_arrow if is_sel else attr_desc)
            safe_addstr(stdscr, row, col + 4, f"[{key}]  ", attr_key)
            safe_addstr(stdscr, row, col + 9, label, attr_label)
            if desc:
                safe_addstr(stdscr, row + 1, col + 9, desc, attr_desc)

        doc_count = len(state["documents"])
        safe_addstr(
            stdscr,
            h - 3,
            2,
            f"Documents in store: {doc_count}",
            curses.color_pair(C_DIM) | curses.A_DIM,
        )
        stdscr.refresh()

        dest = None
        ch = stdscr.getch()

        if ch == curses.KEY_UP:
            selected = (selected - 1) % len(menu_items)
        elif ch == curses.KEY_DOWN:
            selected = (selected + 1) % len(menu_items)
        elif ch in (10, 13):
            dest = menu_items[selected][2]
        elif 0 < ch < 256 and chr(ch).lower() in ("1", "2", "q"):
            mapping = {"1": "chat", "2": "upload", "q": "quit"}
            dest = mapping[chr(ch).lower()]

        if dest == "quit":
            return
        elif dest == "chat":
            screen_chat(stdscr)
        elif dest == "upload":
            screen_documents(stdscr)


def screen_documents(stdscr: curses.window) -> None:
    msg = ""
    msg_attr = C_SUCCESS
    selected = 0
    pending_delete_key: str | None = None

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, "Document Manager")
        draw_footer(
            stdscr,
            ["U upload", "D delete (press twice)", "B back", "Up/Down navigate"],
        )

        safe_addstr(
            stdscr,
            2,
            2,
            "Your uploaded documents:",
            curses.color_pair(C_ACCENT) | curses.A_BOLD,
        )

        docs = state["documents"]
        list_top = 4
        list_h = h - 8

        for i, doc in enumerate(docs[:list_h]):
            row = list_top + i
            size_kb = doc["size"] / 1024
            is_sel = i == selected
            arrow = ">> " if is_sel else "   "
            chunks = doc.get("chunks")
            chunk_str = f"{chunks:>4} chunks" if chunks is not None else "  -- chunks"
            line = (
                f"{doc['name']:<28}  {size_kb:>6.1f} KB  {chunk_str}   "
                f"{doc['uploaded_at']}"
            )
            attr = (
                (curses.color_pair(C_ACCENT) | curses.A_BOLD)
                if is_sel
                else curses.color_pair(C_NORMAL)
            )
            safe_addstr(stdscr, row, 2, arrow + line, attr)

        if not docs:
            safe_addstr(
                stdscr,
                list_top,
                2,
                "  (no documents yet)",
                curses.color_pair(C_DIM) | curses.A_DIM,
            )
        if msg:
            safe_addstr(
                stdscr, h - 4, 2, msg, curses.color_pair(msg_attr) | curses.A_BOLD
            )
        stdscr.refresh()

        ch = stdscr.getch()
        c = chr(ch).lower() if 0 < ch < 256 else ""

        if ch == curses.KEY_UP and docs:
            selected = max(0, selected - 1)
            pending_delete_key = None
        elif ch == curses.KEY_DOWN and docs:
            selected = min(len(docs) - 1, selected + 1)
            pending_delete_key = None
        elif c == "u":
            pending_delete_key = None
            safe_addstr(stdscr, h - 6, 2, " " * (w - 4))
            safe_addstr(
                stdscr, h - 6, 2, "File path to upload:", curses.color_pair(C_ACCENT)
            )
            stdscr.refresh()
            raw_input = read_line(stdscr, h - 5, 2, w - 4)
            path = normalize_input_path(raw_input)
            if path and os.path.isfile(path):
                # Show a transient status while we read + ship the bytes.
                safe_addstr(stdscr, h - 4, 2, " " * (w - 4))
                safe_addstr(
                    stdscr,
                    h - 4,
                    2,
                    "Uploading...",
                    curses.color_pair(C_DIM) | curses.A_DIM,
                )
                stdscr.refresh()

                ok, msg = upload_file_to_server(path)
                msg_attr = C_SUCCESS if ok else C_ERROR
            elif path:
                # Show the cleaned-up path so the user can see why we
                # disagree with them (e.g. quotes were stripped).
                msg = f"File not found: {path}"
                msg_attr = C_ERROR
            else:
                msg = ""
                msg_attr = C_SUCCESS
        elif c == "d" and docs:
            doc_to_delete = docs[selected]
            doc_key = str(doc_to_delete.get("id") or f"local:{selected}")
            if pending_delete_key != doc_key:
                pending_delete_key = doc_key
                msg = f"Press D again to permanently delete '{doc_to_delete['name']}'"
                msg_attr = C_ERROR
            else:
                ok, msg = delete_document_on_server(doc_to_delete)
                msg_attr = C_SUCCESS if ok else C_ERROR
                pending_delete_key = None
                if ok:
                    selected = max(0, min(selected, len(state["documents"]) - 1))
        elif c == "b":
            break


def screen_chat(stdscr: curses.window) -> None:
    input_buf = ""
    scroll_offset = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        status = state.get("ai_status", "offline")
        # Keep the header short so it fits narrow terminals.
        status_label = status if len(status) <= 30 else status[:27] + "..."
        draw_header(stdscr, f"Chat  --  AI Document Q&A  [{status_label}]")
        draw_footer(stdscr, ["Enter to send", "PgUp/PgDn scroll", "B to go back"])

        msg_area_h = h - 5

        # Build wrapped display lines
        all_lines = []
        for m in state["messages"]:
            ts = m["time"]
            role = m["role"]
            if role == "user":
                prefix = f"  You [{ts}]: "
                attr = curses.color_pair(C_USER_MSG) | curses.A_BOLD
            else:
                prefix = f"  AI  [{ts}]: "
                attr = curses.color_pair(C_AI_MSG)
            wrap_w = max(1, w - 4 - len(prefix))
            wrapped = textwrap.wrap(m["text"], wrap_w) or [""]
            for j, line in enumerate(wrapped):
                pfx = prefix if j == 0 else " " * len(prefix)
                all_lines.append((attr, pfx + line))
            if role == "ai" and m.get("sources"):
                source_prefix = " " * len(prefix)
                all_lines.append(
                    (
                        curses.color_pair(C_DIM) | curses.A_DIM,
                        source_prefix + "Sources:",
                    )
                )
                for source_index, source in enumerate(m["sources"], start=1):
                    source_text = format_source(source, source_index)
                    for source_line in textwrap.wrap(source_text, wrap_w) or [""]:
                        all_lines.append(
                            (
                                curses.color_pair(C_DIM) | curses.A_DIM,
                                source_prefix + source_line,
                            )
                        )
            all_lines.append((curses.color_pair(C_NORMAL), ""))

        total_lines = len(all_lines)
        max_scroll = max(0, total_lines - msg_area_h)
        scroll_offset = min(scroll_offset, max_scroll)
        view_start = max(0, total_lines - msg_area_h - scroll_offset)

        for i, (attr, text) in enumerate(
            all_lines[view_start : view_start + msg_area_h]
        ):
            safe_addstr(stdscr, 1 + i, 2, text, attr)

        if not state["messages"]:
            safe_addstr(
                stdscr,
                3,
                4,
                "No messages yet. Ask something about your documents!",
                curses.color_pair(C_DIM) | curses.A_DIM,
            )

        safe_addstr(stdscr, h - 4, 0, "-" * (w - 1), curses.color_pair(C_ACCENT))
        safe_addstr(stdscr, h - 3, 2, "> ", curses.color_pair(C_ACCENT) | curses.A_BOLD)
        input_display = input_buf[-(w - 7) :]
        safe_addstr(stdscr, h - 3, 4, input_display, curses.color_pair(C_NORMAL))

        if scroll_offset > 0:
            safe_addstr(
                stdscr,
                h - 2,
                2,
                f"^ scrolled up {scroll_offset} line(s) -- PgDn to go down",
                curses.color_pair(C_DIM) | curses.A_DIM,
            )

        stdscr.refresh()
        safe_curs_set(1)
        try:
            stdscr.move(h - 3, min(4 + len(input_display), w - 2))
        except curses.error:
            pass

        ch = stdscr.getch()
        safe_curs_set(0)

        if ch == curses.KEY_PPAGE:
            scroll_offset = min(scroll_offset + max(1, msg_area_h // 2), max_scroll)
        elif ch == curses.KEY_NPAGE:
            scroll_offset = max(0, scroll_offset - max(1, msg_area_h // 2))
        elif ch in (127, curses.KEY_BACKSPACE, 8):
            input_buf = input_buf[:-1]
        elif ch in (10, 13):
            text = input_buf.strip()
            if text:
                now = datetime.now().strftime("%H:%M")
                state["messages"].append({"role": "user", "text": text, "time": now})
                input_buf = ""
                scroll_offset = 0

                # Show a "thinking" hint while we wait on the server.
                safe_addstr(
                    stdscr,
                    h - 2,
                    2,
                    "AI is thinking...",
                    curses.color_pair(C_DIM) | curses.A_DIM,
                )
                stdscr.refresh()

                response = fetch_ai_response(text)
                reply_time = datetime.now().strftime("%H:%M")
                state["messages"].append(
                    {
                        "role": "ai",
                        "text": response.answer,
                        "time": reply_time,
                        "sources": response.sources,
                    }
                )
        elif ch in (ord("b"), ord("B")) and not input_buf:
            break
        elif 32 <= ch < 127:
            input_buf += chr(ch)


# Entry point


def main(stdscr: curses.window) -> None:
    safe_curs_set(0)
    if hasattr(curses, "set_escdelay"):
        curses.set_escdelay(50)
    stdscr.keypad(True)
    init_colors()

    try:
        choice = screen_auth_choice(stdscr)
        if choice == "login":
            screen_login(stdscr)
        else:
            screen_signup(stdscr)
        screen_main(stdscr)
    except KeyboardInterrupt:
        pass

    stdscr.clear()
    h, w = stdscr.getmaxyx()
    center_text(stdscr, h // 2, "Goodbye!", curses.color_pair(C_ACCENT) | curses.A_BOLD)
    stdscr.refresh()
    time.sleep(0.6)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DocChat TUI client")
    p.add_argument(
        "--host", default="127.0.0.1", help="dummy AI server host (default %(default)s)"
    )
    p.add_argument(
        "--port",
        type=int,
        default=5555,
        help="dummy AI server port (default %(default)s)",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="skip the server entirely and use the local stub",
    )
    return p.parse_args()


def _setup_ai_client(args: argparse.Namespace) -> None:
    """Build the AI client and probe the server so the chat screen has accurate status."""
    if args.offline:
        state["ai_client"] = None
        state["ai_status"] = "offline (--offline)"
        return
    client = AIClient(host=args.host, port=args.port)
    if client.ping():
        state["ai_client"] = client
        state["ai_status"] = f"online ({args.host}:{args.port})"
    else:
        client.close()
        state["ai_client"] = None
        state["ai_status"] = f"unreachable ({args.host}:{args.port})"


if __name__ == "__main__":
    if curses is None:
        raise RuntimeError(
            "The curses module is not available in this Python environment. "
            "Run the TUI from WSL/Linux/macOS, or install a curses-compatible "
            "Windows environment."
        )
    cli_args = _parse_cli()
    _setup_ai_client(cli_args)
    try:
        curses.wrapper(main)
    finally:
        if state["ai_client"] is not None:
            state["ai_client"].close()
