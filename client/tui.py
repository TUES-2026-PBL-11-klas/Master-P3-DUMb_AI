#!/usr/bin/env python3
"""
DocChat TUI - AI-powered document chat interface (stub/prototype)

Login always succeeds and uploads are tracked in-memory. AI replies are
fetched from the dummy socket server in ``services.dummy_server`` over a
newline-delimited JSON protocol; if the server is unreachable, the chat
screen falls back to a local stub so the UI still works offline.

Run the server in one terminal:
    python -m services.dummy_server

Then launch the TUI in another:
    python -m client.tui
    python -m client.tui --host 127.0.0.1 --port 5555    # custom server
    python -m client.tui --offline                       # force local stub
"""

import argparse
import curses
import os
import sys
import time
import textwrap
from datetime import datetime

# Make ``python client/tui.py`` work as well as ``python -m client.tui``.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from client.ai_client import AIClient, AIClientError

# ── In-memory state ─────────────────────────────────────────────────────────
state = {
    "logged_in":      False,
    "username":       "",
    "documents":      [],   # {"name", "size", "uploaded_at"}
    "messages":       [],   # {"role": "user"|"ai", "text", "time"}
    "current_screen": "login",
    "ai_client":      None, # AIClient or None when running --offline
    "ai_status":      "offline",  # "online" | "offline" | "error: ..."
}

STUB_AI_RESPONSES = [
    "[offline] Based on your documents, here is a placeholder answer.",
    "[offline] The documents mention several relevant points. (Server not connected.)",
    "[offline] This will be answered by the real AI once the server is reachable.",
    "[offline] Interesting question! Start the dummy server to see real replies.",
]
_stub_index = 0

def stub_ai_response():
    """Local fallback used when the dummy server is unreachable."""
    global _stub_index
    r = STUB_AI_RESPONSES[_stub_index % len(STUB_AI_RESPONSES)]
    _stub_index += 1
    return r

def fetch_ai_response(text):
    """
    Ask the dummy server for a reply, with graceful fallback.

    Updates ``state["ai_status"]`` so the chat screen can show the
    current connection state. Always returns a string -- never raises.
    """
    client = state.get("ai_client")
    if client is None:
        state["ai_status"] = "offline"
        return stub_ai_response()
    try:
        answer = client.ask(text, username=state.get("username") or "anonymous")
        state["ai_status"] = "online"
        return answer
    except AIClientError as exc:
        state["ai_status"] = f"error: {exc}"
        return f"[connection lost] {exc}  (falling back to offline stub)"

# ── Color pair IDs ───────────────────────────────────────────────────────────
C_NORMAL   = 1
C_HEADER   = 2
C_ACCENT   = 3
C_DIM      = 4
C_ERROR    = 5
C_SUCCESS  = 6
C_USER_MSG = 7
C_AI_MSG   = 8

# ── Safe wrappers ────────────────────────────────────────────────────────────
def safe_curs_set(v):
    try:
        curses.curs_set(v)
    except curses.error:
        pass

def safe_addstr(win, y, x, s, attr=0):
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

def init_colors():
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK

    curses.init_pair(C_NORMAL,   curses.COLOR_WHITE,  bg)
    curses.init_pair(C_HEADER,   curses.COLOR_BLACK,  curses.COLOR_CYAN)
    curses.init_pair(C_ACCENT,   curses.COLOR_CYAN,   bg)
    curses.init_pair(C_DIM,      curses.COLOR_WHITE,  bg)
    curses.init_pair(C_ERROR,    curses.COLOR_RED,    bg)
    curses.init_pair(C_SUCCESS,  curses.COLOR_GREEN,  bg)
    curses.init_pair(C_USER_MSG, curses.COLOR_CYAN,   bg)
    curses.init_pair(C_AI_MSG,   curses.COLOR_YELLOW, bg)

# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_header(win, title):
    h, w = win.getmaxyx()
    text = f"  DocChat  --  {title}  "
    safe_addstr(win, 0, 0, text.ljust(w - 1),
                curses.color_pair(C_HEADER) | curses.A_BOLD)

def draw_footer(win, hints):
    h, w = win.getmaxyx()
    text = "   ".join(hints)
    safe_addstr(win, h - 1, 0, text.ljust(w - 1), curses.color_pair(C_HEADER))

def center_text(win, row, text, attr=0):
    _, w = win.getmaxyx()
    col = max(0, (w - len(text)) // 2)
    safe_addstr(win, row, col, text, attr)

def draw_box(win, y, x, h, w, title=""):
    """Plain ASCII box - works on every Windows terminal."""
    safe_addstr(win, y,     x,     "+")
    safe_addstr(win, y,     x+w-1, "+")
    safe_addstr(win, y+h-1, x,     "+")
    safe_addstr(win, y+h-1, x+w-1, "+")
    for i in range(1, w - 1):
        safe_addstr(win, y,     x+i, "-")
        safe_addstr(win, y+h-1, x+i, "-")
    for i in range(1, h - 1):
        safe_addstr(win, y+i, x,     "|")
        safe_addstr(win, y+i, x+w-1, "|")
    if title:
        safe_addstr(win, y, x + 2, f" {title} ",
                    curses.color_pair(C_ACCENT) | curses.A_BOLD)

def read_line(win, y, x, max_len, mask=False):
    curses.echo()
    safe_curs_set(1)
    try:
        win.move(y, x)
    except curses.error:
        pass
    win.refresh()
    buf = []
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

def screen_login(stdscr):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    draw_header(stdscr, "Login")
    draw_footer(stdscr, ["Enter to confirm", "Ctrl+C to quit"])

    center_text(stdscr, 2, "Welcome to DocChat",
                curses.color_pair(C_ACCENT) | curses.A_BOLD)
    center_text(stdscr, 3, "AI-powered document Q&A",
                curses.color_pair(C_DIM) | curses.A_DIM)

    bw, bh = 42, 11
    bx = max(0, (w - bw) // 2)
    by = max(1, (h - bh) // 2)
    draw_box(stdscr, by, bx, bh, bw, "Sign In")

    safe_addstr(stdscr, by + 2, bx + 3, "Username:", curses.color_pair(C_ACCENT))
    safe_addstr(stdscr, by + 5, bx + 3, "Password:", curses.color_pair(C_ACCENT))
    safe_addstr(stdscr, by + 9, bx + 3, "(any credentials work -- stub mode)",
                curses.color_pair(C_DIM) | curses.A_DIM)
    stdscr.refresh()

    username = read_line(stdscr, by + 3, bx + 3, bw - 6)
    read_line(stdscr, by + 6, bx + 3, bw - 6, mask=True)

    state["username"]  = username or "guest"
    state["logged_in"] = True

    safe_addstr(stdscr, by + 7, bx + 3, "Logged in!",
                curses.color_pair(C_SUCCESS) | curses.A_BOLD)
    stdscr.refresh()
    time.sleep(0.7)
    state["current_screen"] = "main"


def screen_main(stdscr):
    """Main menu with >> arrow on selected item."""
    menu_items = [
        ("1", "Chat with AI",     "chat",   "Ask questions about your documents"),
        ("2", "Manage Documents", "upload", "Upload and view your files"),
        ("Q", "Quit",             "quit",   ""),
    ]
    selected = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, f"Main Menu  [{state['username']}]")
        draw_footer(stdscr, ["Up/Down to navigate", "Enter to select", "Q to quit"])

        center_text(stdscr, 2, "What would you like to do?",
                    curses.color_pair(C_ACCENT) | curses.A_BOLD)

        col = max(4, w // 2 - 18)
        for i, (key, label, _, desc) in enumerate(menu_items):
            row    = 5 + i * 3
            is_sel = (i == selected)

            arrow      = ">>  " if is_sel else "    "
            attr_arrow = curses.color_pair(C_ACCENT) | curses.A_BOLD
            attr_key   = (curses.color_pair(C_ACCENT) | curses.A_BOLD) if is_sel \
                         else curses.color_pair(C_DIM) | curses.A_DIM
            attr_label = (curses.color_pair(C_NORMAL) | curses.A_BOLD) if is_sel \
                         else curses.color_pair(C_NORMAL)
            attr_desc  = curses.color_pair(C_DIM) | curses.A_DIM

            safe_addstr(stdscr, row, col,      arrow,         attr_arrow if is_sel else attr_desc)
            safe_addstr(stdscr, row, col + 4,  f"[{key}]  ", attr_key)
            safe_addstr(stdscr, row, col + 9,  label,         attr_label)
            if desc:
                safe_addstr(stdscr, row + 1, col + 9, desc,  attr_desc)

        doc_count = len(state["documents"])
        safe_addstr(stdscr, h - 3, 2, f"Documents in store: {doc_count}",
                    curses.color_pair(C_DIM) | curses.A_DIM)
        stdscr.refresh()

        dest = None
        ch   = stdscr.getch()

        if ch == curses.KEY_UP:
            selected = (selected - 1) % len(menu_items)
        elif ch == curses.KEY_DOWN:
            selected = (selected + 1) % len(menu_items)
        elif ch in (10, 13):
            dest = menu_items[selected][2]
        elif 0 < ch < 256 and chr(ch).lower() in ("1", "2", "q"):
            mapping = {"1": "chat", "2": "upload", "q": "quit"}
            dest    = mapping[chr(ch).lower()]

        if dest == "quit":
            return False
        elif dest == "chat":
            screen_chat(stdscr)
        elif dest == "upload":
            screen_documents(stdscr)


def screen_documents(stdscr):
    msg      = ""
    selected = 0

    while True:
        stdscr.clear()
        h, w = stdscr.getmaxyx()
        draw_header(stdscr, "Document Manager")
        draw_footer(stdscr, ["U upload", "D delete", "B back", "Up/Down navigate"])

        safe_addstr(stdscr, 2, 2, "Your uploaded documents:",
                    curses.color_pair(C_ACCENT) | curses.A_BOLD)

        docs     = state["documents"]
        list_top = 4
        list_h   = h - 8

        for i, doc in enumerate(docs[:list_h]):
            row     = list_top + i
            size_kb = doc["size"] / 1024
            is_sel  = (i == selected)
            arrow   = ">> " if is_sel else "   "
            line    = f"{doc['name']:<28}  {size_kb:>6.1f} KB   {doc['uploaded_at']}"
            attr    = (curses.color_pair(C_ACCENT) | curses.A_BOLD) if is_sel \
                      else curses.color_pair(C_NORMAL)
            safe_addstr(stdscr, row, 2, arrow + line, attr)

        if not docs:
            safe_addstr(stdscr, list_top, 2, "  (no documents yet)",
                        curses.color_pair(C_DIM) | curses.A_DIM)
        if msg:
            safe_addstr(stdscr, h - 4, 2, msg,
                        curses.color_pair(C_SUCCESS) | curses.A_BOLD)
        stdscr.refresh()

        ch = stdscr.getch()
        c  = chr(ch).lower() if 0 < ch < 256 else ""

        if ch == curses.KEY_UP and docs:
            selected = max(0, selected - 1)
        elif ch == curses.KEY_DOWN and docs:
            selected = min(len(docs) - 1, selected + 1)
        elif c == "u":
            safe_addstr(stdscr, h - 6, 2, " " * (w - 4))
            safe_addstr(stdscr, h - 6, 2, "File path to upload:",
                        curses.color_pair(C_ACCENT))
            stdscr.refresh()
            path = read_line(stdscr, h - 5, 2, w - 4).strip()
            if path and os.path.isfile(path):
                name = os.path.basename(path)
                size = os.path.getsize(path)
                state["documents"].append({
                    "name":        name,
                    "size":        size,
                    "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                msg = f"'{name}' uploaded (stub - not sent to DB)"
            elif path:
                msg = f"File not found: {path}"
            else:
                msg = ""
        elif c == "d" and docs:
            removed  = docs.pop(selected)
            selected = max(0, selected - 1)
            msg      = f"'{removed['name']}' removed (stub)"
        elif c == "b":
            break


def screen_chat(stdscr):
    input_buf     = ""
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
            ts   = m["time"]
            role = m["role"]
            if role == "user":
                prefix = f"  You [{ts}]: "
                attr   = curses.color_pair(C_USER_MSG) | curses.A_BOLD
            else:
                prefix = f"  AI  [{ts}]: "
                attr   = curses.color_pair(C_AI_MSG)
            wrap_w  = max(1, w - 4 - len(prefix))
            wrapped = textwrap.wrap(m["text"], wrap_w) or [""]
            for j, line in enumerate(wrapped):
                pfx = prefix if j == 0 else " " * len(prefix)
                all_lines.append((attr, pfx + line))
            all_lines.append((curses.color_pair(C_NORMAL), ""))

        total_lines   = len(all_lines)
        max_scroll    = max(0, total_lines - msg_area_h)
        scroll_offset = min(scroll_offset, max_scroll)
        view_start    = max(0, total_lines - msg_area_h - scroll_offset)

        for i, (attr, text) in enumerate(all_lines[view_start:view_start + msg_area_h]):
            safe_addstr(stdscr, 1 + i, 2, text, attr)

        if not state["messages"]:
            safe_addstr(stdscr, 3, 4,
                        "No messages yet. Ask something about your documents!",
                        curses.color_pair(C_DIM) | curses.A_DIM)

        safe_addstr(stdscr, h - 4, 0, "-" * (w - 1), curses.color_pair(C_ACCENT))
        safe_addstr(stdscr, h - 3, 2, "> ",
                    curses.color_pair(C_ACCENT) | curses.A_BOLD)
        input_display = input_buf[-(w - 7):]
        safe_addstr(stdscr, h - 3, 4, input_display, curses.color_pair(C_NORMAL))

        if scroll_offset > 0:
            safe_addstr(stdscr, h - 2, 2,
                        f"^ scrolled up {scroll_offset} line(s) -- PgDn to go down",
                        curses.color_pair(C_DIM) | curses.A_DIM)

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
                input_buf     = ""
                scroll_offset = 0

                # Show a "thinking" hint while we wait on the server.
                safe_addstr(stdscr, h - 2, 2, "AI is thinking...",
                            curses.color_pair(C_DIM) | curses.A_DIM)
                stdscr.refresh()

                answer = fetch_ai_response(text)
                reply_time = datetime.now().strftime("%H:%M")
                state["messages"].append({"role": "ai", "text": answer, "time": reply_time})
        elif ch in (ord("b"), ord("B")) and not input_buf:
            break
        elif 32 <= ch < 127:
            input_buf += chr(ch)


# ── Entry point ──────────────────────────────────────────────────────────────

def main(stdscr):
    safe_curs_set(0)
    if hasattr(curses, "set_escdelay"):
        curses.set_escdelay(50)
    stdscr.keypad(True)
    init_colors()

    try:
        screen_login(stdscr)
        screen_main(stdscr)
    except KeyboardInterrupt:
        pass

    stdscr.clear()
    h, w = stdscr.getmaxyx()
    center_text(stdscr, h // 2, "Goodbye!",
                curses.color_pair(C_ACCENT) | curses.A_BOLD)
    stdscr.refresh()
    time.sleep(0.6)


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DocChat TUI client")
    p.add_argument("--host", default="127.0.0.1",
                   help="dummy AI server host (default %(default)s)")
    p.add_argument("--port", type=int, default=5555,
                   help="dummy AI server port (default %(default)s)")
    p.add_argument("--offline", action="store_true",
                   help="skip the server entirely and use the local stub")
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
    cli_args = _parse_cli()
    _setup_ai_client(cli_args)
    try:
        curses.wrapper(main)
    finally:
        if state["ai_client"] is not None:
            state["ai_client"].close()