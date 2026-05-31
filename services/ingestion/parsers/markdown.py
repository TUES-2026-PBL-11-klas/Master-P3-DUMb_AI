"""
MarkdownParser — DocumentParser[Document] implementation for Markdown .md files.

Responsibilities:
  - Decode raw file bytes received from the client over the wire.
  - Strip Markdown syntax (headings, emphasis, links, images, code fences,
    blockquotes, horizontal rules, HTML tags) to produce clean plain text.
  - Normalise whitespace (collapse blank lines, strip trailing spaces).
  - Construct and return a shared.domain.Document instance.

This parser intentionally does NO chunking — that is ChunkingObserver's job.
The parser only extracts the full plain-text content and populates metadata.

Network architecture note:
    The server never sees the client's local disk. The TUI reads the file
    locally, sends raw bytes + filename over the persistent socket, and
    this parser receives those bytes directly. There is no filesystem
    access in this module.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import ClassVar

from services.shared.domain import Document, DocumentStatus
from services.shared.exceptions import RAGException, UnsupportedFormatError

logger = logging.getLogger(__name__)

# Extensions this parser handles (used by ParserRegistry.supported_extensions)
_SUPPORTED: frozenset[str] = frozenset({"md", "markdown", "mkd", "mkdn", "mdown"})

# Compiled regexes for Markdown stripping
# Order matters: fenced code blocks must be handled before inline patterns.

_RE_FENCED_CODE = re.compile(
    r"^```[^\n]*\n(.*?)^```",
    re.MULTILINE | re.DOTALL,
)
_RE_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")  # ![alt](url)
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url)
_RE_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")  # [text][ref]
_RE_REF_DEF = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)  # [ref]: url
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)  # ## heading
_RE_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*")  # **bold**
_RE_BOLD_UNDER = re.compile(r"__(.+?)__")  # __bold__
_RE_ITALIC_STAR = re.compile(r"\*(.+?)\*")  # *italic*
_RE_ITALIC_UNDER = re.compile(r"_(.+?)_")  # _italic_
_RE_STRIKETHROUGH = re.compile(r"~~(.+?)~~")  # ~~strike~~
_RE_INLINE_CODE = re.compile(r"`([^`]+)`")  # `code`
_RE_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)  # > quote
_RE_HR = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)  # --- or *** or ___
_RE_UL_BULLET = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)  # - item
_RE_OL_BULLET = re.compile(r"^(\s*)\d+\.\s+", re.MULTILINE)  # 1. item
_RE_HTML_TAG = re.compile(r"<[^>]+>")  # <tag ...>
_RE_MULTI_BLANK = re.compile(r"\n{3,}")  # 3+ newlines → 2


class MarkdownParser:
    """
    Concrete Strategy — parses .md / .markdown files.

    Satisfies shared.protocols.DocumentParser[Document] through
    structural subtyping (no explicit inheritance needed).

    Markdown stripping:
        The parser removes syntactic markers so downstream pipeline stages
        (chunking → embedding → retrieval) operate on clean prose. Fenced
        code blocks are preserved as plain text (the code itself is
        meaningful content) but the fence markers are removed.

    Encoding strategy:
        Identical to TxtParser - try utf-8, utf-8-sig, utf-16, cp1252,
        latin-1 in that order.

    Whitespace normalisation:
        - Trailing spaces/tabs on every line are stripped.
        - More than two consecutive blank lines are collapsed to two.
        - A single trailing newline is ensured.
    """

    # Expose supported extensions for ParserRegistry introspection.
    extensions: ClassVar[tuple[str, ...]] = tuple(_SUPPORTED)

    # DocumentParser[Document] interface

    def parse(self, raw: bytes | bytearray, filename: str) -> Document:
        """
        Decode *raw* bytes and return a shared.domain.Document.

        Args:
            raw:      The raw file bytes as received from the client.
            filename: The original filename supplied by the client (used
                      to validate the extension and to populate
                      Document.filename).

        Returns:
            A shared.domain.Document with content set to the
            Markdown-stripped, normalised file text.

        Raises:
            shared.exceptions.UnsupportedFormatError:
                if the filename's extension is not one of
                .md, .markdown, .mkd, .mkdn, .mdown
            shared.exceptions.RAGException:
                for any decoding or unexpected error.
        """
        if not isinstance(raw, (bytes, bytearray)):
            raise RAGException(
                f"MarkdownParser.parse expected bytes, got {type(raw).__name__}"
            )

        ext = _extension(filename)
        if ext not in _SUPPORTED:
            raise UnsupportedFormatError(
                f"MarkdownParser does not handle '.{ext}' files. "
                f"Supported: {self.extensions}"
            )

        logger.debug("MarkdownParser: decoding '%s' (%d bytes)", filename, len(raw))
        text = self._decode(bytes(raw), filename)
        stripped = self._strip_markdown(text)
        content = self._normalise(stripped)

        logger.info(
            "MarkdownParser: parsed '%s' — %d chars, %d lines",
            filename,
            len(content),
            content.count("\n"),
        )

        return Document(
            id=uuid.uuid4(),
            # user_id is injected by IngestionService after authentication;
            # we use a nil UUID as a sentinel here.
            user_id=uuid.UUID(
                int=0
            ),  # 0 is an invalid user ID, to be replaced later in the ingestion flow
            content=content,
            filename=filename,
            uploaded_at=datetime.now(tz=timezone.utc),
            status=DocumentStatus.PARSED,
        )

    # Private helpers
    @staticmethod
    def _decode(raw: bytes, filename: str) -> str:
        """
        Decode *raw* trying multiple encodings in order.

        Raises:
            shared.exceptions.RAGException: if every supported encoding fails.
        """
        encodings = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")

        last_error: UnicodeDecodeError | None = None

        for encoding in encodings:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError as e:
                last_error = e
                logger.debug(
                    "MarkdownParser: decode failed for '%s' with encoding '%s', trying next encoding.",
                    filename,
                    encoding,
                )
                continue

        raise RAGException(
            f"MarkdownParser: unable to decode '{filename}' with supported encodings"
        ) from last_error

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """
        Remove Markdown syntax from *text*, preserving the prose content.

        Processing order:
            1. Fenced code blocks  — remove ``` fences, keep inner text.
            2. Images              — replace with alt text.
            3. Links               — replace with link text.
            4. Reference-style definitions — remove entirely.
            5. Headings            — strip leading #'s.
            6. Emphasis / strike   — unwrap markers.
            7. Inline code         — unwrap backticks.
            8. Blockquotes         — strip leading >.
            9. Horizontal rules    — remove.
           10. List bullets        — strip markers, keep indentation.
           11. HTML tags           — remove.
        """
        result = text

        # 1. Fenced code blocks: keep content, drop fences
        result = _RE_FENCED_CODE.sub(r"\1", result)

        # 2. Images → alt text (before links, since image syntax contains link syntax)
        result = _RE_IMAGE.sub(r"\1", result)

        # 3. Links → link text
        result = _RE_LINK.sub(r"\1", result)
        result = _RE_REF_LINK.sub(r"\1", result)

        # 4. Reference definitions — remove whole line
        result = _RE_REF_DEF.sub("", result)

        # 5. Headings — strip leading hashes
        result = _RE_HEADING.sub("", result)

        # 6. Emphasis / strikethrough — unwrap (order: bold before italic)
        result = _RE_BOLD_STAR.sub(r"\1", result)
        result = _RE_BOLD_UNDER.sub(r"\1", result)
        result = _RE_ITALIC_STAR.sub(r"\1", result)
        result = _RE_ITALIC_UNDER.sub(r"\1", result)
        result = _RE_STRIKETHROUGH.sub(r"\1", result)

        # 7. Inline code — unwrap backticks
        result = _RE_INLINE_CODE.sub(r"\1", result)

        # 8. Blockquotes
        result = _RE_BLOCKQUOTE.sub("", result)

        # 9. Horizontal rules
        result = _RE_HR.sub("", result)

        # 10. List bullets — keep indentation, strip marker
        result = _RE_UL_BULLET.sub(r"\1", result)
        result = _RE_OL_BULLET.sub(r"\1", result)

        # 11. HTML tags
        result = _RE_HTML_TAG.sub("", result)

        return result

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalise whitespace in *text*:

        1. Strip trailing whitespace from every line.
        2. Collapse runs of more than two consecutive blank lines to two.
        3. Ensure the result ends with exactly one newline.
        """
        lines = text.splitlines()

        # Strip trailing whitespace per line (list comprehension — functional style)
        stripped = [line.rstrip() for line in lines]

        # Collapse >2 consecutive blank lines - generator-based fold
        normalised: list[str] = []
        blank_run = 0
        for line in stripped:
            if line == "":
                blank_run += 1
                if blank_run <= 1:
                    normalised.append(line)
            else:
                blank_run = 0
                normalised.append(line)

        result = "\n".join(normalised)
        return result if result.endswith("\n") else result + "\n"


def _extension(filename: str) -> str:
    """
    Lower-case extension *without* the leading dot.

    Uses os.path.basename so a client that includes a leading directory
    component (which is metadata, never used for I/O) does not throw off
    extension detection.
    """
    base = os.path.basename(filename)
    _, dot_ext = os.path.splitext(base)
    return dot_ext.lstrip(".").lower()
