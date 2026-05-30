"""
TxtParser — DocumentParser[Document] implementation for plain-text .txt files.

Responsibilities:
  - Decode raw file bytes received from the client over the wire.
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
import uuid
from datetime import datetime, timezone
from typing import ClassVar

from services.shared.domain import Document, DocumentStatus
from services.shared.exceptions import RAGException, UnsupportedFormatError

logger = logging.getLogger(__name__)

# Extensions this parser handles (used by ParserRegistry.supported_extensions)
_SUPPORTED: frozenset[str] = frozenset({"txt"})


class TxtParser:
    """
    Concrete Strategy - parses .txt files.

    Satisfies :shared.protocols.DocumentParser[Document] through
    structural subtyping (no explicit inheritance needed).

    Encoding strategy:
        Try the following encodings in order,
        returning the first that successfully decodes the bytes -
        utf-8, utf-8-sig, utf-16, cp1252, latin-1

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
            normalised file text.

        Raises:
            shared.exceptions.UnsupportedFormatError:
                if the filename's extension is not .txt
            shared.exceptions.RAGException:
                for any decoding or unexpected error.
        """
        if not isinstance(raw, (bytes, bytearray)):
            raise RAGException(
                f"TxtParser.parse expected bytes, got {type(raw).__name__}"
            )

        ext = _extension(filename)
        if ext not in _SUPPORTED:
            raise UnsupportedFormatError(
                f"TxtParser does not handle '.{ext}' files. "
                f"Supported: {self.extensions}"
            )

        logger.debug("TxtParser: decoding '%s' (%d bytes)", filename, len(raw))
        text = self._decode(bytes(raw), filename)
        content = self._normalise(text)

        logger.info(
            "TxtParser: parsed '%s' — %d chars, %d lines",
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
                    "TxtParser: decode failed for '%s' with encoding '%s', trying next encoding.",
                    filename,
                    encoding,
                )
                continue

        raise RAGException(
            f"TxtParser: unable to decode '{filename}' with supported encodings"
        ) from last_error

    @staticmethod
    def _normalise(text: str) -> str:
        """
        Normalise whitespace in *text*:

        1. Strip trailing whitespace from every line.
        2. Collapse runs of more than two consecutive blank lines to two.
        3. Ensure the result ends with exactly one newline.
        """
        lines = text.splitlines()

        # Strip trailing whitespace per line (list comprehension - functional style)
        stripped = [line.rstrip() for line in lines]

        # Collapse >2 consecutive blank lines - generator-based fold
        normalised: list[str] = []
        blank_run = 0
        for line in stripped:
            if line == "":
                blank_run += 1
                if blank_run <= 1:  # at most 1 blank line (= 2 newlines when joined)
                    normalised.append(line)
            else:
                blank_run = 0
                normalised.append(line)

        # Guarantee a single trailing newline
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
