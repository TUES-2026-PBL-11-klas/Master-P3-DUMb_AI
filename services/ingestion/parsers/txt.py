"""
TxtParser — DocumentParser[Document] implementation for plain-text .txt files.

Responsibilities:
  - Read a .txt file from disk.
  - Normalise whitespace (collapse blank lines, strip trailing spaces)
  - Construct and return a shared.domain.Document instance.

This parser intentionally does NO chunking — that is ChunkingObserver's job.
The parser only extracts the full plain-text content and populates metadata.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from services.shared.domain import Document, DocumentStatus
from services.shared.exceptions import RAGException

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
        returning the first that successfully decodes the file -
        utf-8, utf-8-sig, utf-16, cp1252, latin-1

    Whitespace normalisation:
        - Trailing spaces/tabs on every line are stripped.
        - More than two consecutive blank lines are collapsed to two.
        - A single trailing newline is ensured.
    """

    # Expose supported extensions for ParserRegistry introspection.
    extensions: tuple[str, ...] = tuple(_SUPPORTED)

    # DocumentParser[Document] interface
    def parse(self, path: Path) -> Document:
        """
        Read *path* and return a shared.domain.Document.

        Args:
            path: Absolute or relative path to a .txt file.

        Returns:
            A shared.domain.Document with content set to the
            normalised file text

        Raises:
            shared.exceptions.UnsupportedFormatError: if the file extension is not .txt
            shared.exceptions.RAGException: for any I/O or unexpected error encountered while reading the file.
        """

        if not path.exists():
            raise RAGException(f"File does not exist: {path}")

        if not path.is_file():
            raise RAGException(f"Path is not a regular file: {path}")

        if path.suffix[1:].lower() not in _SUPPORTED:
            from services.shared.exceptions import UnsupportedFormatError

            raise UnsupportedFormatError(
                f"TxtParser does not handle '{path.suffix}' files. "
                f"Supported: {self.extensions}"
            )

        logger.debug("TxtParser: reading '%s'", path)
        raw = self._read_file(path)
        content = self._normalise(raw)

        logger.info(
            "TxtParser: parsed '%s' — %d chars, %d lines",
            path.name,
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
            filename=path.name,
            uploaded_at=datetime.now(tz=timezone.utc),
            content_type="text/plain",
            status=DocumentStatus.PARSED,
        )

    # Private helpers
    @staticmethod
    def _read_file(path: Path) -> str:
        """
        Read *path* trying multiple encodings in order.

        Raises:
            shared.exceptions.RAGException: wraps any OSError (file not found, permission denied, etc.).
        """

        encodings = ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1")

        last_error: UnicodeDecodeError | None = None

        for encoding in encodings:
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError as e:
                last_error = e
                logger.debug(
                    "TxtParser: decode failed for '%s' with encoding '%s', trying next encoding.",
                    path,
                    encoding,
                )
                continue

        # If we exhausted all encodings, raise an error.
        raise RAGException(
            f"TxtParser: unable to decode '{path}' with supported encodings"
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
