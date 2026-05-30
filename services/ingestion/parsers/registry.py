"""
ParserRegistry — runtime selector for DocumentParser implementations.

Design pattern: Strategy
    The registry is the Context object in the Strategy pattern.
    It holds a list of DocumentParser strategies and delegates to the
    correct one based on extension. IngestionService never knows which
    parser is active — it only calls registry.get(ext) or
    registry.get_for_filename(filename).

Open/Closed Principle:
    Adding support for a new file format (e.g. PDF, DOCX) requires only:
        1. Writing a new parser class that satisfies DocumentParser[Document]
        2. Calling registry.register(MyNewParser())
    Zero changes to IngestionService or any existing parser.

Network architecture note:
    All operations are filename-based, not path-based. The server receives
    uploads as (filename, bytes) over a socket and never has a real path
    to dispatch on.
"""

from __future__ import annotations

import os
from typing import Generic, TypeVar

from services.shared.exceptions import UnsupportedFormatError
from services.shared.protocols import DocumentParser

T = TypeVar("T")


class ParserRegistry(Generic[T]):
    """
    Maintains an ordered list of shared.protocols.DocumentParser
    implementations and returns the correct one for a given file extension.

    Usage:
        registry: ParserRegistry[Document] = ParserRegistry()
        registry.register(TxtParser())

        parser = registry.get("txt")
        document = parser.parse(raw_bytes, "notes.txt")

    Attributes:
        _parsers: Internal list of registered parser instances in
                  registration order — the first parser whose
                  ``extensions`` contains the requested extension wins.
    """

    def __init__(self) -> None:
        self._parsers: list[DocumentParser[T]] = []

    # Mutation
    def register(self, parser: DocumentParser[T]) -> None:
        """
        Add *parser* to the registry.

        Registered parsers are appended at the end of the internal
        list — they are checked last. Register more-specific parsers
        before catch-all ones.

        Args:
            parser: Any object that satisfies DocumentParser[T]
        """
        self._parsers.append(parser)

    # Lookup
    def get(self, ext: str) -> DocumentParser[T]:
        """
        Return the first registered parser that supports *ext*.

        Args:
            ext: File extension, lower-cased. A leading dot is tolerated
                 ("txt" and ".txt" both work).

        Returns:
            A DocumentParser[T] instance.

        Raises:
            shared.exceptions.UnsupportedFormatError:
                if no registered parser handles *ext*.
        """
        normalised = ext.lstrip(".").lower()

        for parser in self._parsers:
            if normalised in parser.extensions:
                return parser

        raise UnsupportedFormatError(
            f"No parser registered for extension '.{normalised}'. "
            f"Registered parsers: {[type(p).__name__ for p in self._parsers]}"
        )

    def get_for_filename(self, filename: str) -> DocumentParser[T]:
        """
        Convenience wrapper — derive the extension from *filename* and
        delegate to :meth:`get`.

        Args:
            filename: Original filename supplied by the client (e.g.
                      ``"notes.md"``). Any leading directory component
                      is stripped before the extension is read, so a
                      client that accidentally sends ``"~/dir/notes.md"``
                      still resolves to the markdown parser.

        Returns:
            A matching DocumentParser[T].

        Raises:
            shared.exceptions.UnsupportedFormatError:
                if the extension is not supported or *filename* has none.
        """
        base = os.path.basename(filename)
        _, dot_ext = os.path.splitext(base)

        if not dot_ext:
            raise UnsupportedFormatError(
                f"Cannot determine file format: '{filename}' has no extension."
            )

        return self.get(dot_ext)

    # Introspection helpers (useful for logging / tests)
    @property
    def supported_extensions(self) -> list[str]:
        """
        Return a sorted list of all extensions supported by registered parsers.

        Parsers expose an ``extensions`` attribute containing their
        canonical supported extensions. This property aggregates those
        values across all registered parsers and is primarily useful for
        the WebSocket layer's early-rejection check.
        """
        result: list[str] = []
        for parser in self._parsers:
            result.extend(parser.extensions)
        return sorted(set(result))

    def __len__(self) -> int:
        return len(self._parsers)

    def __repr__(self) -> str:
        names = [type(p).__name__ for p in self._parsers]
        return f"ParserRegistry(parsers={names})"
