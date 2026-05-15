"""
ParserRegistry — runtime selector for DocumentParser implementations.

Design pattern: Strategy
    The registry is the Context object in the Strategy pattern.
    It holds a list of DocumentParser strategies and delegates to the correct one based on extension
    IngestionService never knows which parser is active - it only calls registry.get(ext)

Open/Closed Principle:
    Adding support for a new file format (e.g. PDF, DOCX) requires only:
        1. Writing a new parser class that satisfies DocumentParser[Document]
        2. Calling registry.register(MyNewParser())
    Zero changes to IngestionService or any existing parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from services.shared.exceptions import UnsupportedFormatError
from services.shared.protocols import DocumentParser

T = TypeVar("T")


class ParserRegistry(Generic[T]):
    """
    Maintains an ordered list of shared.protocols.DocumentParser
    implementations and returns the correct one for a given file extension

    Usage:
        registry: ParserRegistry[Document] = ParserRegistry()
        registry.register(TxtParser())

        parser = registry.get("txt")
        document = parser.parse(Path("notes.txt"))

    Attributes:
        _parsers: Internal list of registered parser instances in registration order
        the first parser whose supports() returns True wins.
    """

    def __init__(self) -> None:
        self._parsers: list[DocumentParser[T]] = []

    # Mutation
    def register(self, parser: DocumentParser[T]) -> None:
        """
        Add *parser* to the registry.

        Registered parsers are appended at the end of the internal
        list - they are checked last.  Register more-specific parsers before
        catch-all ones.

        Args:
            parser: Any object that satisfies DocumentParser[T]
        """
        self._parsers.append(parser)

    # Lookup
    def get(self, ext: str) -> DocumentParser[T]:
        """
        Return the first registered parser that supports *ext*.

        Args:
            ext: File extension without a leading dot, lower-cased ("md", "txt").

        Returns:
            A DocumentParser[T] instance.

        Raises:
            shared.exceptions.UnsupportedFormatError: if no registered parser handles *ext*.
        """
        normalised = ext.lstrip(".").lower()

        for parser in self._parsers:
            if normalised in parser.extensions:
                return parser

        raise UnsupportedFormatError(
            f"No parser registered for extension '.{normalised}'. "
            f"Registered parsers: {[type(p).__name__ for p in self._parsers]}"
        )

    def get_for_path(self, path: Path) -> DocumentParser[T]:
        """
        Convenience wrapper - derive the extension from *path* and delegate
        to method get

        Args:
            path: Filesystem path to the file that needs parsing.

        Returns:
            A matching DocumentParser[T]

        Raises:
            shared.exceptions.UnsupportedFormatError: if the extension is not supported or the path has no extension.
        """
        suffix = path.suffix  # includes the dot, e.g. ".txt"

        if not suffix:
            raise UnsupportedFormatError(
                f"Cannot determine file format: '{path}' has no extension."
            )

        return self.get(suffix)

    # Introspection helpers (useful for logging / tests)
    @property
    def supported_extensions(self) -> list[str]:
        """
        Return a sorted list of all extensions supported by registered parsers.

        Parsers may optionally expose an extensions attribute containing
        their canonical supported extensions. This property aggregates those
        values across all registered parsers.

        Primarily useful for debugging, logging, tests, and diagnostics.
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
