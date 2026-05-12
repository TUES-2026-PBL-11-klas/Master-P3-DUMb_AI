"""
Custom exception hierarchy

All project exceptions inherit from RAGException so callers can catch
everything with a single ``except RAGException`` while still being able
to handle specific failure modes individually.

Hierarchy:
    RAGException
    ├── UnsupportedFormatError   (parser received an unrecognised extension)
    └── EmbeddingError           (Ollama embedding call failed)
"""


class RAGException(Exception):
    """Base class for all DUMb_AI app exceptions."""


class UnsupportedFormatError(RAGException):
    """
    Raised when a file format has no registered parser.
    """


class EmbeddingError(RAGException):
    """
    Raised when the Ollama BGE-M3 embedding call fails.

    Wraps the underlying network or model error as ``__cause__``.
    """
