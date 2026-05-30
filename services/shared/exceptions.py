"""
Custom exception hierarchy

All project exceptions inherit from RAGException so callers can catch
everything with a single ``except RAGException`` while still being able
to handle specific failure modes individually.

Hierarchy:
    RAGException
    ├── UnsupportedFormatError   (parser received an unrecognised extension)
    ├── EmbeddingError           (llama.cpp embedding call failed)
    ├── BigSentenceError         (chunker received a sentence over the token cap)
    ├── StorageError             (vector store read/write failed)
    └── AuthError                (authentication failed — wrong password, etc.)
"""


class RAGException(Exception):
    """Base class for all DUMb_AI app exceptions."""


class UnsupportedFormatError(RAGException):
    """
    Raised when a file format has no registered parser.
    """


class EmbeddingError(RAGException):
    """
    Raised when the llama.cpp BGE-M3 embedding call fails.

    Wraps the underlying network or model error as ``__cause__``.
    """


class BigSentenceError(RAGException):
    """
    Raised when there is a sentence too large for the chunker to handle.
    """


class StorageError(RAGException):
    """
    Raised when a vector store operation (store / search) fails.

    Wraps the underlying driver error (PyMongo, network, BSON, …) as
    ``__cause__`` so callers can introspect the root cause while still
    catching the high-level ``RAGException``.
    """


class AuthError(RAGException):
    """
    Raised when an authentication attempt fails.

    The message is safe to surface to the end user (e.g. "incorrect
    password"); callers should NOT leak whether it was the username or
    the password that was wrong if they care about user enumeration —
    use a single message like "invalid credentials" instead.
    """
