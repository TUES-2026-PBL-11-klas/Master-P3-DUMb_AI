from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from services.shared.domain import IngestionEvent


# Generic type variable used across the parser and store protocols.
# T is intentionally unbound so callers can specialise it (e.g. Document, Chunk, str)
T_co = TypeVar("T_co", covariant=True)
T_inv = TypeVar("T_inv")
# DocumentParser[T]
#
# Strategy interface — each concrete parser handles one (or more) file types
# and converts raw file bytes into a typed domain object T.
#
# Parsers operate on in-memory bytes, not filesystem paths: the server
# receives uploads over a socket and never sees the client's local disk. The
# filename is carried alongside the bytes only for traceability — it ends up
# in Document.filename and eventually in Chunk.metadata["source_file"].


@runtime_checkable
class AIInterface(Protocol):
    """Structural interface for any embedding backend."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class DocumentParser(Protocol[T_co]):
    """
    Strategy interface for file-format-specific document parsers.
    Type parameter T is the domain object produced by ``parse()``.
    In the current system T == Document, but the generic bound keeps the
    protocol reusable for future pipeline stages.
    """

    extensions: ClassVar[tuple[str, ...]]

    """
            Decode *raw* and return a fully-constructed domain object.

            Args:
                raw:      The raw file bytes as received from the client over
                          the wire. Parsers are responsible for character-set
                          decoding (each parser carries its own encoding
                          fallback ladder).
                filename: The original filename supplied by the client. Used
                          for extension validation, for the Document.filename
                          field, and for log/error messages. Parsers MUST NOT
                          treat this as a filesystem path — it is metadata.
                          Raises:
                                      UnsupportedFormatError: if the filename's extension is not
                                          handled by this parser.
                                      RAGException: for any decoding or unexpected error.
                                  """

    def parse(self, raw: bytes | bytearray, filename: str) -> T_co: ...


# IngestionObserver
#
# Observer interface — each observer implements one step of the ingestion
# pipeline (chunking, embedding, storage).  IngestionService is the Subject.
@runtime_checkable
class IngestionObserver(Protocol):
    """
    Observer interface for the ingestion pipeline.
    Implementations receive a shared.domain.IngestionEvent and
    carry out exactly one responsibility (chunk / embed / store).
    """

    def on_ingest(self, event: "IngestionEvent") -> None:
        """
        Called by ingestion.service.IngestionService once for every
        document that enters the pipeline.

        The observer may mutate *event* in-place (e.g. attach generated
        chunks or embeddings) so that downstream observers can use them.
        """
        ...


# VectorStore[T]
#
# Generic interface for any vector-capable backing store.
# MongoVectorStore[Chunk] is the production implementation; an in-memory
# stub can satisfy the same protocol for unit tests.
@runtime_checkable
class VectorStore(Protocol[T_inv]):
    """Generic vector store - store chunks, search by embedding."""

    def store(self, chunks: list[T_inv]) -> None:
        """Persist *chunks* (text + embedding + metadata) to the store."""
        ...

    def search(self, vec: list[float], k: int) -> list[T_inv]:
        """
        Return the *k* nearest neighbours to *vec* (cosine similarity).
        Results are ordered by descending similarity score.
        """
        ...


@runtime_checkable
class LlamaCppClient(Protocol):
    """Structural interface for the llama.cpp embedding client."""

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per entry in *texts*."""
        ...


@runtime_checkable
class IngestionServiceProtocol(Protocol):
    """Structural interface for an ingestion backend used by dummy_server."""

    @property
    def supported_extensions(self) -> list[str]: ...

    def ingest(
        self, filename: str, raw: bytes, user_id: uuid.UUID
    ) -> "IngestionEvent": ...
