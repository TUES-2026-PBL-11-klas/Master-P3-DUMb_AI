from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from services.shared.domain import IngestionEvent

# Generic type variable used across the parser and store protocols.
# T is intentionally unbound so callers can specialise it (e.g. Document, Chunk, str)
T_co = TypeVar("T_co", covariant=True)  # for DocumentParser (return-only)
T_inv = TypeVar("T_inv")  # for VectorStore (input + output)


# DocumentParser[T]
#
# Strategy interface — each concrete parser handles one (or more) file types
# and converts raw file bytes into a typed domain object T
# The registry dispatches to the right parser based on file extension
@runtime_checkable
class DocumentParser(Protocol[T_co]):
    """
    Strategy interface for file-format-specific document parsers.

    Type parameter T is the domain object produced by ``parse()``.
    In the current system T == Document, but the generic bound keeps the
    protocol reusable for future pipeline stages.
    """

    extensions: ClassVar[
        tuple[str, ...]
    ]  # Attribute listing canonical supported extensions ("txt", "md")

    def parse(self, path: Path) -> T_co:
        """
        Read *path* from disk and return a fully-constructed domain object.

        Raises:
            UnsupportedFormatError: if the file extension is not handled by
            this parser.
            RAGException: for any other parsing failure.
        """
        ...


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

    def on_ingest(self, event: "IngestionEvent") -> None:  # noqa: F821
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
