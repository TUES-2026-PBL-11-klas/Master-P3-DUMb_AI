"""
StorageObserver — IngestionObserver that persists chunks to a vector store.

Responsibilities:
  - Receive an IngestionEvent whose chunks already carry text + embeddings
    (produced by ChunkingObserver and EmbeddingObserver, in that order).
  - Delegate persistence to an injected VectorStore[Chunk].
  - Advance event.status to STORED on success, or mark it FAILED on error.

This observer MUST run after EmbeddingObserver in the observer chain —
attempting to store a chunk without an embedding is a programming error
and is rejected at the store layer with StorageError.

Design pattern: Observer
    Implements shared.protocols.IngestionObserver through structural
    subtyping (no explicit inheritance needed).
    The store dependency is the VectorStore Protocol, never a concrete
    class — this is what lets the unit tests run with a MagicMock and
    lets future backends drop in unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from services.shared.domain import DocumentStatus, IngestionStatus
from services.shared.exceptions import StorageError

if TYPE_CHECKING:
    from services.shared.domain import Chunk, IngestionEvent
    from services.shared.protocols import VectorStore

logger = logging.getLogger(__name__)


class StorageObserver:
    """
    Observer that persists embedded Chunk objects to a vector store.

    Satisfies shared.protocols.IngestionObserver through structural
    subtyping.

    Attributes:
        _store: Any object satisfying VectorStore[Chunk] —
                MongoVectorStore in production, a stub or MagicMock
                in tests.
    """

    def __init__(self, store: "VectorStore[Chunk]") -> None:
        """
        Args:
            store: Vector store satisfying the VectorStore[Chunk] Protocol.
                   Dependency injection — the observer never constructs
                   its own store, which keeps it trivially testable.
        """
        self._store = store

    # IngestionObserver interface

    def on_ingest(self, event: "IngestionEvent") -> None:
        """
        Persist every chunk in *event* to the injected vector store.

        On success, advances event.status to IngestionStatus.STORED and
        the document's status to DocumentStatus.STORED.
        On any failure, calls event.fail(...) before re-raising so the
        IngestionService and downstream observers see the failed state.

        An event with no chunks is treated as a warning and skipped —
        consistent with EmbeddingObserver's behavior on empty input.

        Args:
            event: The IngestionEvent populated by ChunkingObserver
                   and EmbeddingObserver. Each chunk must carry a
                   non-empty embedding.

        Raises:
            shared.exceptions.StorageError:
                if the store fails to persist any chunk.
        """
        chunks = event.chunks

        if not chunks:
            logger.warning(
                "StorageObserver: no chunks in event for document '%s', "
                "skipping storage.",
                event.document.filename,
            )
            return

        # Guard rail — surface a clear error if the chain ran out of order.
        # Without this, the store-layer validation would still fire, but the
        # message would be less obviously a pipeline-ordering bug.
        missing = [i for i, c in enumerate(chunks) if not c.has_embedding]
        if missing:
            message = (
                f"StorageObserver: {len(missing)} of {len(chunks)} chunk(s) "
                f"have no embedding (positions={missing[:5]}...) — "
                f"EmbeddingObserver must run before StorageObserver"
            )
            logger.error(message)
            event.fail(message)
            raise StorageError(message)

        logger.info(
            "StorageObserver: persisting %d chunk(s) for document '%s'",
            len(chunks),
            event.document.filename,
        )

        try:
            self._store.store(chunks)
        except StorageError as exc:
            logger.error(
                "StorageObserver: failed to persist chunks for '%s': %s",
                event.document.filename,
                exc,
            )
            self._cleanup_partial_write(event)
            event.fail(str(exc))
            raise
        except Exception as exc:
            # A store that is not the production MongoVectorStore (e.g. a
            # third-party backend) may raise its own native exception.
            # Wrap once so callers can always rely on RAGException.
            wrapped = StorageError(
                f"Vector store raised an unexpected error while persisting "
                f"{len(chunks)} chunk(s) for '{event.document.filename}': {exc}"
            )
            logger.error(str(wrapped))
            self._cleanup_partial_write(event)
            event.fail(str(wrapped))
            raise wrapped from exc

        event.status = IngestionStatus.STORED
        event.document.mark_status(DocumentStatus.STORED)

        logger.info(
            "StorageObserver: successfully persisted %d chunk(s) for '%s'",
            len(chunks),
            event.document.filename,
        )

    def _cleanup_partial_write(self, event: "IngestionEvent") -> None:
        """
        Best-effort removal of any chunks an aborted store() left behind.

        An unordered bulk upsert can persist some chunks before failing, which
        would leave a partial set in the index for a document that never
        reached READY. If the backing store can delete by document, drop those
        chunks. Never raises — cleanup failure must not mask the original
        StorageError, it is only logged.
        """
        delete = getattr(self._store, "delete_document", None)
        if not callable(delete):
            return
        try:
            delete(event.document.id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "StorageObserver: cleanup of partial write for '%s' failed: %s",
                event.document.filename,
                exc,
            )

    def __repr__(self) -> str:
        return f"StorageObserver(store={self._store!r})"
