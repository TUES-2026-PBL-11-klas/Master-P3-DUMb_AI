"""
DocumentStorageObserver - persists uploaded document metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from services.shared.exceptions import StorageError

if TYPE_CHECKING:
    from services.shared.domain import Document, IngestionEvent

logger = logging.getLogger(__name__)


class DocumentStore(Protocol):
    def upsert(self, document: "Document") -> None: ...


class DocumentStorageObserver:
    """Observer that writes Document metadata to a document repository."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def on_ingest(self, event: "IngestionEvent") -> None:
        try:
            self._store.upsert(event.document)
        except StorageError:
            event.fail(f"Failed to persist document metadata for {event.document.id}")
            raise
        except Exception as exc:
            message = (
                "Document store raised an unexpected error while persisting "
                f"metadata for '{event.document.filename}': {exc}"
            )
            event.fail(message)
            raise StorageError(message) from exc

        logger.info(
            "DocumentStorageObserver: persisted metadata for '%s'",
            event.document.filename,
        )

    def __repr__(self) -> str:
        return f"DocumentStorageObserver(store={self._store!r})"
