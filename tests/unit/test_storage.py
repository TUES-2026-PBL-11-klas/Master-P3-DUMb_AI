"""
Unit tests for StorageObserver

Covers:
  - Successful persistence delegates to the injected VectorStore
  - Empty chunk list is skipped gracefully
  - Missing embeddings are rejected with StorageError before the store is touched
  - Store failures propagate as StorageError and mark the event failed
  - Non-RAGException store errors are wrapped into StorageError
  - Event and document status both advance to STORED on success
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Module under test
from services.ingestion.observers.storage import StorageObserver

# Project imports
from services.shared.domain import (
    Chunk,
    Document,
    DocumentStatus,
    IngestionEvent,
    IngestionStatus,
)
from services.shared.exceptions import StorageError


# Helpers

_DIM = 1024


def _make_vector(seed: float = 0.1) -> list[float]:
    """Return a 1024-dim dummy vector."""
    return [seed] * _DIM


def _make_event(num_chunks: int = 3, *, with_embeddings: bool = True) -> IngestionEvent:
    """Create an IngestionEvent with *num_chunks* chunks."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content="full doc content",
        filename="notes.md",
        uploaded_at=datetime.now(timezone.utc),
    )
    chunks = [
        Chunk(
            text=f"Chunk number {i} with some content.",
            doc_id=doc.id,
            user_id=doc.user_id,
            position=i,
            embedding=_make_vector(0.1 * (i + 1)) if with_embeddings else [],
        )
        for i in range(num_chunks)
    ]
    # Advance status as if the previous observers had run.
    event = IngestionEvent(document=doc, chunks=chunks)
    if with_embeddings:
        event.status = IngestionStatus.EMBEDDED
    return event


# Fixtures


@pytest.fixture
def store() -> MagicMock:
    """A MagicMock that satisfies VectorStore[Chunk] structurally."""
    s = MagicMock()
    s.store = MagicMock(return_value=None)
    s.search = MagicMock(return_value=[])
    return s


@pytest.fixture
def event() -> IngestionEvent:
    return _make_event(3)


@pytest.fixture
def empty_event() -> IngestionEvent:
    return _make_event(0)


# Happy path


class TestSuccessfulStorage:
    def test_delegates_to_store(
        self, store: MagicMock, event: IngestionEvent
    ) -> None:
        observer = StorageObserver(store)
        observer.on_ingest(event)

        store.store.assert_called_once()
        passed_chunks = store.store.call_args.args[0]
        assert passed_chunks is event.chunks
        assert len(passed_chunks) == 3

    def test_advances_event_status(
        self, store: MagicMock, event: IngestionEvent
    ) -> None:
        observer = StorageObserver(store)
        observer.on_ingest(event)

        assert event.status is IngestionStatus.STORED

    def test_advances_document_status(
        self, store: MagicMock, event: IngestionEvent
    ) -> None:
        observer = StorageObserver(store)
        observer.on_ingest(event)

        assert event.document.status is DocumentStatus.STORED

    def test_single_chunk(self, store: MagicMock) -> None:
        event = _make_event(1)
        observer = StorageObserver(store)
        observer.on_ingest(event)

        store.store.assert_called_once()
        assert event.status is IngestionStatus.STORED


# Empty event


class TestEmptyEvent:
    def test_no_chunks_skips_store_call(
        self, store: MagicMock, empty_event: IngestionEvent
    ) -> None:
        observer = StorageObserver(store)
        observer.on_ingest(empty_event)

        store.store.assert_not_called()

    def test_no_chunks_does_not_advance_status(
        self, store: MagicMock, empty_event: IngestionEvent
    ) -> None:
        original_status = empty_event.status
        observer = StorageObserver(store)
        observer.on_ingest(empty_event)

        # An empty event is a warning, not a success — status stays put.
        assert empty_event.status is original_status


# Missing embeddings — pipeline-ordering bug


class TestMissingEmbeddings:
    def test_missing_embedding_raises_storage_error(
        self, store: MagicMock
    ) -> None:
        event = _make_event(3, with_embeddings=False)
        observer = StorageObserver(store)

        with pytest.raises(StorageError, match="no embedding"):
            observer.on_ingest(event)

    def test_missing_embedding_marks_event_failed(
        self, store: MagicMock
    ) -> None:
        event = _make_event(3, with_embeddings=False)
        observer = StorageObserver(store)

        with pytest.raises(StorageError):
            observer.on_ingest(event)

        assert event.status is IngestionStatus.FAILED
        assert event.error_message is not None
        assert "embedding" in event.error_message.lower()

    def test_missing_embedding_skips_store_call(
        self, store: MagicMock
    ) -> None:
        event = _make_event(3, with_embeddings=False)
        observer = StorageObserver(store)

        with pytest.raises(StorageError):
            observer.on_ingest(event)

        store.store.assert_not_called()

    def test_partial_missing_embedding_raises(
        self, store: MagicMock
    ) -> None:
        """Some chunks have embeddings, some don't — still a hard error."""
        event = _make_event(3, with_embeddings=True)
        event.chunks[1].embedding = []  # clear one embedding

        observer = StorageObserver(store)
        with pytest.raises(StorageError, match="no embedding"):
            observer.on_ingest(event)

        store.store.assert_not_called()


# Store failure handling


class TestStoreFailures:
    def test_storage_error_propagates(self, event: IngestionEvent) -> None:
        store = MagicMock()
        store.store = MagicMock(
            side_effect=StorageError("mongo write timed out")
        )

        observer = StorageObserver(store)

        with pytest.raises(StorageError, match="mongo write timed out"):
            observer.on_ingest(event)

    def test_storage_error_marks_event_failed(
        self, event: IngestionEvent
    ) -> None:
        store = MagicMock()
        store.store = MagicMock(side_effect=StorageError("disk full"))

        observer = StorageObserver(store)

        with pytest.raises(StorageError):
            observer.on_ingest(event)

        assert event.status is IngestionStatus.FAILED
        assert event.error_message == "disk full"
        assert event.document.status is DocumentStatus.FAILED

    def test_unexpected_exception_wrapped_as_storage_error(
        self, event: IngestionEvent
    ) -> None:
        """A non-RAGException error from the store is wrapped, not propagated raw."""
        original = ConnectionResetError("peer closed connection")
        store = MagicMock()
        store.store = MagicMock(side_effect=original)

        observer = StorageObserver(store)

        with pytest.raises(StorageError) as exc_info:
            observer.on_ingest(event)

        assert exc_info.value.__cause__ is original

    def test_unexpected_exception_marks_event_failed(
        self, event: IngestionEvent
    ) -> None:
        store = MagicMock()
        store.store = MagicMock(side_effect=ValueError("garbled"))

        observer = StorageObserver(store)

        with pytest.raises(StorageError):
            observer.on_ingest(event)

        assert event.status is IngestionStatus.FAILED


# Repr


class TestRepr:
    def test_repr_contains_store(self, store: MagicMock) -> None:
        observer = StorageObserver(store)
        assert "StorageObserver" in repr(observer)