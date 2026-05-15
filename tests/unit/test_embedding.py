"""
Unit tests for EmbeddingObserver

Covers:
  - Successful embedding via sequential (embed) path
  - Successful embedding via batch (embed_batch) path
  - Empty chunk list (skip gracefully)
  - Dimension validation (wrong vector size)
  - Client failure handling (raises EmbeddingError)
  - Batch size splitting
  - Batch count mismatch detection
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

# Module under test
from services.ingestion.observers.embedding import EmbeddingObserver

# Project imports
from services.shared.domain import Chunk, Document, IngestionEvent
from services.shared.exceptions import EmbeddingError


# Helpers

_DIM = 1024


def _make_vector(seed: float = 0.1) -> list[float]:
    """Return a 1024-dim dummy vector."""
    return [seed] * _DIM


def _make_event(num_chunks: int = 3) -> IngestionEvent:
    """Create an IngestionEvent with *num_chunks* chunks containing text."""
    doc = Document(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content="full doc content",
        filename="notes.md",
        uploaded_at=datetime.now(timezone.utc),
    )
    chunks = [
        Chunk(
            id=uuid.uuid4(),
            text=f"Chunk number {i} with some content.",
            doc_id=doc.id,
            user_id=doc.user_id,
            position=i,
        )
        for i in range(num_chunks)
    ]
    return IngestionEvent(document=doc, chunks=chunks)


def _make_sequential_client() -> MagicMock:
    """Client with embed() only (no embed_batch)."""
    client = MagicMock()
    client.embed = MagicMock(return_value=_make_vector())
    # Remove embed_batch so the observer falls back to sequential
    del client.embed_batch
    return client


def _make_batch_client() -> MagicMock:
    """Client with both embed() and embed_batch()."""
    client = MagicMock()
    client.embed = MagicMock(return_value=_make_vector())
    client.embed_batch = MagicMock(
        side_effect=lambda texts: [_make_vector(0.1 * i) for i, _ in enumerate(texts)]
    )
    return client


# Fixtures


@pytest.fixture
def seq_client() -> MagicMock:
    return _make_sequential_client()


@pytest.fixture
def batch_client() -> MagicMock:
    return _make_batch_client()


@pytest.fixture
def event() -> IngestionEvent:
    return _make_event(3)


@pytest.fixture
def empty_event() -> IngestionEvent:
    return _make_event(0)


# Sequential embedding


class TestSequentialEmbedding:
    def test_embeds_all_chunks(
        self, seq_client: MagicMock, event: IngestionEvent
    ) -> None:
        observer = EmbeddingObserver(seq_client)
        observer.on_ingest(event)

        assert seq_client.embed.call_count == 3
        for chunk in event.chunks:
            assert len(chunk.embedding) == _DIM

    def test_calls_embed_with_chunk_text(
        self, seq_client: MagicMock, event: IngestionEvent
    ) -> None:
        observer = EmbeddingObserver(seq_client)
        observer.on_ingest(event)

        call_texts = [call.args[0] for call in seq_client.embed.call_args_list]
        expected = [c.text for c in event.chunks]
        assert call_texts == expected

    def test_single_chunk(self, seq_client: MagicMock) -> None:
        event = _make_event(1)
        observer = EmbeddingObserver(seq_client)
        observer.on_ingest(event)

        assert len(event.chunks[0].embedding) == _DIM
        assert seq_client.embed.call_count == 1


# Batch embedding


class TestBatchEmbedding:
    def test_uses_embed_batch_when_available(
        self, batch_client: MagicMock, event: IngestionEvent
    ) -> None:
        observer = EmbeddingObserver(batch_client, batch_size=16)
        observer.on_ingest(event)

        batch_client.embed_batch.assert_called_once()
        # embed() should NOT have been called
        batch_client.embed.assert_not_called()

        for chunk in event.chunks:
            assert len(chunk.embedding) == _DIM

    def test_batch_splitting(self, batch_client: MagicMock) -> None:
        """With batch_size=2 and 5 chunks, expect 3 batch calls."""
        event = _make_event(5)
        observer = EmbeddingObserver(batch_client, batch_size=2)
        observer.on_ingest(event)

        assert batch_client.embed_batch.call_count == 3
        for chunk in event.chunks:
            assert len(chunk.embedding) == _DIM

    def test_batch_size_exact_multiple(self, batch_client: MagicMock) -> None:
        """4 chunks with batch_size=2 → exactly 2 calls."""
        event = _make_event(4)
        observer = EmbeddingObserver(batch_client, batch_size=2)
        observer.on_ingest(event)

        assert batch_client.embed_batch.call_count == 2


# Empty event


class TestEmptyEvent:
    def test_no_chunks_skips_gracefully(
        self, seq_client: MagicMock, empty_event: IngestionEvent
    ) -> None:
        observer = EmbeddingObserver(seq_client)
        observer.on_ingest(empty_event)

        seq_client.embed.assert_not_called()

    def test_no_chunks_batch_client(
        self, batch_client: MagicMock, empty_event: IngestionEvent
    ) -> None:
        observer = EmbeddingObserver(batch_client)
        observer.on_ingest(empty_event)

        batch_client.embed_batch.assert_not_called()


# Dimension validation


class TestDimensionValidation:
    def test_wrong_dimension_raises(self, event: IngestionEvent) -> None:
        client = MagicMock()
        client.embed = MagicMock(return_value=[0.1] * 512)  # wrong dim
        del client.embed_batch

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError, match="Expected 1024-dim"):
            observer.on_ingest(event)

    def test_wrong_dimension_batch_raises(self, event: IngestionEvent) -> None:
        client = MagicMock()
        client.embed_batch = MagicMock(
            return_value=[[0.1] * 512 for _ in range(3)]  # wrong dim
        )

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError, match="Expected 1024-dim"):
            observer.on_ingest(event)

    def test_empty_vector_raises(self, event: IngestionEvent) -> None:
        client = MagicMock()
        client.embed = MagicMock(return_value=[])
        del client.embed_batch

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError, match="Expected 1024-dim.*got 0-dim"):
            observer.on_ingest(event)


# Client errors


class TestClientErrors:
    def test_embed_failure_raises_embedding_error(self, event: IngestionEvent) -> None:
        client = MagicMock()
        client.embed = MagicMock(side_effect=ConnectionError("server down"))
        del client.embed_batch

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError, match="Embedding failed"):
            observer.on_ingest(event)

    def test_embed_failure_preserves_cause(self, event: IngestionEvent) -> None:
        original = ConnectionError("server down")
        client = MagicMock()
        client.embed = MagicMock(side_effect=original)
        del client.embed_batch

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError) as exc_info:
            observer.on_ingest(event)

        assert exc_info.value.__cause__ is original

    def test_batch_failure_raises_embedding_error(self, event: IngestionEvent) -> None:
        client = MagicMock()
        client.embed_batch = MagicMock(side_effect=TimeoutError("timeout"))

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError, match="Batch embedding failed"):
            observer.on_ingest(event)

    def test_batch_count_mismatch_raises(self, event: IngestionEvent) -> None:
        """embed_batch returns fewer vectors than inputs."""
        client = MagicMock()
        client.embed_batch = MagicMock(
            return_value=[_make_vector()]  # only 1 vector for 3 chunks
        )

        observer = EmbeddingObserver(client)

        with pytest.raises(EmbeddingError, match="returned 1 vectors for 3 inputs"):
            observer.on_ingest(event)


# Repr


class TestRepr:
    def test_repr_contains_model(self, seq_client: MagicMock) -> None:
        observer = EmbeddingObserver(seq_client, model="testing-model")
        assert "testing-model" in repr(observer)

    def test_repr_contains_batch_size(self, seq_client: MagicMock) -> None:
        observer = EmbeddingObserver(seq_client, batch_size=32)
        assert "32" in repr(observer)
