"""Unit tests for ChunkingObserver."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import tiktoken

from services.ingestion.observers.chunking import ChunkingObserver
from services.shared.domain import Document, IngestionEvent, IngestionStatus
from services.shared.exceptions import BigSentenceError

_ENC = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


def _doc(content: str) -> Document:
    return Document(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        content=content,
        filename="test.txt",
        uploaded_at=datetime.now(timezone.utc),
    )


def _event(content: str) -> IngestionEvent:
    return IngestionEvent(document=_doc(content))


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_default_chunk_size() -> None:
    assert ChunkingObserver().chunk_size == 400


def test_zero_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        ChunkingObserver(chunk_size=0)


def test_negative_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        ChunkingObserver(chunk_size=-1)


# ---------------------------------------------------------------------------
# _chunk_text — core algorithm
# ---------------------------------------------------------------------------


def test_empty_text_yields_nothing() -> None:
    obs = ChunkingObserver(chunk_size=50)
    assert list(obs._chunk_text("")) == []


def test_single_short_sentence_yields_one_chunk() -> None:
    obs = ChunkingObserver(chunk_size=50)
    chunks = list(obs._chunk_text("Hello world."))
    assert len(chunks) == 1
    assert "Hello" in chunks[0]


def test_multiple_short_sentences_fit_in_one_chunk() -> None:
    obs = ChunkingObserver(chunk_size=50)
    chunks = list(obs._chunk_text("Hi. Bye. Yes."))
    assert len(chunks) == 1


def test_text_splits_into_multiple_chunks() -> None:
    # chunk_size=10; "Hello world." ≈ 3 tokens, so ~3 sentences fit per chunk.
    obs = ChunkingObserver(chunk_size=10)
    text = " ".join(["Hello world."] * 10)
    chunks = list(obs._chunk_text(text))
    assert len(chunks) >= 2


def test_each_chunk_within_token_budget() -> None:
    obs = ChunkingObserver(chunk_size=15)
    text = " ".join(["Short sentence here."] * 30)
    for chunk in obs._chunk_text(text):
        assert _token_count(chunk) <= 15


def test_big_sentence_raises() -> None:
    obs = ChunkingObserver(chunk_size=5)
    long_sentence = "This is a very long sentence that definitely exceeds five tokens."
    with pytest.raises(BigSentenceError):
        list(obs._chunk_text(long_sentence))


def test_sentences_are_never_split_mid_way() -> None:
    obs = ChunkingObserver(chunk_size=20)
    sentences = [f"Sentence number {i} ends here." for i in range(10)]
    text = " ".join(sentences)
    combined = " ".join(obs._chunk_text(text))
    for s in sentences:
        assert s in combined


def test_whitespace_preserved_between_packed_sentences() -> None:
    # Two short sentences that fit in one chunk — the decoded text must have a
    # space between them, not be fused as "Hello world.Goodbye."
    obs = ChunkingObserver(chunk_size=50)
    chunks = list(obs._chunk_text("Hello world. Goodbye."))
    assert len(chunks) == 1
    assert "Hello world." in chunks[0]
    assert "Goodbye." in chunks[0]
    assert "world.Goodbye" not in chunks[0]


def test_chunks_carry_token_count() -> None:
    obs = ChunkingObserver(chunk_size=400)
    chunks = list(obs._chunk_document(_doc("Short sentence one. Short sentence two.")))

    assert all(c.token_count > 0 for c in chunks)
    assert all(c.token_count <= 400 for c in chunks)


# ---------------------------------------------------------------------------
# on_ingest
# ---------------------------------------------------------------------------


def test_on_ingest_attaches_chunks() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("The sky is blue. Water is wet. Fire is hot.")
    obs.on_ingest(event)
    assert len(event.chunks) >= 1


def test_on_ingest_advances_status_to_chunked() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("Hello world. This is a test.")
    obs.on_ingest(event)
    assert event.status is IngestionStatus.CHUNKED


def test_on_ingest_chunk_positions_are_sequential() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("First sentence. Second sentence. Third sentence.")
    obs.on_ingest(event)
    positions = [c.position for c in event.chunks]
    assert positions == list(range(len(event.chunks)))


def test_on_ingest_chunk_doc_id_matches() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("Alpha. Beta. Gamma. Delta.")
    obs.on_ingest(event)
    for chunk in event.chunks:
        assert chunk.doc_id == event.document.id


def test_on_ingest_chunk_user_id_matches() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("One fish. Two fish. Red fish. Blue fish.")
    obs.on_ingest(event)
    for chunk in event.chunks:
        assert chunk.user_id == event.document.user_id


def test_on_ingest_embeddings_start_empty() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("Hello world.")
    obs.on_ingest(event)
    for chunk in event.chunks:
        assert chunk.embedding == []


def test_on_ingest_empty_content_produces_no_chunks() -> None:
    obs = ChunkingObserver(chunk_size=50)
    event = _event("")
    obs.on_ingest(event)
    assert event.chunks == []


def test_on_ingest_big_sentence_marks_event_failed() -> None:
    obs = ChunkingObserver(chunk_size=5)
    event = _event("This is a very long sentence that definitely exceeds five tokens.")
    with pytest.raises(BigSentenceError):
        obs.on_ingest(event)
    assert event.status is IngestionStatus.FAILED
    assert event.error_message is not None


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


def test_chunk_size_property() -> None:
    assert ChunkingObserver(chunk_size=123).chunk_size == 123


def test_repr_contains_chunk_size() -> None:
    assert "99" in repr(ChunkingObserver(chunk_size=99))
