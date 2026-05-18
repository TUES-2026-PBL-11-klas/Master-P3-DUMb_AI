"""Unit tests for ChunkingObserver."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import tiktoken

from ingestion.observers.chunking import ChunkingObserver
from shared.domain import Document
from shared.exceptions import BigSentenceError

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


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

def test_default_chunk_size():
    assert ChunkingObserver().chunk_size == 400


def test_zero_chunk_size_raises():
    with pytest.raises(ValueError):
        ChunkingObserver(chunk_size=0)


def test_negative_chunk_size_raises():
    with pytest.raises(ValueError):
        ChunkingObserver(chunk_size=-1)


# ---------------------------------------------------------------------------
# _chunk_text — core algorithm
# ---------------------------------------------------------------------------

def test_empty_text_yields_nothing():
    obs = ChunkingObserver(chunk_size=50)
    assert list(obs._chunk_text("")) == []


def test_single_short_sentence_yields_one_chunk():
    obs = ChunkingObserver(chunk_size=50)
    chunks = list(obs._chunk_text("Hello world."))
    assert len(chunks) == 1
    assert "Hello" in chunks[0]


def test_multiple_short_sentences_fit_in_one_chunk():
    obs = ChunkingObserver(chunk_size=50)
    chunks = list(obs._chunk_text("Hi. Bye. Yes."))
    assert len(chunks) == 1


def test_text_splits_into_multiple_chunks():
    # chunk_size=10; "Hello world." ≈ 3 tokens, so ~3 sentences fit per chunk.
    obs = ChunkingObserver(chunk_size=10)
    text = " ".join(["Hello world."] * 10)
    chunks = list(obs._chunk_text(text))
    assert len(chunks) >= 2


def test_each_chunk_within_token_budget():
    obs = ChunkingObserver(chunk_size=15)
    text = " ".join(["Short sentence here."] * 30)
    for chunk in obs._chunk_text(text):
        assert _token_count(chunk) <= 15


def test_big_sentence_raises():
    obs = ChunkingObserver(chunk_size=5)
    long_sentence = "This is a very long sentence that definitely exceeds five tokens."
    with pytest.raises(BigSentenceError):
        list(obs._chunk_text(long_sentence))


def test_sentences_are_never_split_mid_way():
    obs = ChunkingObserver(chunk_size=20)
    sentences = [f"Sentence number {i} ends here." for i in range(10)]
    text = " ".join(sentences)
    combined = " ".join(obs._chunk_text(text))
    for s in sentences:
        assert s in combined


# ---------------------------------------------------------------------------
# on_ingest
# ---------------------------------------------------------------------------

def test_on_ingest_attaches_chunks():
    obs = ChunkingObserver(chunk_size=50)
    doc = _doc("Hello world. This is a test.")
    obs.on_ingest(doc)
    assert len(doc.chunks) >= 1  # type: ignore[attr-defined]


def test_on_ingest_chunk_positions_are_sequential():
    obs = ChunkingObserver(chunk_size=50)
    doc = _doc("Hello world. This is a test.")
    obs.on_ingest(doc)
    positions = [c.position for c in doc.chunks]  # type: ignore[attr-defined]
    assert positions == list(range(len(doc.chunks)))  # type: ignore[attr-defined]


def test_on_ingest_chunk_doc_id_matches():
    obs = ChunkingObserver(chunk_size=50)
    doc = _doc("Hello world. This is a test.")
    obs.on_ingest(doc)
    for chunk in doc.chunks:  # type: ignore[attr-defined]
        assert chunk.doc_id == doc.id


def test_on_ingest_chunk_user_id_matches():
    obs = ChunkingObserver(chunk_size=50)
    doc = _doc("Hello world. This is a test.")
    obs.on_ingest(doc)
    for chunk in doc.chunks:  # type: ignore[attr-defined]
        assert chunk.user_id == doc.user_id


def test_on_ingest_embeddings_start_empty():
    obs = ChunkingObserver(chunk_size=50)
    doc = _doc("Hello world.")
    obs.on_ingest(doc)
    for chunk in doc.chunks:  # type: ignore[attr-defined]
        assert chunk.embedding == []


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------

def test_chunk_size_property():
    assert ChunkingObserver(chunk_size=123).chunk_size == 123


def test_repr_contains_chunk_size():
    assert "99" in repr(ChunkingObserver(chunk_size=99))
