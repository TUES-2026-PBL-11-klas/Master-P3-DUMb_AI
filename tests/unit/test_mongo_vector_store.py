"""
Unit tests for MongoVectorStore

These tests use a MagicMock Collection — no real MongoDB is required.
The Atlas Vector Search aggregation pipeline is asserted by structure,
since we cannot actually run $vectorSearch against a mock.

Covers:
  - store() upserts on the (doc_id, position) composite key
  - store() rejects chunks without embeddings
  - store() rejects chunks with wrong-dim embeddings
  - store() of empty list is a no-op
  - store() wraps PyMongo failures as StorageError with __cause__ preserved
  - search() builds the correct $vectorSearch pipeline
  - search() applies optional user_id / doc_id filters
  - search() validates k and vector dimensionality
  - search() maps result docs back to Chunk with similarity populated
  - search() wraps aggregation failures as StorageError
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

# Module under test
from services.db.mongo_vector_store import MongoVectorStore

# Project imports
from services.shared.domain import Chunk
from services.shared.exceptions import StorageError


# Helpers

_DIM = 1024


def _make_vector(seed: float = 0.1) -> list[float]:
    return [seed] * _DIM


def _make_chunk(
    *,
    position: int = 0,
    doc_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    with_embedding: bool = True,
) -> Chunk:
    return Chunk(
        text=f"Chunk {position}",
        doc_id=doc_id or uuid.uuid4(),
        user_id=user_id or uuid.uuid4(),
        position=position,
        embedding=_make_vector(0.1 * (position + 1)) if with_embedding else [],
    )


def _make_collection() -> MagicMock:
    """A MagicMock that stands in for pymongo.collection.Collection."""
    col = MagicMock()

    # Default: bulk_write returns a result-like object with sane counts.
    bulk_result = MagicMock()
    bulk_result.upserted_count = 0
    bulk_result.modified_count = 0
    col.bulk_write = MagicMock(return_value=bulk_result)

    # Default: aggregate returns an empty iterable.
    col.aggregate = MagicMock(return_value=iter([]))
    return col


# Fixtures


@pytest.fixture
def collection() -> MagicMock:
    return _make_collection()


@pytest.fixture
def store(collection: MagicMock) -> MongoVectorStore[Chunk]:
    return MongoVectorStore(collection)


# store()


class TestStore:
    def test_empty_input_is_noop(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        store.store([])
        collection.bulk_write.assert_not_called()

    def test_upserts_each_chunk(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        doc_id = uuid.uuid4()
        chunks = [
            _make_chunk(position=i, doc_id=doc_id) for i in range(3)
        ]

        store.store(chunks)

        collection.bulk_write.assert_called_once()
        operations = collection.bulk_write.call_args.args[0]
        assert len(operations) == 3

    def test_upsert_filter_uses_composite_key(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        """Each ReplaceOne filter must be {doc_id, position} — the natural key."""
        doc_id = uuid.uuid4()
        chunks = [_make_chunk(position=i, doc_id=doc_id) for i in range(2)]

        store.store(chunks)

        operations = collection.bulk_write.call_args.args[0]
        # ReplaceOne stores the filter in ._filter on pymongo's operation
        # objects, but with a MagicMock'd ReplaceOne we instead just inspect
        # the documents passed by re-constructing them from the call args.
        # Easier: check that the per-op filter dicts include the right keys
        # by looking at the operation's ``_filter`` attribute that the real
        # pymongo.ReplaceOne exposes. We hit pymongo's real class here.
        for op, chunk in zip(operations, chunks):
            # ReplaceOne stores filter as ._filter in pymongo 4.x
            assert op._filter == {"doc_id": str(chunk.doc_id), "position": chunk.position}

    def test_upsert_payload_stringifies_uuids(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        chunk = _make_chunk(position=0)
        store.store([chunk])

        operations = collection.bulk_write.call_args.args[0]
        # pymongo.ReplaceOne stores replacement as ._doc.
        replacement = operations[0]._doc
        assert replacement["doc_id"] == str(chunk.doc_id)
        assert replacement["user_id"] == str(chunk.user_id)
        assert replacement["position"] == chunk.position
        assert replacement["text"] == chunk.text
        assert replacement["embedding"] == chunk.embedding
        # No _id supplied — MongoDB will auto-generate one on insert.
        assert "_id" not in replacement

    def test_upsert_is_unordered(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        """unordered=True lets one bad chunk not block the rest."""
        store.store([_make_chunk(position=0)])

        assert collection.bulk_write.call_args.kwargs.get("ordered") is False

    def test_rejects_chunk_without_embedding(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        chunk = _make_chunk(with_embedding=False)
        with pytest.raises(StorageError, match="no embedding"):
            store.store([chunk])

        collection.bulk_write.assert_not_called()

    def test_rejects_wrong_dim_embedding(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        chunk = _make_chunk()
        chunk.embedding = [0.1] * 512  # wrong dim

        with pytest.raises(StorageError, match="wrong embedding dim"):
            store.store([chunk])

        collection.bulk_write.assert_not_called()

    def test_pymongo_failure_wraps_as_storage_error(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        original = ConnectionError("mongo down")
        collection.bulk_write = MagicMock(side_effect=original)

        with pytest.raises(StorageError, match="Bulk upsert") as exc_info:
            store.store([_make_chunk()])

        assert exc_info.value.__cause__ is original


# search()


class TestSearch:
    def test_pipeline_structure(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        store.search(_make_vector(), k=5)

        collection.aggregate.assert_called_once()
        pipeline = collection.aggregate.call_args.args[0]
        assert len(pipeline) == 2
        assert "$vectorSearch" in pipeline[0]
        assert "$addFields" in pipeline[1]

    def test_vector_search_stage_fields(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        vec = _make_vector()
        store.search(vec, k=7)

        stage: dict[str, Any] = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert stage["index"] == "chunk_embedding_vector_index"
        assert stage["path"] == "embedding"
        assert stage["queryVector"] == vec
        assert stage["limit"] == 7
        assert stage["numCandidates"] >= 7  # multiplier-driven; just sanity-check

    def test_no_filter_when_no_scoping_args(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        store.search(_make_vector(), k=3)

        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert "filter" not in stage

    def test_user_id_filter(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        user_id = uuid.uuid4()
        store.search(_make_vector(), k=3, user_id=user_id)

        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert stage["filter"] == {"user_id": str(user_id)}

    def test_doc_id_filter(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        doc_id = uuid.uuid4()
        store.search(_make_vector(), k=3, doc_id=doc_id)

        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert stage["filter"] == {"doc_id": str(doc_id)}

    def test_combined_filters(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        user_id = uuid.uuid4()
        doc_id = uuid.uuid4()
        store.search(_make_vector(), k=3, user_id=user_id, doc_id=doc_id)

        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert stage["filter"] == {
            "user_id": str(user_id),
            "doc_id": str(doc_id),
        }

    def test_rejects_zero_k(self, store: MongoVectorStore[Chunk]) -> None:
        with pytest.raises(StorageError, match="k must be positive"):
            store.search(_make_vector(), k=0)

    def test_rejects_negative_k(self, store: MongoVectorStore[Chunk]) -> None:
        with pytest.raises(StorageError, match="k must be positive"):
            store.search(_make_vector(), k=-1)

    def test_rejects_wrong_dim_query_vector(
        self, store: MongoVectorStore[Chunk]
    ) -> None:
        with pytest.raises(StorageError, match="1024-dim"):
            store.search([0.1] * 512, k=3)

    def test_maps_results_to_chunks(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        fake_docs = [
            {
                "_id": "irrelevant",
                "text": "hello",
                "doc_id": str(doc_id),
                "user_id": str(user_id),
                "position": 0,
                "embedding": _make_vector(0.2),
                "metadata": {"source": "test"},
                "created_at": now,
                "schema_version": 1,
                "_similarity": 0.87,
            },
            {
                "_id": "irrelevant2",
                "text": "world",
                "doc_id": str(doc_id),
                "user_id": str(user_id),
                "position": 1,
                "embedding": _make_vector(0.3),
                "metadata": {},
                "created_at": now,
                "schema_version": 1,
                "_similarity": 0.62,
            },
        ]
        collection.aggregate = MagicMock(return_value=iter(fake_docs))

        results = store.search(_make_vector(), k=2)

        assert len(results) == 2
        assert all(isinstance(r, Chunk) for r in results)
        assert results[0].text == "hello"
        assert results[0].similarity == 0.87
        assert results[0].doc_id == doc_id
        assert results[1].text == "world"
        assert results[1].similarity == 0.62

    def test_missing_optional_fields_tolerated(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        """Older documents may lack metadata / created_at / schema_version."""
        doc_id = uuid.uuid4()
        user_id = uuid.uuid4()
        sparse_doc = {
            "text": "minimal",
            "doc_id": str(doc_id),
            "user_id": str(user_id),
            "position": 0,
            "embedding": _make_vector(),
        }
        collection.aggregate = MagicMock(return_value=iter([sparse_doc]))

        results = store.search(_make_vector(), k=1)

        assert len(results) == 1
        assert results[0].text == "minimal"
        assert results[0].metadata == {}
        assert results[0].similarity is None
        assert results[0].schema_version == 1

    def test_aggregate_failure_wraps_as_storage_error(
        self, store: MongoVectorStore[Chunk], collection: MagicMock
    ) -> None:
        original = TimeoutError("query timed out")
        collection.aggregate = MagicMock(side_effect=original)

        with pytest.raises(StorageError, match="Vector search") as exc_info:
            store.search(_make_vector(), k=3)

        assert exc_info.value.__cause__ is original


# Custom index name


class TestCustomIndexName:
    def test_index_name_is_respected(self, collection: MagicMock) -> None:
        store: MongoVectorStore[Chunk] = MongoVectorStore(
            collection, index_name="custom_index"
        )
        store.search(_make_vector(), k=1)

        stage = collection.aggregate.call_args.args[0][0]["$vectorSearch"]
        assert stage["index"] == "custom_index"


# Repr


class TestRepr:
    def test_repr_mentions_class(self, store: MongoVectorStore[Chunk]) -> None:
        assert "MongoVectorStore" in repr(store)