"""
MongoVectorStore — VectorStore[Chunk] implementation backed by MongoDB
Atlas Vector Search.

Responsibilities:
  - Persist Chunk objects (text + 1024-dim embedding + metadata) to the
    ``document_chunks`` collection.
  - Retrieve the top-k nearest chunks to a query vector via the
    ``chunk_embedding_vector_index`` Atlas Vector Search index using
    cosine similarity.

Design pattern: Strategy / Repository
    Implements shared.protocols.VectorStore[Chunk] through structural
    subtyping (no explicit inheritance required). Code that needs a
    vector store depends on the Protocol — never on this concrete class
    — so an in-memory stub can be substituted in unit tests.

Composite key:
    Chunks are upserted on the natural composite key (doc_id, position),
    which mirrors the unique index declared in infra/mongo/init_db.js.
    Re-ingesting the same document overwrites its chunks instead of
    creating duplicates.

Atlas Vector Search:
    The aggregation pipeline uses the ``$vectorSearch`` stage. The index
    name and filter fields are configured at deploy time by
    init_db.js — change those constants here if the index is renamed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Generic, TypeVar
from uuid import UUID

from services.shared.exceptions import StorageError

if TYPE_CHECKING:
    from pymongo import MongoClient
    from pymongo.collection import Collection

    from services.shared.domain import Chunk

logger = logging.getLogger(__name__)

# Defaults — matched to infra/mongo/init_db.js.
# If init_db.js is changed, update these in lockstep.
_DEFAULT_DB_NAME = "dumb_ai"
_DEFAULT_COLLECTION = "document_chunks"
_DEFAULT_VECTOR_INDEX = "chunk_embedding_vector_index"

# BGE-M3 produces 1024-dimensional vectors.
_EXPECTED_DIM = 1024

# Atlas Vector Search recommends numCandidates ≈ 10×–20× the requested k.
# We use 10× as a conservative middle-ground that still keeps recall high.
_NUM_CANDIDATES_MULTIPLIER = 10

T_inv = TypeVar("T_inv")


class MongoVectorStore(Generic[T_inv]):
    """
    MongoDB-Atlas-backed vector store for Chunk objects.

    Satisfies shared.protocols.VectorStore[Chunk] through structural
    subtyping. The class is declared generic over T_inv only to match
    the Protocol signature — the on-disk serialization is hard-wired to
    the Chunk dataclass, since BSON cannot serialise an unknown T.

    Construction:
        Prefer dependency injection in tests::

            store = MongoVectorStore(collection=some_collection)

        In production, use the convenience factory::

            store = MongoVectorStore.from_uri(
                "mongodb://localhost:27018",
            )

    Attributes:
        _col:        pymongo Collection handle for the chunks collection.
        _index_name: Name of the Atlas Vector Search index to query.
        _owns_client: True if the store opened its own MongoClient
                      (via from_uri) and is responsible for closing it.
    """

    def __init__(
        self,
        collection: "Collection[dict[str, Any]]",
        *,
        index_name: str = _DEFAULT_VECTOR_INDEX,
        _client: "MongoClient[dict[str, Any]] | None" = None,
    ) -> None:
        """
        Args:
            collection: pymongo Collection bound to the chunks collection.
                        Injecting the collection (rather than a URI) keeps
                        the class trivially testable with mongomock or a
                        MagicMock.
            index_name: Name of the Atlas Vector Search index used by
                        search(). Defaults to ``chunk_embedding_vector_index``
                        as declared in infra/mongo/init_db.js.
            _client:    Internal — set by from_uri() when the store opens
                        its own MongoClient. Not part of the public API.
        """
        self._col = collection
        self._index_name = index_name
        self._client = _client
        self._owns_client = _client is not None

    # Construction helpers

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        db_name: str = _DEFAULT_DB_NAME,
        collection_name: str = _DEFAULT_COLLECTION,
        index_name: str = _DEFAULT_VECTOR_INDEX,
    ) -> "MongoVectorStore[T_inv]":
        """
        Open a MongoClient against *uri* and return a store bound to it.

        The returned store owns the client and will close it when
        ``close()`` is called.

        Raises:
            StorageError: if pymongo cannot be imported or the connection
                          parameters are malformed.
        """
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover — import-time only
            raise StorageError(
                "pymongo is not installed — `pip install pymongo` is required "
                "to use MongoVectorStore.from_uri()"
            ) from exc

        try:
            client: MongoClient[dict[str, Any]] = MongoClient(uri)
            collection = client[db_name][collection_name]
        except Exception as exc:
            raise StorageError(
                f"Failed to construct MongoClient for {uri!r}: {exc}"
            ) from exc

        logger.info(
            "MongoVectorStore: connected to %s.%s (index=%s)",
            db_name,
            collection_name,
            index_name,
        )
        return cls(collection=collection, index_name=index_name, _client=client)

    def close(self) -> None:
        """Close the underlying MongoClient, if this store owns one."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
            self._owns_client = False

    # VectorStore[Chunk] interface

    def store(self, chunks: "list[Chunk]") -> None:
        """
        Upsert *chunks* into the collection on the (doc_id, position)
        composite key.

        Empty input is a no-op. Each chunk must have a non-empty embedding
        of the expected dimensionality; chunks without embeddings indicate
        a pipeline bug (EmbeddingObserver should have run first) and are
        rejected with StorageError to keep junk out of the index.

        Args:
            chunks: List of Chunk objects to persist. ``chunk.embedding``
                    must be a 1024-dim ``list[float]``.

        Raises:
            StorageError: if any chunk is missing an embedding, has a
                          wrong-sized embedding, or if the PyMongo write
                          itself fails.
        """
        if not chunks:
            logger.debug("MongoVectorStore.store: empty input — no-op")
            return

        # Validate up-front so we never hit Mongo with a partial batch.
        for i, chunk in enumerate(chunks):
            self._validate_chunk_for_storage(chunk, chunk_index=i)

        # PyMongo's bulk_write is in ReplaceOne+upsert mode — this gives
        # us the (doc_id, position) idempotency we want without any
        # round-trip-per-chunk overhead.
        try:
            from pymongo import ReplaceOne
        except ImportError as exc:  # pragma: no cover — import-time only
            raise StorageError(
                "pymongo is not installed — install it or inject a mock collection"
            ) from exc

        operations = [
            ReplaceOne(
                filter={"doc_id": str(chunk.doc_id), "position": chunk.position},
                replacement=self._chunk_to_doc(chunk),
                upsert=True,
            )
            for chunk in chunks
        ]

        try:
            result = self._col.bulk_write(operations, ordered=False)
        except Exception as exc:
            raise StorageError(
                f"Bulk upsert of {len(chunks)} chunk(s) failed: {exc}"
            ) from exc

        # bulk_write counters may be None on some drivers; coalesce for the log.
        inserted = getattr(result, "upserted_count", 0) or 0
        modified = getattr(result, "modified_count", 0) or 0
        logger.info(
            "MongoVectorStore.store: upserted=%d modified=%d (input=%d)",
            inserted,
            modified,
            len(chunks),
        )

    def search(
        self,
        vec: list[float],
        k: int,
        *,
        user_id: UUID | str | None = None,
        doc_id: UUID | str | None = None,
    ) -> "list[Chunk]":
        """
        Return the *k* nearest chunks to *vec* (cosine similarity),
        ordered by descending similarity.

        Optional filters (user_id, doc_id) are pushed into the
        ``$vectorSearch`` stage. Both fields are declared as filter
        paths in the Atlas index — see init_db.js.

        Args:
            vec:     Query embedding. Must be 1024-dim to match the index.
            k:       Number of results to return. Must be positive.
            user_id: Optional — restrict results to a single user.
            doc_id:  Optional — restrict results to a single document.

        Returns:
            List of Chunk objects, length <= k, ordered by descending
            similarity. ``chunk.similarity`` is populated from
            ``$vectorSearchScore``.

        Raises:
            StorageError: if the vector dimensionality is wrong, k is
                          not positive, or the aggregation pipeline fails.
        """
        if k <= 0:
            raise StorageError(f"k must be positive, got {k}")
        if len(vec) != _EXPECTED_DIM:
            raise StorageError(
                f"Query vector must be {_EXPECTED_DIM}-dim, got {len(vec)}-dim"
            )

        vector_search_stage: dict[str, Any] = {
            "index": self._index_name,
            "path": "embedding",
            "queryVector": vec,
            "numCandidates": k * _NUM_CANDIDATES_MULTIPLIER,
            "limit": k,
        }

        # Build the optional filter — Atlas only accepts the `filter` key
        # when at least one clause is present.
        filter_clauses: dict[str, Any] = {}
        if user_id is not None:
            filter_clauses["user_id"] = str(user_id)
        if doc_id is not None:
            filter_clauses["doc_id"] = str(doc_id)
        if filter_clauses:
            vector_search_stage["filter"] = filter_clauses

        pipeline: list[dict[str, Any]] = [
            {"$vectorSearch": vector_search_stage},
            {
                "$addFields": {
                    "_similarity": {"$meta": "vectorSearchScore"},
                },
            },
        ]

        try:
            cursor = self._col.aggregate(pipeline)
            raw_docs = list(cursor)
        except Exception as exc:
            raise StorageError(
                f"Vector search aggregation failed (index={self._index_name!r}, "
                f"k={k}): {exc}"
            ) from exc

        return [self._doc_to_chunk(doc) for doc in raw_docs]

    # Serialization helpers

    @staticmethod
    def _chunk_to_doc(chunk: "Chunk") -> dict[str, Any]:
        """
        Convert a Chunk dataclass into a BSON-ready dict.

        UUIDs are stringified — this matches how init_db.js declares
        ``doc_id`` and ``user_id`` as ``string`` filter fields in the
        vector index. Storing them as native BSON UUIDs would silently
        break the filter.

        MongoDB will auto-generate an ``_id`` (an ObjectId) on insert
        when one isn't supplied; the natural composite key
        (doc_id, position) — enforced by a unique index — is what gives
        chunks their durable identity.
        """
        return {
            "user_id": str(chunk.user_id),
            "doc_id": str(chunk.doc_id),
            "position": chunk.position,
            "text": chunk.text,
            "embedding": list(chunk.embedding),
            "token_count": chunk.token_count,
            "metadata": dict(chunk.metadata),
            "created_at": chunk.created_at or datetime.now(timezone.utc),
            "schema_version": chunk.schema_version,
        }

    @staticmethod
    def _doc_to_chunk(doc: dict[str, Any]) -> "Chunk":
        """
        Convert a BSON dict back into a Chunk dataclass.

        Tolerates missing optional fields (older documents written before
        a schema bump may lack ``metadata`` or ``created_at``).
        """
        # Lazy import — keeps the module importable without the rest of
        # the project on the path (helpful for documentation tooling).
        from services.shared.domain import Chunk

        similarity = doc.get("_similarity")
        return Chunk(
            text=doc["text"],
            doc_id=UUID(doc["doc_id"]),
            user_id=UUID(doc["user_id"]),
            position=int(doc["position"]),
            embedding=list(doc.get("embedding", [])),
            token_count=int(doc.get("token_count", 0)),
            similarity=float(similarity) if similarity is not None else None,
            metadata=dict(doc.get("metadata", {})),
            created_at=doc.get("created_at"),
            schema_version=int(doc.get("schema_version", 1)),
        )

    @staticmethod
    def _validate_chunk_for_storage(chunk: "Chunk", *, chunk_index: int) -> None:
        """Reject chunks that are not ready for persistence."""
        if not chunk.embedding:
            raise StorageError(
                f"Chunk {chunk_index} (doc_id={chunk.doc_id}, "
                f"position={chunk.position}) has no embedding — "
                f"EmbeddingObserver must run before StorageObserver"
            )
        if len(chunk.embedding) != _EXPECTED_DIM:
            raise StorageError(
                f"Chunk {chunk_index} has wrong embedding dim: "
                f"expected {_EXPECTED_DIM}, got {len(chunk.embedding)}"
            )

    def __repr__(self) -> str:
        return f"MongoVectorStore(collection={self._col!r}, index={self._index_name!r})"
