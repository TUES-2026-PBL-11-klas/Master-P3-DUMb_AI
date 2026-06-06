"""
Ingestion composition root.

This is the single place where the ingestion pipeline is wired together —
the "composition root" referenced in IngestionService's docstring ("the
service never registers parsers itself, keeping the wiring in one place").

It assembles, in order:

    ParserRegistry (Strategy)         <- TxtParser + MarkdownParser
        │
    IngestionService (Subject)        <- observes:
        ├── ChunkingObserver          (sentence-aligned, ≤400 tokens)
        ├── EmbeddingObserver         (BGE-M3, in-process)
        └── StorageObserver           (-> MongoVectorStore[Chunk])

The embedding backend is PlatformEmbeddingClient (services/shared/client.py):
it loads BGE-M3 *in-process* — mlx_embeddings on macOS, the llama_cpp Python
bindings on Linux. There is no separate embedding HTTP server, so production
wiring needs only a MongoDB URI (and, optionally, an override path to the
Linux .gguf); the client self-configures by platform.

Two entry points:

  build_ingestion_service(...)
      Production wiring. Builds the real MongoVectorStore + the in-process
      PlatformEmbeddingClient and returns a ready IngestionService. Used by
      dummy_server.main().

  build_ingestion_service_with(...)
      Dependency-injection wiring. Takes an already-built vector store and an
      embedding client satisfying AIInterface. Used by integration tests
      (inject mongomock + a fake embed client) and by callers that own those
      resources.

Keeping both here means dummy_server.py only ever calls one function and
never imports the observers/parsers/store directly.
"""

from __future__ import annotations

import logging
import threading

from services.db.mongo_vector_store import MongoVectorStore
from services.ingestion.observers.chunking import ChunkingObserver
from services.ingestion.observers.embedding import EmbeddingObserver
from services.ingestion.observers.storage import StorageObserver
from services.ingestion.parsers.markdown import MarkdownParser
from services.ingestion.parsers.registry import ParserRegistry
from services.ingestion.parsers.txt import TxtParser
from services.ingestion.service import IngestionService
from services.shared.domain import Chunk, Document
from services.shared.protocols import AIInterface, DocumentStore, VectorStore

logger = logging.getLogger(__name__)


class _SerializedEmbeddingClient:
    """
    Thread-safe AIInterface wrapper that serializes calls to an in-process model.

    dummy_server handles each connection on its own thread (ThreadingMixIn), so
    two simultaneous uploads can reach the shared embedding client at the same
    time. llama.cpp / mlx inference over a single shared model context is not
    safe under concurrent calls, so a lock guards every embed call. The model
    is the throughput bottleneck anyway, so serializing costs little.

    Wrapping a *mock* (in tests) or a stateless HTTP client is harmless but
    unnecessary — build_ingestion_service_with() leaves the client untouched;
    only build_ingestion_service() opts in for the real in-process backend.
    """

    def __init__(self, inner: AIInterface) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def embed(self, text: str) -> list[float]:
        with self._lock:
            return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            return self._inner.embed_batch(texts)

    def __repr__(self) -> str:
        return f"_SerializedEmbeddingClient({self._inner!r})"


def build_parser_registry() -> ParserRegistry[Document]:
    """
    Return a registry with the project's two parsers registered.

    Adding a new format later is a one-line change here (register the new
    parser) — IngestionService is untouched (Open/Closed Principle).
    """
    registry: ParserRegistry[Document] = ParserRegistry()
    registry.register(TxtParser())
    registry.register(MarkdownParser())
    return registry


def build_ingestion_service_with(
    store: VectorStore[Chunk],
    embed_client: AIInterface,
    *,
    document_store: "DocumentStore | None" = None,
    chunk_size: int = 400,
    embed_model: str = "bge-m3",
    max_workers: int = 4,
) -> IngestionService:
    """
    Assemble an IngestionService from already-constructed dependencies.

    This is the DI seam: tests inject a mongomock-backed store and a fake
    embed client (any object satisfying AIInterface — embed / embed_batch);
    production passes the real ones built by build_ingestion_service().

    *document_store*, when supplied, is handed to the service so it can persist
    document metadata across the lifecycle (PARSED before chunks → READY after
    storage → FAILED on error). Leave it None to skip metadata persistence.

    The observer order is the canonical chunk → embed → store; the service
    runs observers in registration order, so order matters here. *embed_model*
    is a label passed to EmbeddingObserver for logging only — the actual model
    is whatever the injected client was constructed with.
    """
    service = IngestionService(
        registry=build_parser_registry(),
        observers=[
            ChunkingObserver(chunk_size=chunk_size),
            EmbeddingObserver(embed_client, model=embed_model),
            StorageObserver(store),
        ],
        document_store=document_store,
        max_workers=max_workers,
    )
    logger.info("ingestion pipeline assembled: %r", service)
    return service


def build_ingestion_service(
    mongo_uri: str,
    *,
    document_store: "DocumentStore | None" = None,
    bge_model_path: str | None = None,
    chunk_size: int = 400,
    embed_model: str = "bge-m3",
    max_workers: int = 4,
    serialize_embeddings: bool = True,
) -> IngestionService:
    """
    Production wiring: build the real store + in-process embedding client.

    Args:
        mongo_uri:      MongoDB connection string (Atlas Vector Search).
        document_store: Optional DocumentStore the service uses to persist
                        document metadata across the lifecycle. dummy_server
                        builds one MongoDocumentStore and passes the same
                        instance here and to set_document_store(), so the
                        pipeline writes and the listing reads share it.
        bge_model_path: Optional override for the Linux BGE-M3 .gguf path.
                        Ignored on macOS (which uses the mlx model). When None,
                        PlatformEmbeddingClient falls back to its default
                        (<project_root>/models/bge-m3-q8_0.gguf on Linux).
        chunk_size:     Max tokens per chunk (default 400, per reference).
        embed_model:    Logging label for the EmbeddingObserver.
        max_workers:    Thread-pool size for batch ingestion.
        serialize_embeddings:
                        Wrap the in-process client in a lock so concurrent
                        upload threads can't call the model at once. Leave on
                        unless you know the backend is concurrency-safe.

    Raises:
        StorageError: if the MongoVectorStore cannot be constructed
                      (e.g. pymongo missing or a malformed URI).
        OSError:      if PlatformEmbeddingClient can't init a backend for the
                      current platform.
    """
    store: MongoVectorStore[Chunk] = MongoVectorStore.from_uri(mongo_uri)

    # Lazy import: PlatformEmbeddingClient pulls in mlx_embeddings / llama_cpp,
    # which only the production path needs — tests using the DI seam don't.
    from services.shared.client import PlatformEmbeddingClient

    embed_client: AIInterface = PlatformEmbeddingClient(linux_model_path=bge_model_path)
    if serialize_embeddings:
        embed_client = _SerializedEmbeddingClient(embed_client)

    return build_ingestion_service_with(
        store,
        embed_client,
        document_store=document_store,
        chunk_size=chunk_size,
        embed_model=embed_model,
        max_workers=max_workers,
    )
