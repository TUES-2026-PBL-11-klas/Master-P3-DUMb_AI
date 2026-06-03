"""
Ingestion composition root. This is the single place where the ingestion pipeline is wired together.

It assembles, in order:

    ParserRegistry (Strategy)         TxtParser + MarkdownParser
        │
    IngestionService (Subject)        observes:
        ├── ChunkingObserver          (sentence-aligned, ≤400 tokens)
        ├── EmbeddingObserver         (BGE-M3 via llama.cpp)
        └── StorageObserver           (MongoVectorStore[Chunk])

Two entry points:

  build_ingestion_service(...)
      Production wiring. Reads a MongoDB URI and a llama.cpp base URL,
      constructs the real MongoVectorStore and LlamaCppEmbeddingClient,
      and returns a ready IngestionService. Used by dummy_server.main().

  build_ingestion_service_with(...)
      Dependency-injection wiring. Takes an already-built vector store and
      embedding client. Used by integration tests (inject mongomock + a
      fake embed client) and by callers that own those resources.

Keeping both here means dummy_server.py only ever calls one function and
never imports the observers/parsers/store directly.
"""

from __future__ import annotations

import logging

from services.db.mongo_vector_store import MongoVectorStore
from services.ingestion.observers.chunking import ChunkingObserver
from services.ingestion.observers.embedding import EmbeddingObserver
from services.ingestion.observers.storage import StorageObserver
from services.ingestion.parsers.markdown import MarkdownParser
from services.ingestion.parsers.registry import ParserRegistry
from services.ingestion.parsers.txt import TxtParser
from services.ingestion.service import IngestionService
from services.shared.protocols import LlamaCppClient
from services.shared.domain import Chunk, Document
from services.shared.protocols import VectorStore

logger = logging.getLogger(__name__)


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
    embed_client: LlamaCppClient,
    *,
    chunk_size: int = 400,
    embed_model: str = "bge-m3",
    max_workers: int = 4,
) -> IngestionService:
    """
    Assemble an IngestionService from already-constructed dependencies.

    This is the DI seam: tests inject a mongomock-backed store and a fake
    embed client; production passes the real ones built by
    build_ingestion_service().

    The observer order is the canonical chunk → embed → store; the service
    runs observers in registration order, so order matters here.
    """
    service = IngestionService(
        registry=build_parser_registry(),
        observers=[
            ChunkingObserver(chunk_size=chunk_size),
            EmbeddingObserver(embed_client, model=embed_model),
            StorageObserver(store),
        ],
        max_workers=max_workers,
    )
    logger.info("ingestion pipeline assembled: %r", service)
    return service


def build_ingestion_service(
    mongo_uri: str,
    llama_base_url: str,
    *,
    chunk_size: int = 400,
    embed_model: str = "bge-m3",
    max_workers: int = 4,
) -> IngestionService:
    """
    Production wiring: build the real store + embedding client, then assemble.

    Args:
        mongo_uri:      MongoDB connection string (Atlas Vector Search).
        llama_base_url: Root URL of the llama.cpp server hosting BGE-M3,
                        e.g. "http://llama-cpp:8080".
        chunk_size:     Max tokens per chunk (default 400, per reference).
        embed_model:    Embedding model name passed to llama.cpp.
        max_workers:    Thread-pool size for batch ingestion.

    Raises:
        StorageError: if the MongoVectorStore cannot be constructed
                      (e.g. pymongo missing or a malformed URI).
    """
    store: MongoVectorStore[Chunk] = MongoVectorStore.from_uri(mongo_uri)
    # Lazy import: the concrete client lives in a teammate-owned module.
    from services.shared.client import PlatformEmbeddingClient

    embed_client = PlatformEmbeddingClient(llama_base_url, model=embed_model)
    return build_ingestion_service_with(
        store,
        embed_client,
        chunk_size=chunk_size,
        embed_model=embed_model,
        max_workers=max_workers,
    )
