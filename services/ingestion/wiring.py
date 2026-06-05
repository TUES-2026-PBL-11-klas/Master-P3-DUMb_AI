"""
Runtime wiring for the document ingestion pipeline.

Builds the production observer chain:
ParserRegistry -> ChunkingObserver -> EmbeddingObserver -> StorageObserver.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from services.db.mongo_document_store import MongoDocumentStore
from services.db.mongo_vector_store import MongoVectorStore
from services.ingestion.observers.chunking import ChunkingObserver
from services.ingestion.observers.document_storage import DocumentStorageObserver
from services.ingestion.observers.embedding import EmbeddingObserver
from services.ingestion.observers.storage import StorageObserver
from services.ingestion.parsers.markdown import MarkdownParser
from services.ingestion.parsers.registry import ParserRegistry
from services.ingestion.parsers.txt import TxtParser
from services.ingestion.service import IngestionService
from services.query.wiring import (
    DEFAULT_CHUNK_COLLECTION,
    DEFAULT_DB_NAME,
    DEFAULT_VECTOR_INDEX,
)
from services.shared.client import (
    DEFAULT_EMBEDDING_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_LINUX_EMBEDDING_MODEL_PATH,
    DEFAULT_MAC_EMBEDDING_MODEL,
    LlamaCppClient,
    PlatformEmbeddingClient,
)
from services.shared.domain import Chunk, Document
from services.shared.exceptions import QueryError
from services.shared.protocols import VectorStore

DEFAULT_CHUNK_SIZE = 400
DEFAULT_EMBED_BATCH_SIZE = 16
DEFAULT_INGEST_MAX_WORKERS = 4


class VectorStoreFactory(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        db_name: str,
        collection_name: str,
        index_name: str,
    ) -> VectorStore[Chunk]: ...


class DocumentStoreFactory(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        db_name: str,
        collection_name: str,
    ) -> object: ...


class EmbeddingClientFactory(Protocol):
    def __call__(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> LlamaCppClient: ...


@dataclass(frozen=True)
class IngestionRuntimeConfig:
    mongo_uri: str
    mongo_db_name: str = DEFAULT_DB_NAME
    document_collection: str = "documents"
    chunk_collection: str = DEFAULT_CHUNK_COLLECTION
    vector_index: str = DEFAULT_VECTOR_INDEX
    mac_embedding_model: str = DEFAULT_MAC_EMBEDDING_MODEL
    linux_embedding_model_path: str = DEFAULT_LINUX_EMBEDDING_MODEL_PATH
    embedding_context: int = DEFAULT_EMBEDDING_CONTEXT
    gpu_layers: int = DEFAULT_GPU_LAYERS
    chunk_size: int = DEFAULT_CHUNK_SIZE
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
    max_workers: int = DEFAULT_INGEST_MAX_WORKERS

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "IngestionRuntimeConfig | None":
        env = environ or os.environ
        mongo_uri = _first_non_empty(env, "RAG_MONGODB_URI", "MONGODB_URI")
        if mongo_uri is None:
            return None

        return cls(
            mongo_uri=mongo_uri,
            mongo_db_name=env.get("RAG_MONGODB_DB", DEFAULT_DB_NAME),
            document_collection=env.get("RAG_DOCUMENT_COLLECTION", "documents"),
            chunk_collection=env.get("RAG_CHUNK_COLLECTION", DEFAULT_CHUNK_COLLECTION),
            vector_index=env.get("RAG_VECTOR_INDEX", DEFAULT_VECTOR_INDEX),
            mac_embedding_model=env.get(
                "RAG_MAC_EMBED_MODEL",
                DEFAULT_MAC_EMBEDDING_MODEL,
            ),
            linux_embedding_model_path=env.get(
                "RAG_LINUX_EMBED_MODEL_PATH",
                DEFAULT_LINUX_EMBEDDING_MODEL_PATH,
            ),
            embedding_context=_parse_positive_int(
                env.get("RAG_EMBED_N_CTX"),
                default=DEFAULT_EMBEDDING_CONTEXT,
                name="RAG_EMBED_N_CTX",
            ),
            gpu_layers=_parse_int(env.get("RAG_GPU_LAYERS"), default=DEFAULT_GPU_LAYERS),
            chunk_size=_parse_positive_int(
                env.get("RAG_CHUNK_SIZE"),
                default=DEFAULT_CHUNK_SIZE,
                name="RAG_CHUNK_SIZE",
            ),
            embed_batch_size=_parse_positive_int(
                env.get("RAG_EMBED_BATCH_SIZE"),
                default=DEFAULT_EMBED_BATCH_SIZE,
                name="RAG_EMBED_BATCH_SIZE",
            ),
            max_workers=_parse_positive_int(
                env.get("RAG_INGEST_MAX_WORKERS"),
                default=DEFAULT_INGEST_MAX_WORKERS,
                name="RAG_INGEST_MAX_WORKERS",
            ),
        )


def build_ingestion_service(
    config: IngestionRuntimeConfig,
    *,
    vector_store_factory: VectorStoreFactory = MongoVectorStore.from_uri,
    document_store_factory: DocumentStoreFactory = MongoDocumentStore.from_uri,
    embedding_client_factory: EmbeddingClientFactory = PlatformEmbeddingClient,
) -> IngestionService:
    registry: ParserRegistry[Document] = ParserRegistry()
    registry.register(TxtParser())
    registry.register(MarkdownParser())

    embedding_client = embedding_client_factory(
        mac_model=config.mac_embedding_model,
        linux_model_path=config.linux_embedding_model_path,
        n_ctx=config.embedding_context,
        n_gpu_layers=config.gpu_layers,
    )
    vector_store = vector_store_factory(
        config.mongo_uri,
        db_name=config.mongo_db_name,
        collection_name=config.chunk_collection,
        index_name=config.vector_index,
    )
    document_store = document_store_factory(
        config.mongo_uri,
        db_name=config.mongo_db_name,
        collection_name=config.document_collection,
    )

    return IngestionService(
        registry=registry,
        observers=[
            ChunkingObserver(chunk_size=config.chunk_size),
            EmbeddingObserver(
                client=embedding_client,
                model=config.mac_embedding_model,
                batch_size=config.embed_batch_size,
            ),
            StorageObserver(vector_store),
            DocumentStorageObserver(document_store),
        ],
        max_workers=config.max_workers,
    )


def build_ingestion_service_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    vector_store_factory: VectorStoreFactory = MongoVectorStore.from_uri,
    document_store_factory: DocumentStoreFactory = MongoDocumentStore.from_uri,
    embedding_client_factory: EmbeddingClientFactory = PlatformEmbeddingClient,
) -> IngestionService | None:
    config = IngestionRuntimeConfig.from_env(environ)
    if config is None:
        return None
    return build_ingestion_service(
        config,
        vector_store_factory=vector_store_factory,
        document_store_factory=document_store_factory,
        embedding_client_factory=embedding_client_factory,
    )


def _first_non_empty(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _parse_positive_int(
    value: str | None,
    *,
    default: int,
    name: str,
) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise QueryError(f"{name} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise QueryError(f"{name} must be positive, got {parsed}")
    return parsed


def _parse_int(value: str | None, *, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise QueryError(f"RAG_GPU_LAYERS must be an integer, got {value!r}") from exc
