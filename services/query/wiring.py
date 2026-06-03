"""
Runtime wiring for the RAG query pipeline.

This module is the composition root for QueryService: it reads environment
configuration, creates the concrete Mongo/native model dependencies, and returns a
ready-to-use QueryService instance.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from services.db.mongo_vector_store import MongoVectorStore
from services.query.service import DEFAULT_TOP_K, QueryService, QueryVectorStore
from services.shared.client import (
    DEFAULT_EMBEDDING_CONTEXT,
    DEFAULT_GENERATION_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_LINUX_EMBEDDING_MODEL_PATH,
    DEFAULT_LINUX_GENERATION_MODEL_PATH,
    DEFAULT_MAC_EMBEDDING_MODEL,
    DEFAULT_MAC_GENERATION_MODEL,
    DEFAULT_MAX_TOKENS,
    GenerationClient,
    LlamaCppClient,
    NativeGenerationClient,
    PlatformEmbeddingClient,
)
from services.shared.exceptions import QueryError

DEFAULT_DB_NAME = "dumb_ai"
DEFAULT_CHUNK_COLLECTION = "document_chunks"
DEFAULT_VECTOR_INDEX = "chunk_embedding_vector_index"


class VectorStoreFactory(Protocol):
    def __call__(
        self,
        uri: str,
        *,
        db_name: str,
        collection_name: str,
        index_name: str,
    ) -> QueryVectorStore: ...


class EmbeddingClientFactory(Protocol):
    def __call__(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> LlamaCppClient: ...


class GenerationClientFactory(Protocol):
    def __call__(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        max_tokens: int,
    ) -> GenerationClient: ...


@dataclass(frozen=True)
class QueryRuntimeConfig:
    mongo_uri: str
    mongo_db_name: str = DEFAULT_DB_NAME
    chunk_collection: str = DEFAULT_CHUNK_COLLECTION
    vector_index: str = DEFAULT_VECTOR_INDEX
    mac_embedding_model: str = DEFAULT_MAC_EMBEDDING_MODEL
    linux_embedding_model_path: str = DEFAULT_LINUX_EMBEDDING_MODEL_PATH
    mac_generation_model: str = DEFAULT_MAC_GENERATION_MODEL
    linux_generation_model_path: str = DEFAULT_LINUX_GENERATION_MODEL_PATH
    embedding_context: int = DEFAULT_EMBEDDING_CONTEXT
    generation_context: int = DEFAULT_GENERATION_CONTEXT
    gpu_layers: int = DEFAULT_GPU_LAYERS
    max_tokens: int = DEFAULT_MAX_TOKENS
    top_k: int = DEFAULT_TOP_K

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "QueryRuntimeConfig | None":
        env = environ or os.environ
        mongo_uri = _first_non_empty(env, "RAG_MONGODB_URI", "MONGODB_URI")
        if mongo_uri is None:
            return None

        top_k = _parse_positive_int(env.get("RAG_TOP_K"), default=DEFAULT_TOP_K)
        embedding_context = _parse_positive_int(
            env.get("RAG_EMBED_N_CTX"),
            default=DEFAULT_EMBEDDING_CONTEXT,
            name="RAG_EMBED_N_CTX",
        )
        generation_context = _parse_positive_int(
            env.get("RAG_GENERATE_N_CTX"),
            default=DEFAULT_GENERATION_CONTEXT,
            name="RAG_GENERATE_N_CTX",
        )
        max_tokens = _parse_positive_int(
            env.get("RAG_GENERATE_MAX_TOKENS"),
            default=DEFAULT_MAX_TOKENS,
            name="RAG_GENERATE_MAX_TOKENS",
        )

        return cls(
            mongo_uri=mongo_uri,
            mongo_db_name=env.get("RAG_MONGODB_DB", DEFAULT_DB_NAME),
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
            mac_generation_model=env.get(
                "RAG_MAC_GENERATE_MODEL",
                DEFAULT_MAC_GENERATION_MODEL,
            ),
            linux_generation_model_path=env.get(
                "RAG_LINUX_GENERATE_MODEL_PATH",
                DEFAULT_LINUX_GENERATION_MODEL_PATH,
            ),
            embedding_context=embedding_context,
            generation_context=generation_context,
            gpu_layers=_parse_int(env.get("RAG_GPU_LAYERS"), default=DEFAULT_GPU_LAYERS),
            max_tokens=max_tokens,
            top_k=top_k,
        )


def build_query_service(
    config: QueryRuntimeConfig,
    *,
    vector_store_factory: VectorStoreFactory = MongoVectorStore.from_uri,
    embedding_client_factory: EmbeddingClientFactory = PlatformEmbeddingClient,
    generation_client_factory: GenerationClientFactory = NativeGenerationClient,
) -> QueryService:
    """
    Build a QueryService from explicit runtime config.

    Factories are injectable so tests can verify wiring without opening real
    MongoDB connections or loading real local model runtimes.
    """
    embedding_client = embedding_client_factory(
        mac_model=config.mac_embedding_model,
        linux_model_path=config.linux_embedding_model_path,
        n_ctx=config.embedding_context,
        n_gpu_layers=config.gpu_layers,
    )
    generation_client = generation_client_factory(
        mac_model=config.mac_generation_model,
        linux_model_path=config.linux_generation_model_path,
        n_ctx=config.generation_context,
        n_gpu_layers=config.gpu_layers,
        max_tokens=config.max_tokens,
    )
    vector_store = vector_store_factory(
        config.mongo_uri,
        db_name=config.mongo_db_name,
        collection_name=config.chunk_collection,
        index_name=config.vector_index,
    )

    return QueryService(
        embedding_client=embedding_client,
        generation_client=generation_client,
        vector_store=vector_store,
        top_k=config.top_k,
    )


def build_query_service_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    vector_store_factory: VectorStoreFactory = MongoVectorStore.from_uri,
    embedding_client_factory: EmbeddingClientFactory = PlatformEmbeddingClient,
    generation_client_factory: GenerationClientFactory = NativeGenerationClient,
) -> QueryService | None:
    """
    Build QueryService from env vars.

    Returns None when no Mongo URI is configured, letting the server keep query
    handling disabled while still booting for auth/upload demos.
    """
    config = QueryRuntimeConfig.from_env(environ)
    if config is None:
        return None
    return build_query_service(
        config,
        vector_store_factory=vector_store_factory,
        embedding_client_factory=embedding_client_factory,
        generation_client_factory=generation_client_factory,
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
    name: str = "RAG_TOP_K",
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
