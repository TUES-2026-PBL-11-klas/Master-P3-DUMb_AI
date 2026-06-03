"""
Runtime wiring for the RAG query pipeline.

This module is the composition root for QueryService: it reads environment
configuration, creates the concrete Mongo/Ollama dependencies, and returns a
ready-to-use QueryService instance.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from services.db.mongo_vector_store import MongoVectorStore
from services.query.service import DEFAULT_TOP_K, QueryService, QueryVectorStore
from services.shared.ollama_client import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaClient,
)
from services.shared.client import GenerationClient, LlamaCppClient
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


class OllamaClientFactory(Protocol):
    def __call__(
        self,
        *,
        base_url: str,
        embedding_model: str,
        generation_model: str,
    ) -> LlamaCppClient | GenerationClient: ...


@dataclass(frozen=True)
class QueryRuntimeConfig:
    mongo_uri: str
    mongo_db_name: str = DEFAULT_DB_NAME
    chunk_collection: str = DEFAULT_CHUNK_COLLECTION
    vector_index: str = DEFAULT_VECTOR_INDEX
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    generation_model: str = DEFAULT_GENERATION_MODEL
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

        return cls(
            mongo_uri=mongo_uri,
            mongo_db_name=env.get("RAG_MONGODB_DB", DEFAULT_DB_NAME),
            chunk_collection=env.get("RAG_CHUNK_COLLECTION", DEFAULT_CHUNK_COLLECTION),
            vector_index=env.get("RAG_VECTOR_INDEX", DEFAULT_VECTOR_INDEX),
            ollama_base_url=env.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
            embedding_model=env.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBEDDING_MODEL),
            generation_model=env.get(
                "OLLAMA_GENERATE_MODEL",
                DEFAULT_GENERATION_MODEL,
            ),
            top_k=top_k,
        )


def build_query_service(
    config: QueryRuntimeConfig,
    *,
    vector_store_factory: VectorStoreFactory = MongoVectorStore.from_uri,
    ollama_client_factory: OllamaClientFactory = OllamaClient,
) -> QueryService:
    """
    Build a QueryService from explicit runtime config.

    Factories are injectable so tests can verify wiring without opening real
    MongoDB connections or calling a real Ollama process.
    """
    model_client = ollama_client_factory(
        base_url=config.ollama_base_url,
        embedding_model=config.embedding_model,
        generation_model=config.generation_model,
    )
    vector_store = vector_store_factory(
        config.mongo_uri,
        db_name=config.mongo_db_name,
        collection_name=config.chunk_collection,
        index_name=config.vector_index,
    )

    return QueryService(
        embedding_client=model_client,  # type: ignore[arg-type]
        generation_client=model_client,  # type: ignore[arg-type]
        vector_store=vector_store,
        top_k=config.top_k,
    )


def build_query_service_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    vector_store_factory: VectorStoreFactory = MongoVectorStore.from_uri,
    ollama_client_factory: OllamaClientFactory = OllamaClient,
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
        ollama_client_factory=ollama_client_factory,
    )


def _first_non_empty(env: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _parse_positive_int(value: str | None, *, default: int) -> int:
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise QueryError(f"RAG_TOP_K must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise QueryError(f"RAG_TOP_K must be positive, got {parsed}")
    return parsed
