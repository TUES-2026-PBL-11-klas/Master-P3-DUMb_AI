from __future__ import annotations

import uuid

import pytest

from services.query.wiring import (
    DEFAULT_CHUNK_COLLECTION,
    DEFAULT_DB_NAME,
    DEFAULT_VECTOR_INDEX,
    QueryRuntimeConfig,
    build_query_service_from_env,
)
from services.shared.domain import Chunk
from services.shared.exceptions import QueryError


class FakeOllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        embedding_model: str,
        generation_model: str,
    ) -> None:
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.prompts: list[str] = []

    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Generated answer."


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int, uuid.UUID | str | None]] = []
        self.chunk = Chunk(
            text="TCP provides reliable delivery.",
            doc_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            position=0,
        )

    def search(
        self,
        vec: list[float],
        k: int,
        *,
        user_id: uuid.UUID | str | None = None,
        doc_id: uuid.UUID | str | None = None,
    ) -> list[Chunk]:
        self.calls.append((vec, k, user_id))
        return [self.chunk]


def test_query_runtime_config_returns_none_without_mongo_uri() -> None:
    assert QueryRuntimeConfig.from_env({}) is None


def test_query_runtime_config_reads_env_values() -> None:
    config = QueryRuntimeConfig.from_env(
        {
            "MONGODB_URI": "mongodb://fallback:27017",
            "RAG_MONGODB_URI": "mongodb://rag:27017",
            "RAG_MONGODB_DB": "custom_db",
            "RAG_CHUNK_COLLECTION": "chunks",
            "RAG_VECTOR_INDEX": "vector_idx",
            "OLLAMA_BASE_URL": "http://ollama:11434",
            "OLLAMA_EMBED_MODEL": "bge-custom",
            "OLLAMA_GENERATE_MODEL": "qwen-custom",
            "RAG_TOP_K": "7",
        }
    )

    assert config is not None
    assert config.mongo_uri == "mongodb://rag:27017"
    assert config.mongo_db_name == "custom_db"
    assert config.chunk_collection == "chunks"
    assert config.vector_index == "vector_idx"
    assert config.ollama_base_url == "http://ollama:11434"
    assert config.embedding_model == "bge-custom"
    assert config.generation_model == "qwen-custom"
    assert config.top_k == 7


@pytest.mark.parametrize("top_k", ["0", "-1", "not-int"])
def test_query_runtime_config_rejects_invalid_top_k(top_k: str) -> None:
    with pytest.raises(QueryError, match="RAG_TOP_K"):
        QueryRuntimeConfig.from_env(
            {
                "MONGODB_URI": "mongodb://localhost:27018",
                "RAG_TOP_K": top_k,
            }
        )


def test_build_query_service_from_env_wires_factories() -> None:
    created_clients: list[FakeOllamaClient] = []
    created_stores: list[FakeVectorStore] = []
    vector_factory_calls: list[tuple[str, str, str, str]] = []

    def ollama_factory(
        *,
        base_url: str,
        embedding_model: str,
        generation_model: str,
    ) -> FakeOllamaClient:
        client = FakeOllamaClient(
            base_url=base_url,
            embedding_model=embedding_model,
            generation_model=generation_model,
        )
        created_clients.append(client)
        return client

    def vector_store_factory(
        uri: str,
        *,
        db_name: str,
        collection_name: str,
        index_name: str,
    ) -> FakeVectorStore:
        vector_factory_calls.append((uri, db_name, collection_name, index_name))
        store = FakeVectorStore()
        created_stores.append(store)
        return store

    service = build_query_service_from_env(
        {
            "MONGODB_URI": "mongodb://localhost:27018",
            "OLLAMA_BASE_URL": "http://ollama:11434",
            "OLLAMA_EMBED_MODEL": "bge-m3",
            "OLLAMA_GENERATE_MODEL": "qwen2.5:3b",
            "RAG_TOP_K": "3",
        },
        vector_store_factory=vector_store_factory,
        ollama_client_factory=ollama_factory,
    )

    assert service is not None
    user_id = uuid.uuid4()
    result = service.ask(user_id, "What is TCP?")

    assert result.answer == "Generated answer."
    assert result.sources == [created_stores[0].chunk]
    assert created_stores[0].calls[0][1] == 3
    assert created_stores[0].calls[0][2] == user_id
    assert created_clients[0].base_url == "http://ollama:11434"
    assert created_clients[0].embedding_model == "bge-m3"
    assert created_clients[0].generation_model == "qwen2.5:3b"
    assert vector_factory_calls == [
        (
            "mongodb://localhost:27018",
            DEFAULT_DB_NAME,
            DEFAULT_CHUNK_COLLECTION,
            DEFAULT_VECTOR_INDEX,
        )
    ]


def test_build_query_service_from_env_returns_none_without_mongo_uri() -> None:
    assert build_query_service_from_env({}) is None
