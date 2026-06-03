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
from services.shared.client import (
    DEFAULT_EMBEDDING_CONTEXT,
    DEFAULT_GENERATION_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_LINUX_EMBEDDING_MODEL_PATH,
    DEFAULT_LINUX_GENERATION_MODEL_PATH,
    DEFAULT_MAC_EMBEDDING_MODEL,
    DEFAULT_MAC_GENERATION_MODEL,
    DEFAULT_MAX_TOKENS,
)
from services.shared.domain import Chunk
from services.shared.exceptions import QueryError


class FakeEmbeddingClient:
    def __init__(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> None:
        self.mac_model = mac_model
        self.linux_model_path = linux_model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers

    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class FakeGenerationClient:
    def __init__(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        max_tokens: int,
    ) -> None:
        self.mac_model = mac_model
        self.linux_model_path = linux_model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.prompts: list[str] = []

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


def test_query_runtime_config_uses_native_defaults() -> None:
    config = QueryRuntimeConfig.from_env({"MONGODB_URI": "mongodb://localhost:27018"})

    assert config is not None
    assert config.mac_embedding_model == DEFAULT_MAC_EMBEDDING_MODEL
    assert config.linux_embedding_model_path == DEFAULT_LINUX_EMBEDDING_MODEL_PATH
    assert config.mac_generation_model == DEFAULT_MAC_GENERATION_MODEL
    assert config.linux_generation_model_path == DEFAULT_LINUX_GENERATION_MODEL_PATH
    assert config.embedding_context == DEFAULT_EMBEDDING_CONTEXT
    assert config.generation_context == DEFAULT_GENERATION_CONTEXT
    assert config.gpu_layers == DEFAULT_GPU_LAYERS
    assert config.max_tokens == DEFAULT_MAX_TOKENS


def test_query_runtime_config_reads_env_values() -> None:
    config = QueryRuntimeConfig.from_env(
        {
            "MONGODB_URI": "mongodb://fallback:27017",
            "RAG_MONGODB_URI": "mongodb://rag:27017",
            "RAG_MONGODB_DB": "custom_db",
            "RAG_CHUNK_COLLECTION": "chunks",
            "RAG_VECTOR_INDEX": "vector_idx",
            "RAG_MAC_EMBED_MODEL": "mac-embed",
            "RAG_LINUX_EMBED_MODEL_PATH": "models/embed.gguf",
            "RAG_MAC_GENERATE_MODEL": "mac-generate",
            "RAG_LINUX_GENERATE_MODEL_PATH": "models/generate.gguf",
            "RAG_EMBED_N_CTX": "4096",
            "RAG_GENERATE_N_CTX": "2048",
            "RAG_GPU_LAYERS": "12",
            "RAG_GENERATE_MAX_TOKENS": "300",
            "RAG_TOP_K": "7",
        }
    )

    assert config is not None
    assert config.mongo_uri == "mongodb://rag:27017"
    assert config.mongo_db_name == "custom_db"
    assert config.chunk_collection == "chunks"
    assert config.vector_index == "vector_idx"
    assert config.mac_embedding_model == "mac-embed"
    assert config.linux_embedding_model_path == "models/embed.gguf"
    assert config.mac_generation_model == "mac-generate"
    assert config.linux_generation_model_path == "models/generate.gguf"
    assert config.embedding_context == 4096
    assert config.generation_context == 2048
    assert config.gpu_layers == 12
    assert config.max_tokens == 300
    assert config.top_k == 7


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_TOP_K", "0"),
        ("RAG_EMBED_N_CTX", "-1"),
        ("RAG_GENERATE_N_CTX", "not-int"),
        ("RAG_GENERATE_MAX_TOKENS", "0"),
        ("RAG_GPU_LAYERS", "not-int"),
    ],
)
def test_query_runtime_config_rejects_invalid_numeric_env(
    name: str,
    value: str,
) -> None:
    with pytest.raises(QueryError, match=name):
        QueryRuntimeConfig.from_env(
            {
                "MONGODB_URI": "mongodb://localhost:27018",
                name: value,
            }
        )


def test_build_query_service_from_env_wires_factories() -> None:
    created_embeddings: list[FakeEmbeddingClient] = []
    created_generators: list[FakeGenerationClient] = []
    created_stores: list[FakeVectorStore] = []
    vector_factory_calls: list[tuple[str, str, str, str]] = []

    def embedding_factory(
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> FakeEmbeddingClient:
        client = FakeEmbeddingClient(
            mac_model=mac_model,
            linux_model_path=linux_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )
        created_embeddings.append(client)
        return client

    def generation_factory(
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
        max_tokens: int,
    ) -> FakeGenerationClient:
        client = FakeGenerationClient(
            mac_model=mac_model,
            linux_model_path=linux_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            max_tokens=max_tokens,
        )
        created_generators.append(client)
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
            "RAG_MAC_EMBED_MODEL": "mac-embed",
            "RAG_LINUX_EMBED_MODEL_PATH": "models/embed.gguf",
            "RAG_MAC_GENERATE_MODEL": "mac-generate",
            "RAG_LINUX_GENERATE_MODEL_PATH": "models/generate.gguf",
            "RAG_EMBED_N_CTX": "4096",
            "RAG_GENERATE_N_CTX": "2048",
            "RAG_GPU_LAYERS": "5",
            "RAG_GENERATE_MAX_TOKENS": "128",
            "RAG_TOP_K": "3",
        },
        vector_store_factory=vector_store_factory,
        embedding_client_factory=embedding_factory,
        generation_client_factory=generation_factory,
    )

    assert service is not None
    user_id = uuid.uuid4()
    result = service.ask(user_id, "What is TCP?")

    assert result.answer == "Generated answer."
    assert result.sources == [created_stores[0].chunk]
    assert created_stores[0].calls[0][1] == 3
    assert created_stores[0].calls[0][2] == user_id
    assert created_embeddings[0].mac_model == "mac-embed"
    assert created_embeddings[0].linux_model_path == "models/embed.gguf"
    assert created_embeddings[0].n_ctx == 4096
    assert created_embeddings[0].n_gpu_layers == 5
    assert created_generators[0].mac_model == "mac-generate"
    assert created_generators[0].linux_model_path == "models/generate.gguf"
    assert created_generators[0].n_ctx == 2048
    assert created_generators[0].n_gpu_layers == 5
    assert created_generators[0].max_tokens == 128
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
