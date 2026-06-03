from __future__ import annotations

import pytest

from services.ingestion.wiring import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_INGEST_MAX_WORKERS,
    IngestionRuntimeConfig,
    build_ingestion_service,
    build_ingestion_service_from_env,
)
from services.shared.client import (
    DEFAULT_EMBEDDING_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_LINUX_EMBEDDING_MODEL_PATH,
    DEFAULT_MAC_EMBEDDING_MODEL,
)
from services.shared.exceptions import QueryError


class FakeEmbeddingClient:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class FakeVectorStore:
    pass


class FakeChunkingObserver:
    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size


class FakeEmbeddingObserver:
    def __init__(
        self,
        *,
        client: FakeEmbeddingClient,
        model: str,
        batch_size: int,
    ) -> None:
        self.client = client
        self.model = model
        self.batch_size = batch_size


class FakeStorageObserver:
    def __init__(self, store: FakeVectorStore) -> None:
        self.store = store


def test_ingestion_runtime_config_returns_none_without_mongo_uri() -> None:
    assert IngestionRuntimeConfig.from_env({}) is None


def test_ingestion_runtime_config_uses_defaults() -> None:
    config = IngestionRuntimeConfig.from_env(
        {"MONGODB_URI": "mongodb://localhost:27018"}
    )

    assert config is not None
    assert config.mac_embedding_model == DEFAULT_MAC_EMBEDDING_MODEL
    assert config.linux_embedding_model_path == DEFAULT_LINUX_EMBEDDING_MODEL_PATH
    assert config.embedding_context == DEFAULT_EMBEDDING_CONTEXT
    assert config.gpu_layers == DEFAULT_GPU_LAYERS
    assert config.chunk_size == DEFAULT_CHUNK_SIZE
    assert config.embed_batch_size == DEFAULT_EMBED_BATCH_SIZE
    assert config.max_workers == DEFAULT_INGEST_MAX_WORKERS


def test_ingestion_runtime_config_reads_env_values() -> None:
    config = IngestionRuntimeConfig.from_env(
        {
            "RAG_MONGODB_URI": "mongodb://rag:27017",
            "RAG_MONGODB_DB": "custom_db",
            "RAG_CHUNK_COLLECTION": "chunks",
            "RAG_VECTOR_INDEX": "vector_idx",
            "RAG_MAC_EMBED_MODEL": "mac-embed",
            "RAG_LINUX_EMBED_MODEL_PATH": "models/embed.gguf",
            "RAG_EMBED_N_CTX": "4096",
            "RAG_GPU_LAYERS": "8",
            "RAG_CHUNK_SIZE": "320",
            "RAG_EMBED_BATCH_SIZE": "4",
            "RAG_INGEST_MAX_WORKERS": "2",
        }
    )

    assert config is not None
    assert config.mongo_uri == "mongodb://rag:27017"
    assert config.mongo_db_name == "custom_db"
    assert config.chunk_collection == "chunks"
    assert config.vector_index == "vector_idx"
    assert config.mac_embedding_model == "mac-embed"
    assert config.linux_embedding_model_path == "models/embed.gguf"
    assert config.embedding_context == 4096
    assert config.gpu_layers == 8
    assert config.chunk_size == 320
    assert config.embed_batch_size == 4
    assert config.max_workers == 2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_EMBED_N_CTX", "0"),
        ("RAG_CHUNK_SIZE", "-1"),
        ("RAG_EMBED_BATCH_SIZE", "not-int"),
        ("RAG_INGEST_MAX_WORKERS", "0"),
        ("RAG_GPU_LAYERS", "not-int"),
    ],
)
def test_ingestion_runtime_config_rejects_invalid_numeric_env(
    name: str,
    value: str,
) -> None:
    with pytest.raises(QueryError, match=name):
        IngestionRuntimeConfig.from_env(
            {
                "MONGODB_URI": "mongodb://localhost:27018",
                name: value,
            }
        )


def test_build_ingestion_service_wires_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.ingestion.wiring as wiring

    monkeypatch.setattr(wiring, "ChunkingObserver", FakeChunkingObserver)
    monkeypatch.setattr(wiring, "EmbeddingObserver", FakeEmbeddingObserver)
    monkeypatch.setattr(wiring, "StorageObserver", FakeStorageObserver)

    created_clients: list[FakeEmbeddingClient] = []
    created_stores: list[FakeVectorStore] = []
    vector_factory_calls: list[tuple[str, str, str, str]] = []

    def embedding_factory(
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> FakeEmbeddingClient:
        assert mac_model == "mac-embed"
        assert linux_model_path == "models/embed.gguf"
        assert n_ctx == 4096
        assert n_gpu_layers == 5
        client = FakeEmbeddingClient()
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

    service = build_ingestion_service(
        IngestionRuntimeConfig(
            mongo_uri="mongodb://localhost:27018",
            mac_embedding_model="mac-embed",
            linux_embedding_model_path="models/embed.gguf",
            embedding_context=4096,
            gpu_layers=5,
            chunk_size=320,
            embed_batch_size=4,
            max_workers=2,
        ),
        vector_store_factory=vector_store_factory,
        embedding_client_factory=embedding_factory,
    )

    assert len(service.observers) == 3
    assert isinstance(service.observers[0], FakeChunkingObserver)
    assert service.observers[0].chunk_size == 320
    assert isinstance(service.observers[1], FakeEmbeddingObserver)
    assert service.observers[1].client is created_clients[0]
    assert service.observers[1].batch_size == 4
    assert isinstance(service.observers[2], FakeStorageObserver)
    assert service.observers[2].store is created_stores[0]
    assert vector_factory_calls == [
        (
            "mongodb://localhost:27018",
            "dumb_ai",
            "document_chunks",
            "chunk_embedding_vector_index",
        )
    ]
    service.close()


def test_build_ingestion_service_from_env_returns_none_without_mongo_uri() -> None:
    assert build_ingestion_service_from_env({}) is None
