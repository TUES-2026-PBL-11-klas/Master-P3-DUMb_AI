"""
Unit tests for the ingestion composition root (services.ingestion.composition).

Covers:
  - build_parser_registry(): registers TxtParser + MarkdownParser and
    exposes their combined extensions
  - build_ingestion_service_with(): assembles the canonical
    chunk -> embed -> store observer chain, in order, around the injected
    store and embed client
  - chunk_size is propagated to the ChunkingObserver
  - StorageObserver wraps the injected store (DI seam)
  - the assembled service exposes supported_extensions — the property
    dummy_server's upload allow-list relies on

These tests use a MagicMock vector store and a MagicMock embed client so
they never touch MongoDB or llama.cpp. They DO construct the real
ChunkingObserver, which loads tiktoken's cl100k_base encoding (cached
after first download) — exactly as test_chunker.py already does.

build_ingestion_service() (the production variant that builds a real
MongoVectorStore + LlamaCppEmbeddingClient from a URI/URL) is intentionally
NOT unit-tested here: it needs the live backends and is covered by manual
smoke testing + the socket integration suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.ingestion.composition import (
    build_ingestion_service_with,
    build_parser_registry,
)
from services.ingestion.observers.chunking import ChunkingObserver
from services.ingestion.observers.embedding import EmbeddingObserver
from services.ingestion.observers.storage import StorageObserver
from services.ingestion.parsers.markdown import MarkdownParser
from services.ingestion.parsers.txt import TxtParser
from services.ingestion.service import IngestionService


# Fixtures


@pytest.fixture
def store() -> MagicMock:
    """A MagicMock that satisfies VectorStore[Chunk] structurally."""
    s = MagicMock()
    s.store = MagicMock(return_value=None)
    s.search = MagicMock(return_value=[])
    return s


@pytest.fixture
def embed_client() -> MagicMock:
    """A MagicMock that satisfies LlamaCppClient structurally."""
    c = MagicMock()
    c.embed = MagicMock(return_value=[0.0] * 1024)
    c.embed_batch = MagicMock(side_effect=lambda texts: [[0.0] * 1024 for _ in texts])
    return c


# Registry assembly


class TestParserRegistry:
    def test_registers_two_parsers(self) -> None:
        registry = build_parser_registry()
        assert len(registry) == 2

    def test_resolves_txt_to_txt_parser(self) -> None:
        assert isinstance(build_parser_registry().get("txt"), TxtParser)

    def test_resolves_md_to_markdown_parser(self) -> None:
        assert isinstance(build_parser_registry().get("md"), MarkdownParser)

    def test_supported_extensions_cover_txt_and_md(self) -> None:
        exts = build_parser_registry().supported_extensions
        assert "txt" in exts
        assert "md" in exts
        assert "markdown" in exts


# Service assembly


class TestServiceAssembly:
    def test_returns_ingestion_service(
        self, store: MagicMock, embed_client: MagicMock
    ) -> None:
        with build_ingestion_service_with(store, embed_client) as svc:
            assert isinstance(svc, IngestionService)

    def test_canonical_observer_order(
        self, store: MagicMock, embed_client: MagicMock
    ) -> None:
        with build_ingestion_service_with(store, embed_client) as svc:
            chain = [type(o).__name__ for o in svc.observers]
        assert chain == ["ChunkingObserver", "EmbeddingObserver", "StorageObserver"]

    def test_storage_observer_wraps_injected_store(
        self, store: MagicMock, embed_client: MagicMock
    ) -> None:
        with build_ingestion_service_with(store, embed_client) as svc:
            storage_obs = svc.observers[2]
        assert isinstance(storage_obs, StorageObserver)
        assert storage_obs._store is store

    def test_embedding_observer_present(
        self, store: MagicMock, embed_client: MagicMock
    ) -> None:
        with build_ingestion_service_with(store, embed_client) as svc:
            assert isinstance(svc.observers[1], EmbeddingObserver)

    def test_chunk_size_is_propagated(
        self, store: MagicMock, embed_client: MagicMock
    ) -> None:
        with build_ingestion_service_with(store, embed_client, chunk_size=128) as svc:
            chunking = svc.observers[0]
        assert isinstance(chunking, ChunkingObserver)
        assert chunking.chunk_size == 128

    def test_supported_extensions_delegates_to_registry(
        self, store: MagicMock, embed_client: MagicMock
    ) -> None:
        with build_ingestion_service_with(store, embed_client) as svc:
            exts = svc.supported_extensions
        assert "txt" in exts
        assert "md" in exts
