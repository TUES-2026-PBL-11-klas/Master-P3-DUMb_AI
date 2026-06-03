from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.shared.domain import Chunk
    from services.shared.protocols import LlamaCppClient, VectorStore


class PureVectorSearchEngine:
    def __init__(self, client: "LlamaCppClient", vector_store: "VectorStore[Chunk]") -> None:
        self._client = client
        self._store = vector_store

    def search_sources(self, query_str: str, top_k: int = 5) -> list["Chunk"]:
        query_vector = self._client.embed(query_str)
        return self._store.search(vec=query_vector, k=top_k)
