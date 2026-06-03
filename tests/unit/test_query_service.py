from __future__ import annotations

import uuid

import pytest

from services.query.service import NO_CONTEXT_ANSWER, QueryService
from services.shared.domain import Chunk
from services.shared.exceptions import QueryError


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [0.1] * 1024

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]


class FakeGenerationClient:
    def __init__(self, answer: str = "TCP is reliable. [1]") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeVectorStore:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[list[float], int, uuid.UUID]] = []

    def search(
        self,
        vec: list[float],
        k: int,
        *,
        user_id: uuid.UUID | str | None = None,
        doc_id: uuid.UUID | str | None = None,
    ) -> list[Chunk]:
        assert isinstance(user_id, uuid.UUID)
        self.calls.append((vec, k, user_id))
        return self.chunks


def _chunk() -> Chunk:
    user_id = uuid.uuid4()
    return Chunk(
        text="TCP provides reliable delivery.",
        doc_id=uuid.uuid4(),
        user_id=user_id,
        position=0,
    )


def test_query_service_retrieves_builds_prompt_and_generates_answer() -> None:
    user_id = uuid.uuid4()
    embedding = FakeEmbeddingClient()
    generation = FakeGenerationClient()
    store = FakeVectorStore([_chunk()])
    service = QueryService(
        embedding_client=embedding,
        generation_client=generation,
        vector_store=store,
        top_k=3,
    )

    result = service.ask(user_id, " What is TCP? ")

    assert result.answer == "TCP is reliable. [1]"
    assert len(result.sources) == 1
    assert embedding.inputs == ["What is TCP?"]
    assert store.calls[0][1] == 3
    assert store.calls[0][2] == user_id
    assert "What is TCP?" in generation.prompts[0]
    assert "TCP provides reliable delivery." in generation.prompts[0]


def test_query_service_returns_no_context_answer_without_generation() -> None:
    user_id = uuid.uuid4()
    generation = FakeGenerationClient()
    service = QueryService(
        embedding_client=FakeEmbeddingClient(),
        generation_client=generation,
        vector_store=FakeVectorStore([]),
    )

    result = service.ask(user_id, "What is UDP?")

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert generation.prompts == []


def test_query_service_rejects_empty_question() -> None:
    service = QueryService(
        embedding_client=FakeEmbeddingClient(),
        generation_client=FakeGenerationClient(),
        vector_store=FakeVectorStore([]),
    )

    with pytest.raises(QueryError, match="question"):
        service.ask(uuid.uuid4(), "   ")


def test_query_service_wraps_retrieval_errors() -> None:
    class BrokenStore(FakeVectorStore):
        def search(self, *args: object, **kwargs: object) -> list[Chunk]:
            raise RuntimeError("mongo down")

    service = QueryService(
        embedding_client=FakeEmbeddingClient(),
        generation_client=FakeGenerationClient(),
        vector_store=BrokenStore([]),
    )

    with pytest.raises(QueryError, match="retrieval failed"):
        service.ask(uuid.uuid4(), "question")

