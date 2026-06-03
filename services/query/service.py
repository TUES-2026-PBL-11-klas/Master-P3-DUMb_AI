"""
QueryService for the RAG question-answering pipeline.

This service is the orchestration layer for user questions. It does not know
about sockets or the TUI; callers provide a user_id and question, and it returns
a QueryResult built from retrieved chunks and a generation model answer.
"""

from __future__ import annotations

import logging
from typing import Protocol
from uuid import UUID

from services.query.prompt_builder import PromptBuilder
from services.shared.client import GenerationClient, LlamaCppClient
from services.shared.domain import Chunk, QueryResult
from services.shared.exceptions import QueryError

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
NO_CONTEXT_ANSWER = (
    "I don't know based on the provided documents."
)


class QueryVectorStore(Protocol):
    """Vector store surface QueryService needs for retrieval."""

    def search(
        self,
        vec: list[float],
        k: int,
        *,
        user_id: UUID | str | None = None,
        doc_id: UUID | str | None = None,
    ) -> list[Chunk]:
        """Return nearest chunks for *vec*, optionally scoped by user/document."""
        ...


class QueryService:
    """
    Orchestrates the RAG query flow.

    Dependencies are injected so unit tests can use fake clients and stores,
    while production can wire MongoVectorStore and an Ollama-backed model client.
    """

    def __init__(
        self,
        *,
        embedding_client: LlamaCppClient,
        generation_client: GenerationClient,
        vector_store: QueryVectorStore,
        prompt_builder: PromptBuilder | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        self._embedding_client = embedding_client
        self._generation_client = generation_client
        self._vector_store = vector_store
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._top_k = top_k

    def ask(self, user_id: UUID, question: str) -> QueryResult:
        """
        Answer *question* using only chunks retrieved for *user_id*.

        Raises:
            QueryError: if validation, embedding, retrieval, prompt building,
                        or generation fails.
        """
        cleaned = question.strip()
        if not cleaned:
            raise QueryError("question must not be empty")

        logger.info("QueryService: answering query for user %s", user_id)

        try:
            query_vector = self._embedding_client.embed(cleaned)
            chunks = self._vector_store.search(
                query_vector,
                self._top_k,
                user_id=user_id,
            )
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError(f"retrieval failed: {exc}") from exc

        if not chunks:
            logger.info("QueryService: no sources found for user %s", user_id)
            return QueryResult(answer=NO_CONTEXT_ANSWER, sources=[])

        try:
            prompt = self._prompt_builder.build(cleaned, chunks)
            answer = self._generation_client.generate(prompt).strip()
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError(f"generation failed: {exc}") from exc

        if not answer:
            raise QueryError("generation returned an empty answer")

        return QueryResult(answer=answer, sources=chunks)
