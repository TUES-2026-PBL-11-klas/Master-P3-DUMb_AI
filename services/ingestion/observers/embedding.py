"""
EmbeddingObserver — IngestionObserver that generates vector embeddings.

Responsibilities:
  - Receive an IngestionEvent (which already contains chunks from ChunkingObserver).
  - Call the embedding model (BGE-M3 via the llama.cpp client) for each chunk's text.
  - Write the resulting 1024-dimensional vector into each Chunk's embedding field.

This observer MUST run after ChunkingObserver and before StorageObserver
in the observer chain, since it reads chunk text and writes embeddings
that StorageObserver will persist.

Design pattern: Observer
    Implements shared.protocols.IngestionObserver.
    IngestionService (the Subject) calls on_ingest() for every document
    entering the pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from shared.exceptions import EmbeddingError

if TYPE_CHECKING:
    from shared.client import (
        LlamaCppClient,
    )  # if the name of the client class/dir changes, the import here needs to be updated
    from shared.domain import IngestionEvent

logger = logging.getLogger(__name__)

# BGE-M3 produces 1024-dimensional vectors.
_EXPECTED_DIM = 1024


class EmbeddingObserver:
    """
    Observer that enriches Chunk objects with BGE-M3 embeddings.

    Satisfies shared.protocols.IngestionObserver through structural
    subtyping (no explicit inheritance needed).

    The observer delegates embedding generation to a client object that
    exposes an embed(text: str) -> list[float] method.  In production
    this is the llama.cpp HTTP client wrapping BGE-M3; in tests it can be
    any object with the same signature.

    Attributes:
        _client: Object with an embed(text) -> list[float] method.
        _model:  Model identifier passed through to the client (default
                 "bge-m3").  Stored for logging / diagnostics only if
                 the client doesn't need it.
    """

    def __init__(
        self,
        client: LlamaCppClient,
        *,
        model: str = "bge-m3",
        batch_size: int = 16,
    ) -> None:
        """
        Args:
            client:     Embedding client with embed(text) -> list[float]
                        and optionally embed_batch(texts) -> list[list[float]].
            model:      Model name for logging / diagnostics.
            batch_size: Maximum number of texts to embed in a single
                        embed_batch call.  Ignored when the client does
                        not support batching.
        """
        self._client = client
        self._model = model
        self._batch_size = batch_size

    # IngestionObserver interface

    def on_ingest(self, event: IngestionEvent) -> None:
        """
        Generate embeddings for every chunk in *event* and write them
        into each chunk's embedding field in-place.

        The method tries embed_batch first (if the client supports it)
        for throughput, falling back to per-chunk embed calls.

        Args:
            event: The IngestionEvent populated by ChunkingObserver.
                   event.chunks must be non-empty.

        Raises:
            shared.exceptions.EmbeddingError:
                if the embedding client fails for any chunk.
        """
        chunks = event.chunks

        if not chunks:
            logger.warning(
                "EmbeddingObserver: no chunks in event for document '%s', "
                "skipping embedding.",
                event.document.filename,
            )
            return

        logger.info(
            "EmbeddingObserver: embedding %d chunk(s) from '%s' with model '%s'",
            len(chunks),
            event.document.filename,
            self._model,
        )

        texts = [chunk.text for chunk in chunks]

        # Try batch embedding first, fall back to one-by-one
        if hasattr(self._client, "embed_batch"):
            vectors = self._embed_batch(texts)
        else:
            vectors = self._embed_sequential(texts)

        # Write embeddings into chunks
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        logger.info(
            "EmbeddingObserver: successfully embedded %d chunk(s) "
            "(%d-dim vectors) for '%s'",
            len(chunks),
            len(vectors[0]) if vectors else 0,
            event.document.filename,
        )

    # Private helpers

    def _embed_sequential(self, texts: list[str]) -> list[list[float]]:
        """
        Embed each text individually via client.embed()

        Args:
            texts: List of chunk texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: if any single embed call fails.
        """
        vectors: list[list[float]] = []

        for i, text in enumerate(texts):
            try:
                vec = self._client.embed(text)
            except Exception as exc:
                raise EmbeddingError(
                    f"Embedding failed for chunk {i} ({len(text)} chars): {exc}"
                ) from exc

            self._validate_vector(vec, chunk_index=i)
            vectors.append(vec)

        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts in batches via client.embed_batch()

        Splits *texts* into groups of _batch_size and concatenates
        the results.

        Args:
            texts: List of chunk texts to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            EmbeddingError: if any batch call fails.
        """
        vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]

            try:
                batch_vectors = self._client.embed_batch(batch)
            except Exception as exc:
                raise EmbeddingError(
                    f"Batch embedding failed for chunks "
                    f"{batch_start}–{batch_start + len(batch) - 1}: {exc}"
                ) from exc

            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"embed_batch returned {len(batch_vectors)} vectors "
                    f"for {len(batch)} inputs (batch starting at {batch_start})"
                )

            for j, vec in enumerate(batch_vectors):
                self._validate_vector(vec, chunk_index=batch_start + j)

            vectors.extend(batch_vectors)

        return vectors

    @staticmethod
    def _validate_vector(vec: list[float], *, chunk_index: int) -> None:
        """
        Sanity-check an embedding vector's dimensionality.

        Raises:
            EmbeddingError: if the vector length does not match _EXPECTED_DIM.
        """
        if len(vec) != _EXPECTED_DIM:
            raise EmbeddingError(
                f"Expected {_EXPECTED_DIM}-dim vector for chunk {chunk_index}, "
                f"got {len(vec)}-dim"
            )

    def __repr__(self) -> str:
        return (
            f"EmbeddingObserver(model={self._model!r}, batch_size={self._batch_size})"
        )
