from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from services.shared.domain import DocumentStatus, IngestionStatus
from services.shared.exceptions import EmbeddingError

if TYPE_CHECKING:
    from services.shared.domain import IngestionEvent
    from services.shared.protocols import AIInterface

logger = logging.getLogger(__name__)

_EXPECTED_DIM = 1024


class EmbeddingObserver:
    def __init__(
        self,
        client: "AIInterface",
        *,
        model: str = "bge-m3",
        batch_size: int = 1,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def on_ingest(self, event: "IngestionEvent") -> None:
        chunks = event.chunks
        if not chunks:
            logger.warning(
                "EmbeddingObserver: no chunks for '%s', skipping.",
                event.document.filename,
            )
            return

        logger.info(
            "EmbeddingObserver: embedding %d chunk(s) from '%s'",
            len(chunks),
            event.document.filename,
        )
        texts = [chunk.text for chunk in chunks]

        try:
            if hasattr(self._client, "embed_batch"):
                vectors = self._embed_batch(texts)
            else:
                vectors = self._embed_sequential(texts)
        except EmbeddingError as exc:
            logger.error(
                "EmbeddingObserver: failed for '%s': %s", event.document.filename, exc
            )
            event.fail(str(exc))
            raise

        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector

        event.status = IngestionStatus.EMBEDDED
        event.document.mark_status(DocumentStatus.EMBEDDED)
        logger.info(
            "EmbeddingObserver: embedded %d chunk(s) for '%s'",
            len(chunks),
            event.document.filename,
        )

    def _embed_sequential(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i, text in enumerate(texts):
            try:
                vec = self._client.embed(text)
            except Exception as exc:
                raise EmbeddingError(f"Embedding failed for chunk {i}: {exc}") from exc
            self._validate_vector(vec, chunk_index=i)
            vectors.append(vec)
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            try:
                batch_vectors = self._client.embed_batch(batch)
            except Exception as exc:
                raise EmbeddingError(
                    f"Batch embedding failed for chunks {batch_start}–{batch_start + len(batch) - 1}: {exc}"
                ) from exc
            if len(batch_vectors) != len(batch):
                raise EmbeddingError(
                    f"embed_batch returned {len(batch_vectors)} vectors for {len(batch)} inputs"
                )
            for j, vec in enumerate(batch_vectors):
                self._validate_vector(vec, chunk_index=batch_start + j)
            vectors.extend(batch_vectors)
        return vectors

    @staticmethod
    def _validate_vector(vec: list[float], *, chunk_index: int) -> None:
        if len(vec) != _EXPECTED_DIM:
            raise EmbeddingError(
                f"Expected {_EXPECTED_DIM}-dim vector for chunk {chunk_index}, got {len(vec)}-dim"
            )

    def __repr__(self) -> str:
        return (
            f"EmbeddingObserver(model={self._model!r}, batch_size={self._batch_size})"
        )
