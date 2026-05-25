"""
ChunkingObserver — IngestionObserver implementation for the RAG pipeline.

Responsibilities:
  - Split the IngestionEvent's Document content into sentence-aligned Chunk
    objects whose token count stays at or below a configurable budget.
  - Attach the resulting list[Chunk] to the IngestionEvent for downstream
    observers (embedding, storage).
  - Advance the event status to CHUNKED on success, or mark it FAILED on error.

Design:
  - Implements the IngestionObserver protocol via structural subtyping:
    on_ingest(event: IngestionEvent) -> None.
  - Uses nltk.sent_tokenize for sentence boundary detection.
  - Uses tiktoken for token counting (fast, lightweight, good BGE-M3
    approximation — see note below).
  - Pure function _chunk_text() is stateless and fully unit-testable in isolation.

Chunking strategy (sentence-aligned, variable-width, capped):
  Sentences are packed into a chunk one at a time.
  When adding the next sentence would exceed _chunk_size tokens, the current
  accumulator is closed and yielded as-is — no padding is appended.
  Then a fresh accumulator starts with that sentence.

  This guarantees:
    - No chunk ever splits mid-sentence.
    - Every chunk contains <= _chunk_size tokens (uniform *upper bound* for
      BGE-M3 — the model handles variable-length input and pads internally
      per batch, so explicit padding here adds noise and buys nothing).
    - A single sentence is assumed never to exceed _chunk_size tokens; if it
      does, BigSentenceError is raised.

Note on tiktoken vs BGE-M3:
  cl100k_base and XLM-RoBERTa SentencePiece tokenize differently. Token
  counts here are an approximation, typically within ~30% of BGE-M3's view.
  That's fine for staying under BGE-M3's 8192-token limit.

Dependencies:
  pip install tiktoken nltk
  python -m nltk.downloader punkt_tab   # one-time download
"""

from __future__ import annotations

import logging
from typing import Iterator

import nltk
import tiktoken

from services.shared.domain import Chunk, Document, IngestionEvent, IngestionStatus
from services.shared.exceptions import BigSentenceError

logger = logging.getLogger(__name__)

_ENCODING_NAME = "cl100k_base"


def _ensure_nltk_punkt() -> None:
    """Download the Punkt sentence tokenizer data if it is not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("ChunkingObserver: downloading nltk 'punkt_tab' data…")
        nltk.download("punkt_tab", quiet=True)


class ChunkingObserver:
    """
    Concrete Observer — splits the event's Document.content into
    sentence-aligned Chunk objects, each <= _chunk_size tokens wide, and
    attaches them to the IngestionEvent.

    Satisfies shared.protocols.IngestionObserver through structural subtyping.

    Attributes:
        _chunk_size:  Maximum token width per emitted chunk (default 400).
        _enc:         tiktoken Encoding instance, shared across calls.
    """

    def __init__(self, chunk_size: int = 400) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        _ensure_nltk_punkt()
        self._chunk_size = chunk_size
        self._enc = tiktoken.get_encoding(_ENCODING_NAME)

    def on_ingest(self, event: IngestionEvent) -> None:
        doc = event.document
        logger.info(
            "ChunkingObserver: chunking document '%s' (id=%s)",
            doc.filename,
            doc.id,
        )
        try:
            chunks: list[Chunk] = list(self._chunk_document(doc))
        except BigSentenceError as exc:
            logger.error(
                "ChunkingObserver: failed to chunk document '%s': %s",
                doc.filename,
                exc,
            )
            event.fail(str(exc))
            raise

        event.chunks = chunks
        event.status = IngestionStatus.CHUNKED
        logger.info(
            "ChunkingObserver: produced %d chunks for document '%s'",
            len(chunks),
            doc.filename,
        )

    def _chunk_document(self, doc: Document) -> Iterator[Chunk]:
        yield from (
            Chunk(
                doc_id=doc.id,
                user_id=doc.user_id,
                text=text,
                embedding=[],
                position=position,
            )
            for position, text in enumerate(self._chunk_text(doc.content))
        )

    def _chunk_text(self, text: str) -> Iterator[str]:
        """
        Pack sentences into windows whose total token count is <= _chunk_size.

        Sentences are accumulated as strings and joined with a single space on
        yield, preserving the whitespace that existed between them in the
        source text. Token counts are tracked separately via tiktoken and are
        never used for decode — this avoids the lossy encode→extend→decode
        round-trip that would fuse adjacent sentences without a separator.

        Raises:
            BigSentenceError: if any single sentence exceeds _chunk_size tokens.
        """
        sentences: list[str] = nltk.sent_tokenize(text)

        if not sentences:
            logger.warning(
                "ChunkingObserver: received empty text — no chunks produced."
            )
            return

        window_texts: list[str] = []
        window_token_count: int = 0

        for sentence in sentences:
            sentence_tokens = self._enc.encode(sentence)

            if len(sentence_tokens) > self._chunk_size:
                raise BigSentenceError(
                    f"Sentence exceeds chunk_size ({self._chunk_size} tokens) — "
                    f"got {len(sentence_tokens)} tokens: {sentence[:100]!r}"
                )

            if window_token_count + len(sentence_tokens) <= self._chunk_size:
                window_texts.append(sentence)
                window_token_count += len(sentence_tokens)
            else:
                if window_texts:
                    yield " ".join(window_texts)
                window_texts = [sentence]
                window_token_count = len(sentence_tokens)

        if window_texts:
            yield " ".join(window_texts)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def __repr__(self) -> str:
        return f"ChunkingObserver(chunk_size={self._chunk_size})"