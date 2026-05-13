"""
ChunkingObserver — IngestionObserver implementation for the RAG pipeline.

Responsibilities:
  - Split a Document's content into fixed-size, sentence-aligned Chunk objects.
  - Attach the resulting list[Chunk] to the Document for downstream observers.

Design:
  - Implements the IngestionObserver protocol via structural subtyping.
  - Uses nltk.sent_tokenize for sentence boundary detection.
  - Uses tiktoken for token counting (fast, lightweight, good BGE-M3 approximation).
  - Pure function _chunk_text() is stateless and fully unit-testable in isolation.

Chunking strategy (sentence-aligned, fixed-width):
  Sentences are packed into a chunk one at a time.
  When adding the next sentence would exceed _chunk_size tokens, the current
  accumulator is closed — the remaining token slots are padded with the pad
  symbol (default "<PAD>") to bring every chunk to exactly _chunk_size tokens.
  Then a fresh accumulator starts with that sentence.

  This guarantees:
    - No chunk ever splits mid-sentence.
    - Every chunk is exactly _chunk_size tokens wide (uniform input for BGE-M3).
    - A single sentence is assumed never to exceed _chunk_size tokens on its own.

Example (chunk_size=400, pad="<PAD>"):
  Sentences totalling 380 tokens → chunk 1 emitted with 20 <PAD> tokens appended.
  Next batch of sentences fills up to ≤400 tokens → chunk 2, padded if needed.
  …and so on until all sentences are consumed.

Dependencies:
  pip install tiktoken nltk
  python -m nltk.downloader punkt_tab   # one-time download
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterator

import nltk
import tiktoken

from shared.domain import Chunk, Document
from shared.exceptions import BigSentenceError

logger = logging.getLogger(__name__)

# tiktoken encoding — cl100k_base is a close approximation for BGE-M3's vocab.
# Using "cl100k_base" (GPT-4 encoding) rather than a model-specific one because
# tiktoken doesn't ship a BGE-M3 codec; token counts will be approximate but
# consistent, which is all the pipeline needs.
_ENCODING_NAME = "cl100k_base"

# Padding symbol appended to fill unused token slots in a chunk.
# Must be a string that encodes to exactly one token in cl100k_base so that
# pad count = (chunk_size - used_tokens), keeping arithmetic simple.
_PAD_SYMBOL = "<|endoftext|>"


def _ensure_nltk_punkt() -> None:
    """Download the Punkt sentence tokenizer data if it is not already present."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("ChunkingObserver: downloading nltk 'punkt_tab' data…")
        nltk.download("punkt_tab", quiet=True)


class ChunkingObserver:
    """
    Concrete Observer — splits Document.content into sentence-aligned,
    fixed-width Chunk objects padded to exactly _chunk_size tokens.

    Satisfies shared.protocols.IngestionObserver through structural subtyping.

    Attributes:
        _chunk_size:  Exact token width of every emitted chunk (default 400).
        _pad_symbol:  String used to pad unused token slots (default "<|endoftext|>").
        _enc:         tiktoken Encoding instance, shared across calls.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        pad_symbol: str = _PAD_SYMBOL,
    ) -> None:
        _ensure_nltk_punkt()
        self._chunk_size = chunk_size
        self._pad_symbol = pad_symbol
        self._enc = tiktoken.get_encoding(_ENCODING_NAME)

        # Verify the pad symbol encodes to exactly 1 token so padding arithmetic
        # stays simple: pad_count = chunk_size - used_tokens.
        pad_tokens = self._enc.encode(self._pad_symbol)
        if len(pad_tokens) != 1:
            raise ValueError(
                f"pad_symbol '{pad_symbol}' encodes to {len(pad_tokens)} tokens; "
                "it must encode to exactly 1 token for fixed-width padding to work."
            )
        self._pad_token_id: int = pad_tokens[0]

    # IngestionObserver protocol
    def on_ingest(self, doc: Document) -> None:
        """
        Chunk doc.content into sentence-aligned, padded chunks and attach
        the results to doc.

        Args:
            doc: The Document produced by a DocumentParser. Its content must
                 be a non-empty string.
        """
        logger.info(
            "ChunkingObserver: chunking document '%s' (id=%s)",
            doc.filename,
            doc.id,
        )

        chunks: list[Chunk] = list(self._chunk_document(doc))
        doc.chunks = chunks

        logger.info(
            "ChunkingObserver: produced %d chunks for document '%s'",
            len(chunks),
            doc.filename,
        )

    # Core algorithm — separated for easy unit testing
    def _chunk_document(self, doc: Document) -> Iterator[Chunk]:
        """
        Sentence-tokenize doc.content and yield one padded Chunk per window.

        Args:
            doc: Source document.

        Yields:
            Chunk objects in position order (position=0, 1, 2, …).
        """
        yield from (
            Chunk(
                id=uuid.uuid4(),
                doc_id=doc.id,
                text=text,
                embedding=[],   # filled later by EmbeddingObserver
                position=position,
            )
            for position, text in enumerate(self._chunk_text(doc.content))
        )

    def _chunk_text(self, text: str) -> Iterator[str]:
        """
        Pack sentences into fixed-width token windows, padding the remainder.

        Algorithm:
          1. Split *text* into sentences with nltk.sent_tokenize.
          2. Encode each sentence to token IDs.
          3. Accumulate sentences into the current window until the next
             sentence would push the total over _chunk_size.
          4. When the limit would be exceeded, pad the accumulator to
             exactly _chunk_size tokens, decode and yield it, then start
             a fresh accumulator with the sentence that didn't fit.
          5. After all sentences are consumed, pad and yield the final
             (possibly partial) window if it contains anything.

        Args:
            text: Raw document content.

        Yields:
            Decoded, padded strings — each encoding to exactly _chunk_size tokens.

        Raises:
            BigSentenceError: if any single sentence exceeds _chunk_size tokens.
        """
        sentences: list[str] = nltk.sent_tokenize(text)

        if not sentences:
            logger.warning("ChunkingObserver: received empty text — no chunks produced.")
            return

        # Accumulator: token IDs for sentences committed to the current chunk.
        window: list[int] = []

        for sentence in sentences:
            sentence_tokens: list[int] = self._enc.encode(sentence)

            if len(sentence_tokens) > self._chunk_size:
                raise BigSentenceError(
                    f"Sentence exceeds chunk_size ({self._chunk_size} tokens) — "
                    f"got {len(sentence_tokens)} tokens: {sentence!r:.100}"
                )

            fits = len(window) + len(sentence_tokens) <= self._chunk_size

            if fits:
                # Sentence fits — add it to the current window.
                window.extend(sentence_tokens)
            else:
                # Sentence does not fit — close the current window first.
                if window:
                    yield self._pad_and_decode(window)

                # Start fresh with the sentence that didn't fit.
                window = sentence_tokens

        # Yield whatever is left in the final window.
        if window:
            yield self._pad_and_decode(window)

    def _pad_and_decode(self, token_ids: list[int]) -> str:
        """
        Pad *token_ids* to exactly _chunk_size tokens and decode to a string.

        Padding is appended at the end using _pad_token_id repeated as many
        times as needed to reach _chunk_size.

        Args:
            token_ids: Token IDs for the sentences in this chunk. Must be
                       <= _chunk_size in length (guaranteed by _chunk_text).

        Returns:
            A decoded string whose token length is exactly _chunk_size.
        """
        pad_count = self._chunk_size - len(token_ids)
        padded = token_ids + [self._pad_token_id] * pad_count
        return self._enc.decode(padded)

    # Introspection
    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def pad_symbol(self) -> str:
        return self._pad_symbol

    def __repr__(self) -> str:
        return (
            f"ChunkingObserver("
            f"chunk_size={self._chunk_size}, "
            f"pad_symbol={self._pad_symbol!r})"
        )