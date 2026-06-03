"""
PromptBuilder for the RAG query pipeline.

The builder is intentionally small and dependency-free: it only formats the
user question and retrieved chunks into a prompt that the generation model can
use. QueryService owns retrieval and model calls.
"""

from __future__ import annotations

from services.shared.domain import Chunk


class PromptBuilder:
    """Build prompts that force the model to answer from retrieved sources."""

    def build(self, question: str, chunks: list[Chunk]) -> str:
        cleaned_question = question.strip()
        source_blocks = "\n\n".join(
            self._format_source(index, chunk)
            for index, chunk in enumerate(chunks, start=1)
        )

        return (
            "You are a study assistant for TUES students.\n\n"
            "Rules:\n"
            "1. Answer only using the provided sources.\n"
            "2. If the sources do not contain enough information, say that you "
            "do not know based on the provided documents.\n"
            "3. Do not invent facts, examples, names, dates, or definitions.\n"
            "4. Cite the sources you used with [1], [2], etc.\n\n"
            f"Question:\n{cleaned_question}\n\n"
            f"Sources:\n{source_blocks}\n\n"
            "Answer:"
        )

    @staticmethod
    def _format_source(index: int, chunk: Chunk) -> str:
        source_meta = (
            f"doc_id={chunk.doc_id}, position={chunk.position}"
        )
        page = chunk.metadata.get("page")
        if page is not None:
            source_meta += f", page={page}"

        return f"[{index}] {source_meta}\n{chunk.text.strip()}"

