from __future__ import annotations

import uuid

from services.query.prompt_builder import PromptBuilder
from services.shared.domain import Chunk


def _chunk(position: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        doc_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        position=position,
        metadata={"page": position + 1},
    )


def test_prompt_contains_question_sources_and_rules() -> None:
    prompt = PromptBuilder().build(
        "What is TCP?",
        [
            _chunk(0, "TCP is a transport protocol."),
            _chunk(1, "TCP provides reliable delivery."),
        ],
    )

    assert "What is TCP?" in prompt
    assert "Answer only using the provided sources" in prompt
    assert "do not know based on the provided documents" in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "TCP is a transport protocol." in prompt
    assert "position=0" in prompt
    assert "page=1" in prompt

