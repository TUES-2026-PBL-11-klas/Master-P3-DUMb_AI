from __future__ import annotations

import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

from services.shared.client import PlatformEmbeddingClient


class DummyMLXOutput:
    def __init__(self, dimensions: int = 1024, batch_size: int = 1) -> None:
        import numpy as np

        self.text_embeds = np.random.randn(batch_size, dimensions)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Globally intercepts deep learning backends with clean MagicMock wrappers."""
    mock_load_fn = MagicMock()
    mock_generate_fn = MagicMock()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_load_fn.return_value = (mock_model, mock_tokenizer)

    mock_generate_fn.return_value = DummyMLXOutput(dimensions=1024, batch_size=2)

    with patch.dict(
        sys.modules, {"mlx_embeddings": MagicMock(), "llama_cpp": MagicMock()}
    ):
        import llama_cpp
        import mlx_embeddings

        mlx_embeddings.load = mock_load_fn
        mlx_embeddings.generate = mock_generate_fn

        yield {
            "mlx_embeddings": mlx_embeddings,
            "llama_cpp": llama_cpp,
            "generate_fn": mock_generate_fn,
        }


def test_ingestion_pipeline_end_to_end(mock_dependencies):
    """
    Integration Test: Assures IngestionService successfully drives a document
    through parsing, event notification, and your PlatformEmbeddingClient engine.
    """
    from datetime import datetime, timezone

    from services.ingestion.observers.embedding import EmbeddingObserver
    from services.ingestion.service import IngestionService
    from services.shared.domain import Document, DocumentStatus, IngestionStatus

    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-m3")

        mock_parser = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_for_filename.return_value = mock_parser

        mock_chunk_1 = MagicMock()
        mock_chunk_1.text = "The horse raced past the barn fell."
        mock_chunk_2 = MagicMock()
        mock_chunk_2.text = "Isn't it nice to be inside such a fancy computer?"

        test_doc_id = uuid.uuid4()
        parsed_document = Document(
            id=test_doc_id,
            user_id=uuid.UUID(int=0),
            content="Full combined document text...",
            filename="sample.md",
            uploaded_at=datetime.now(tz=timezone.utc),
            status=DocumentStatus.PARSED,
        )
        mock_parser.parse.return_value = parsed_document

        embedding_observer = EmbeddingObserver(client=client, batch_size=2)

        ingestion_service = IngestionService(registry=mock_registry)
        ingestion_service.subscribe(embedding_observer)

        def inject_chunks_side_effect(event):
            event.chunks = [mock_chunk_1, mock_chunk_2]

        embedding_observer.on_ingest = MagicMock(
            side_effect=lambda event: [
                inject_chunks_side_effect(event),
                EmbeddingObserver.on_ingest(embedding_observer, event),
            ]
        )

        target_user_id = uuid.uuid4()
        final_event = ingestion_service.ingest(
            filename="sample.md",
            raw=b"Dummy raw bytes payload below 5MB limit",
            user_id=target_user_id,
        )

        assert final_event.document.user_id == target_user_id

        mock_dependencies["generate_fn"].assert_called_once_with(
            client._backend._model,
            client._backend._tokenizer,
            texts=[mock_chunk_1.text, mock_chunk_2.text],
        )

        assert isinstance(mock_chunk_1.embedding, list)
        assert len(mock_chunk_1.embedding) == 1024

        assert final_event.status == IngestionStatus.COMPLETED
        assert final_event.document.status == DocumentStatus.READY
