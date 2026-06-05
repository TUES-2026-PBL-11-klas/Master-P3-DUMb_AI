from __future__ import annotations

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from services.shared.client import PlatformEmbeddingClient


@pytest.fixture
def mock_mlx_embeddings() -> Generator[MagicMock, None, None]:
    """Mocks the mlx-embeddings module to allow test execution on any host OS."""
    mock_module = MagicMock()
    mock_module.load.return_value = (MagicMock(), MagicMock())
    with patch.dict(sys.modules, {"mlx_embeddings": mock_module}):
        yield mock_module


def test_mac_backend_initialization_on_darwin(mock_mlx_embeddings: MagicMock) -> None:
    """Verifies that the client targets the MLX suite when running on macOS."""
    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-small")

        assert client._os == "darwin"
        mock_mlx_embeddings.load.assert_called_once_with(
            "mlx-community/bge-small-en-v1.5-bf16"
        )


def test_mac_backend_embed_returns_list_of_floats(
    mock_mlx_embeddings: MagicMock,
) -> None:
    """Ensures multi-dimensional arrays from the MLX runtime convert into plain float lists."""
    mock_output = MagicMock()
    mock_output.text_embeds.tolist.return_value = [[0.1, -0.2, 0.35]]
    mock_mlx_embeddings.generate.return_value = mock_output

    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-small")
        vector = client.embed("Test query string")

        mock_mlx_embeddings.generate.assert_called_with(
            client._backend._model,
            client._backend._tokenizer,
            texts=["Test query string"],
        )
        assert isinstance(vector, list)
        assert vector == [0.1, -0.2, 0.35]


def test_mac_backend_embed_batch(mock_mlx_embeddings: MagicMock) -> None:
    """Ensures batch processing retains structure and maps to a double-nested float array list."""
    mock_output = MagicMock()
    mock_output.text_embeds.tolist.return_value = [[0.1, -0.2], [0.4, 0.9]]
    mock_mlx_embeddings.generate.return_value = mock_output

    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-small")
        vectors = client.embed_batch(["text one", "text two"])

        mock_mlx_embeddings.generate.assert_called_with(
            client._backend._model,
            client._backend._tokenizer,
            texts=["text one", "text two"],
        )
        assert vectors == [[0.1, -0.2], [0.4, 0.9]]


def test_unsupported_platform_raises_error() -> None:
    """Validates fallback validation branches throw an OSError when meeting invalid platforms."""
    with patch("platform.system", return_value="Windows"):
        with pytest.raises(OSError, match="Unsupported operating system architecture"):
            PlatformEmbeddingClient()
