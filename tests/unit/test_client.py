from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from services.shared.client import PlatformEmbeddingClient


@pytest.fixture
def mock_mlx_embedding_models():
    """Mocks the mlx-embedding-models module to allow test execution on any host OS."""
    with patch.dict(sys.modules, {"mlx_embedding_models.embedding": MagicMock()}):
        import mlx_embedding_models.embedding as mock_emb

        yield mock_emb


def test_mac_backend_initialization_on_darwin(mock_mlx_embedding_models):
    """Verifies that the client targets the MLX suite when running on macOS."""
    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-small")

        assert client._os == "darwin"
        mock_mlx_embedding_models.EmbeddingModel.from_registry.assert_called_once_with(
            "bge-small"
        )


def test_mac_backend_embed_returns_list_of_floats(mock_mlx_embedding_models):
    """Ensures multi-dimensional arrays from the MLX runtime convert into plain float lists."""
    mock_model_instance = MagicMock()

    mock_array = MagicMock()
    mock_array.tolist.return_value = [[0.1, -0.2, 0.35]]
    mock_model_instance.encode.return_value = mock_array

    mock_mlx_embedding_models.EmbeddingModel.from_registry.return_value = (
        mock_model_instance
    )

    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-small")
        vector = client.embed("Test query string")

        mock_model_instance.encode.assert_called_with(["Test query string"])
        assert isinstance(vector, list)
        assert vector == [0.1, -0.2, 0.35]


def test_mac_backend_embed_batch(mock_mlx_embedding_models):
    """Ensures batch processing retains structure and maps to a double-nested float array list."""
    mock_model_instance = MagicMock()
    mock_array = MagicMock()
    mock_array.tolist.return_value = [[0.1, -0.2], [0.4, 0.9]]
    mock_model_instance.encode.return_value = mock_array

    mock_mlx_embedding_models.EmbeddingModel.from_registry.return_value = (
        mock_model_instance
    )

    with patch("platform.system", return_value="Darwin"):
        client = PlatformEmbeddingClient(mac_model="bge-small")
        vectors = client.embed_batch(["text one", "text two"])

        mock_model_instance.encode.assert_called_with(["text one", "text two"])
        assert vectors == [[0.1, -0.2], [0.4, 0.9]]


def test_unsupported_platform_raises_error():
    """Validates fallback validation branches throw an OSError when meeting invalid platforms."""
    with patch("platform.system", return_value="Windows"):
        with pytest.raises(OSError, match="Unsupported operating system architecture"):
            PlatformEmbeddingClient()
