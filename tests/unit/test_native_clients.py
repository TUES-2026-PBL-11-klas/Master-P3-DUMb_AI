from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from services.shared.client import NativeGenerationClient, PlatformEmbeddingClient


class _FakeEmbeds:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = values

    def tolist(self) -> list[list[float]]:
        return self._values


class _FakeEmbeddingOutput:
    def __init__(self, values: list[list[float]]) -> None:
        self.text_embeds = _FakeEmbeds(values)


def test_platform_embedding_client_uses_mlx_on_macos() -> None:
    mlx_embeddings = MagicMock()
    mlx_embeddings.load.return_value = ("model", "tokenizer")
    mlx_embeddings.generate.return_value = _FakeEmbeddingOutput([[0.1, 0.2]])

    with (
        patch("platform.system", return_value="Darwin"),
        patch.dict(sys.modules, {"mlx_embeddings": mlx_embeddings}),
    ):
        client = PlatformEmbeddingClient(mac_model="bge-m3")

        assert client.embed("hello") == [0.1, 0.2]

    mlx_embeddings.load.assert_called_once_with("mlx-community/bge-m3-mlx")
    mlx_embeddings.generate.assert_called_once_with(
        "model",
        "tokenizer",
        texts=["hello"],
    )


def test_platform_embedding_client_uses_llama_cpp_on_linux() -> None:
    fake_llm = MagicMock()
    fake_llm.create_embedding.return_value = {
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]
    }
    llama_cpp = MagicMock()
    llama_cpp.Llama.return_value = fake_llm

    with (
        patch("platform.system", return_value="Linux"),
        patch.dict(sys.modules, {"llama_cpp": llama_cpp}),
    ):
        client = PlatformEmbeddingClient(
            linux_model_path="models/embed.gguf",
            n_ctx=4096,
            n_gpu_layers=5,
        )

        assert client.embed_batch(["one", "two"]) == [[0.1, 0.2], [0.3, 0.4]]

    llama_cpp.Llama.assert_called_once_with(
        model_path="models/embed.gguf",
        embedding=True,
        n_ctx=4096,
        n_gpu_layers=5,
        verbose=False,
    )
    fake_llm.create_embedding.assert_called_once_with(["one", "two"])


def test_native_generation_client_uses_mlx_on_macos() -> None:
    mlx_lm = MagicMock()
    mlx_lm.load.return_value = ("model", "tokenizer")
    mlx_lm.generate.return_value = "Generated answer."

    with (
        patch("platform.system", return_value="Darwin"),
        patch.dict(sys.modules, {"mlx_lm": mlx_lm}),
    ):
        client = NativeGenerationClient(mac_model="mlx-model", max_tokens=42)

        assert client.generate("prompt") == "Generated answer."

    mlx_lm.load.assert_called_once_with("mlx-model")
    mlx_lm.generate.assert_called_once_with(
        "model",
        "tokenizer",
        "prompt",
        max_tokens=42,
    )


def test_native_generation_client_uses_llama_cpp_on_linux() -> None:
    fake_llm = MagicMock()
    fake_llm.return_value = {"choices": [{"text": "Generated answer."}]}
    llama_cpp = MagicMock()
    llama_cpp.Llama.return_value = fake_llm

    with (
        patch("platform.system", return_value="Linux"),
        patch.dict(sys.modules, {"llama_cpp": llama_cpp}),
    ):
        client = NativeGenerationClient(
            linux_model_path="models/generate.gguf",
            n_ctx=2048,
            n_gpu_layers=7,
            max_tokens=64,
        )

        assert client.generate("prompt") == "Generated answer."

    llama_cpp.Llama.assert_called_once_with(
        model_path="models/generate.gguf",
        n_ctx=2048,
        n_gpu_layers=7,
        verbose=False,
    )
    fake_llm.assert_called_once_with(
        "<|user|>\nprompt\n<|assistant|>",
        max_tokens=64,
        stream=False,
    )


def test_native_clients_reject_unsupported_platform() -> None:
    with patch("platform.system", return_value="Windows"):
        with pytest.raises(OSError, match="Unsupported operating system"):
            PlatformEmbeddingClient()
        with pytest.raises(OSError, match="Unsupported operating system"):
            NativeGenerationClient()
