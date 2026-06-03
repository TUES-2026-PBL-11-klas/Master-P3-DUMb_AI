from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

DEFAULT_MAC_EMBEDDING_MODEL = "mlx-community/bge-m3-mlx"
DEFAULT_LINUX_EMBEDDING_MODEL_PATH = "models/bge-m3-q8_0.gguf"
DEFAULT_MAC_GENERATION_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
DEFAULT_LINUX_GENERATION_MODEL_PATH = "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
DEFAULT_EMBEDDING_CONTEXT = 8192
DEFAULT_GENERATION_CONTEXT = 2048
DEFAULT_GPU_LAYERS = -1
DEFAULT_MAX_TOKENS = 256


@runtime_checkable
class LlamaCppClient(Protocol):
    """Structural interface for the llama.cpp HTTP embedding client."""

    def embed(self, text: str) -> list[float]:
        """Return a single embedding vector for *text*."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per entry in *texts*."""
        ...


@runtime_checkable
class GenerationClient(Protocol):
    """Structural interface for a text generation model client."""

    def generate(self, prompt: str) -> str:
        """Return a generated answer for *prompt*."""
        ...


class PlatformEmbeddingClient:
    """
    Native embedding client.

    Uses MLX embeddings on macOS and llama.cpp embeddings on Linux. Imports are
    intentionally lazy so tests and unsupported platforms can still import the
    module without having ML packages installed.
    """

    def __init__(
        self,
        *,
        mac_model: str = DEFAULT_MAC_EMBEDDING_MODEL,
        linux_model_path: str | None = None,
        n_ctx: int = DEFAULT_EMBEDDING_CONTEXT,
        n_gpu_layers: int = DEFAULT_GPU_LAYERS,
    ) -> None:
        self._os = platform.system().lower()
        linux_model_path = linux_model_path or _project_path(
            DEFAULT_LINUX_EMBEDDING_MODEL_PATH
        )
        self._backend = self._init_backend(
            mac_model=mac_model,
            linux_model_path=linux_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )

    def _init_backend(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> Any:
        if self._os == "darwin":
            return _MLXEmbeddingBackend(mac_model)
        if self._os == "linux":
            return _LlamaCppEmbeddingBackend(linux_model_path, n_ctx, n_gpu_layers)
        raise OSError(f"Unsupported operating system architecture: {self._os}")

    def embed(self, text: str) -> list[float]:
        return self._backend.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_batch(texts)


class NativeGenerationClient:
    """
    Native text generation client.

    Uses MLX LM on macOS and llama.cpp on Linux.
    """

    def __init__(
        self,
        *,
        mac_model: str = DEFAULT_MAC_GENERATION_MODEL,
        linux_model_path: str | None = None,
        n_ctx: int = DEFAULT_GENERATION_CONTEXT,
        n_gpu_layers: int = DEFAULT_GPU_LAYERS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._os = platform.system().lower()
        self._max_tokens = max_tokens
        linux_model_path = linux_model_path or _project_path(
            DEFAULT_LINUX_GENERATION_MODEL_PATH
        )
        self._backend = self._init_backend(
            mac_model=mac_model,
            linux_model_path=linux_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )

    def _init_backend(
        self,
        *,
        mac_model: str,
        linux_model_path: str,
        n_ctx: int,
        n_gpu_layers: int,
    ) -> Any:
        if self._os == "darwin":
            return _MLXGenerationBackend(mac_model)
        if self._os == "linux":
            return _LlamaCppGenerationBackend(linux_model_path, n_ctx, n_gpu_layers)
        raise OSError(f"Unsupported operating system architecture: {self._os}")

    def generate(self, prompt: str) -> str:
        return self._backend.generate(prompt, max_tokens=self._max_tokens)


class _MLXEmbeddingBackend:
    def __init__(self, model_name: str) -> None:
        from mlx_embeddings import load

        if model_name == "bge-small":
            model_name = "mlx-community/bge-small-en-v1.5-bf16"
        elif model_name == "bge-m3":
            model_name = DEFAULT_MAC_EMBEDDING_MODEL

        model_instance: Any
        tokenizer_instance: Any
        model_instance, tokenizer_instance = load(model_name)
        self._model = model_instance
        self._tokenizer = tokenizer_instance

    def embed(self, text: str) -> list[float]:
        from mlx_embeddings import generate

        output: Any = generate(self._model, self._tokenizer, texts=[text])
        return cast(list[float], output.text_embeds.tolist()[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        from mlx_embeddings import generate

        output: Any = generate(self._model, self._tokenizer, texts=texts)
        return cast(list[list[float]], output.text_embeds.tolist())


class _LlamaCppEmbeddingBackend:
    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int) -> None:
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def embed(self, text: str) -> list[float]:
        payload = self._llm.create_embedding(text)
        return cast(list[float], payload["data"][0]["embedding"])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = self._llm.create_embedding(texts)
        return [cast(list[float], item["embedding"]) for item in payload["data"]]


class _MLXGenerationBackend:
    def __init__(self, model_name: str) -> None:
        from mlx_lm import load

        loaded = load(model_name)
        self._model = loaded[0]
        self._tokenizer = loaded[1]

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        from mlx_lm import generate

        response = generate(self._model, self._tokenizer, prompt, max_tokens=max_tokens)
        return str(response)


class _LlamaCppGenerationBackend:
    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int) -> None:
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(self, prompt: str, *, max_tokens: int) -> str:
        formatted = f"<|user|>\n{prompt}\n<|assistant|>"
        output = self._llm(formatted, max_tokens=max_tokens, stream=False)
        decoded = cast(dict[str, Any], output)
        return str(decoded["choices"][0]["text"])


def _project_path(relative_path: str) -> str:
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / relative_path)
