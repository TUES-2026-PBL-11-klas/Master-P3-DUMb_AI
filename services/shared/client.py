from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, cast


class PlatformEmbeddingClient:
    def __init__(
        self,
        mac_model: str = "mlx-community/bge-m3-mlx",
        linux_model_path: str | None = None,
        n_ctx: int = 8192,
        n_gpu_layers: int = -1,
    ) -> None:
        self._os = platform.system().lower()

        if linux_model_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            linux_model_path = str(project_root / "models" / "bge-m3-q8_0.gguf")

        self._backend = self._init_backend(
            mac_model, linux_model_path, n_ctx, n_gpu_layers
        )

    def _init_backend(
        self, mac_model: str, linux_model_path: str, n_ctx: int, n_gpu_layers: int
    ) -> Any:
        if self._os == "darwin":
            return _MLXBackend(mac_model)
        elif self._os == "linux":
            return _LlamaCppBackend(linux_model_path, n_ctx, n_gpu_layers)
        raise OSError(f"Unsupported operating system architecture: {self._os}")

    def embed(self, text: str) -> list[float]:
        return self._backend.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._backend.embed_batch(texts)


class _MLXBackend:
    """
    High-performance embedding generation for BGE models on Apple Silicon.
    Utilizes mlx-embeddings functional generate() endpoint.
    """

    def __init__(self, model_name: str) -> None:
        from mlx_embeddings import load

        if model_name == "bge-small":
            model_name = "mlx-community/bge-small-en-v1.5-bf16"
        elif model_name == "bge-m3":
            model_name = "mlx-community/bge-m3-mlx"

        model_instance: Any
        tokenizer_instance: Any
        model_instance, tokenizer_instance = load(model_name)

        self._model: Any = model_instance
        self._tokenizer: Any = tokenizer_instance

    def embed(self, text: str) -> list[float]:
        from mlx_embeddings import generate

        output: Any = generate(self._model, self._tokenizer, texts=[text])
        return cast(list[float], output.text_embeds.tolist()[0])

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        from mlx_embeddings import generate

        output: Any = generate(self._model, self._tokenizer, texts=texts)
        return cast(list[list[float]], output.text_embeds.tolist())


class _LlamaCppBackend:
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
