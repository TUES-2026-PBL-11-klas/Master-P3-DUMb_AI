from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, cast


def run_native_inference(prompt: str, max_tokens: int = 256) -> str:
    current_os = platform.system().lower()

    if current_os == "darwin":
        from mlx_lm import generate, load

        loaded = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")
        model, tokenizer = loaded[0], loaded[1]

        response = generate(model, tokenizer, prompt, max_tokens=max_tokens)
        return str(response)

    elif current_os == "linux":
        from llama_cpp import Llama

        project_root = Path(__file__).resolve().parent.parent.parent
        resolved_model_path = (
            project_root / "models" / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        )

        llm = Llama(
            model_path=str(resolved_model_path),
            n_ctx=2048,
            n_gpu_layers=-1,
        )
        formatted = f"<|user|>\n{prompt}\n<|assistant|>"

        output = llm(formatted, max_tokens=max_tokens, stream=False)

        dict_output = cast(Any, output)
        return str(dict_output["choices"][0]["text"])

    return f"Unsupported platform: {current_os}"
