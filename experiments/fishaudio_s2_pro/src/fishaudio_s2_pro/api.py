from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from .artifacts import write_json


def post_tts(
    *,
    base_url: str,
    text: str,
    output_path: str | Path,
    reference_id: str | None = None,
    seed: int | None = 42,
    chunk_length: int = 300,
    max_new_tokens: int = 1024,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text,
        "references": [],
        "reference_id": reference_id,
        "format": output_path.suffix.lstrip(".") or "wav",
        "latency": "normal",
        "max_new_tokens": max_new_tokens,
        "chunk_length": chunk_length,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "temperature": temperature,
        "streaming": False,
        "use_memory_cache": "on" if reference_id else "off",
        "seed": seed,
    }
    started = time.time()
    response = requests.post(
        f"{base_url.rstrip('/')}/v1/tts",
        json=payload,
        timeout=timeout_seconds,
    )
    elapsed = time.time() - started
    if response.status_code != 200:
        error_path = output_path.with_suffix(output_path.suffix + ".error.json")
        write_json(
            error_path,
            {
                "status_code": response.status_code,
                "body": response.text[:4000],
                "payload": {**payload, "text": text[:400]},
                "elapsed_seconds": elapsed,
            },
        )
        raise RuntimeError(f"TTS request failed with HTTP {response.status_code}; see {error_path}")

    output_path.write_bytes(response.content)
    manifest = {
        "output_path": str(output_path),
        "bytes": len(response.content),
        "elapsed_seconds": elapsed,
        "reference_id": reference_id,
        "seed": seed,
        "text_chars": len(text),
    }
    write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest
