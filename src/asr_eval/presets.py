from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RuntimePreset:
    name: str
    description: str
    chunk_seconds: int
    overlap_seconds: int
    qwen_batch_size: int
    qwen_max_inference_batch_size: int
    qwen_max_new_tokens: int
    whisper_model_size: str = "large-v3"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RUNTIME_PRESETS: dict[str, RuntimePreset] = {
    "T4": RuntimePreset(
        name="T4",
        description="Conservative Colab T4 preset. Lower VRAM; prioritize reliability.",
        chunk_seconds=180,
        overlap_seconds=5,
        qwen_batch_size=1,
        qwen_max_inference_batch_size=1,
        qwen_max_new_tokens=2048,
        whisper_model_size="medium",
    ),
    "A100": RuntimePreset(
        name="A100",
        description="Balanced A100 preset. Good default for long podcasts.",
        chunk_seconds=300,
        overlap_seconds=5,
        qwen_batch_size=4,
        qwen_max_inference_batch_size=4,
        qwen_max_new_tokens=4096,
        whisper_model_size="large-v3",
    ),
    "H100": RuntimePreset(
        name="H100",
        description="Higher-throughput H100 preset. Increase if stable.",
        chunk_seconds=300,
        overlap_seconds=5,
        qwen_batch_size=8,
        qwen_max_inference_batch_size=8,
        qwen_max_new_tokens=4096,
        whisper_model_size="large-v3",
    ),
}


def get_runtime_preset(name: str) -> RuntimePreset:
    key = str(name or "A100").upper()
    if key not in RUNTIME_PRESETS:
        raise KeyError(f"Unknown runtime preset {name!r}. Choose one of {sorted(RUNTIME_PRESETS)}")
    return RUNTIME_PRESETS[key]


def preset_table() -> list[dict[str, Any]]:
    return [preset.to_dict() for preset in RUNTIME_PRESETS.values()]
