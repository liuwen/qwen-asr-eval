from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


def run_command(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.run(
        command,
        cwd=Path(cwd) if cwd else None,
        env=proc_env,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def read_colab_secret(name: str, *, required: bool = True) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not read Colab Secret {name}: {exc!r}") from exc
        return None
    if required and not value:
        raise RuntimeError(f"Missing Colab Secret: {name}")
    return value


def gpu_inventory() -> list[dict[str, int | str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run_command(command, capture=True)
    except Exception:
        return []

    gpus: list[dict[str, int | str]] = []
    for line in (result.stdout or "").splitlines():
        if not line.strip():
            continue
        name, memory_mb = [part.strip() for part in line.split(",", 1)]
        gpus.append({"name": name, "memory_mb": int(memory_mb)})
    return gpus


def require_minimum_vram(minimum_gb: int = 24) -> list[dict[str, int | str]]:
    gpus = gpu_inventory()
    if not gpus:
        raise RuntimeError("No NVIDIA GPU detected. Use a Colab A100/H100 runtime.")
    best_mb = max(int(gpu["memory_mb"]) for gpu in gpus)
    if best_mb < minimum_gb * 1024:
        raise RuntimeError(
            f"Fish Audio S2 Pro needs at least {minimum_gb} GB VRAM; "
            f"largest detected GPU has {best_mb / 1024:.1f} GB."
        )
    return gpus


def _cuda_version_from_nvidia_smi() -> tuple[int, int] | None:
    try:
        result = run_command(["nvidia-smi"], capture=True)
    except Exception:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+)\.([0-9]+)", result.stdout or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_fish_uv_extra(value: str = "auto") -> str:
    if value != "auto":
        if value not in {"cu126", "cu128", "cu129", "cpu"}:
            raise ValueError("FISH_UV_EXTRA must be auto, cu126, cu128, cu129, or cpu")
        return value

    cuda_version = _cuda_version_from_nvidia_smi()
    if cuda_version is None:
        return "cu126"
    major, minor = cuda_version
    if (major, minor) >= (12, 9):
        return "cu129"
    if (major, minor) >= (12, 8):
        return "cu128"
    return "cu126"


def assert_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "This experiment targets Python 3.12. Use a Colab Python 3.12 runtime "
            f"or run Fish Speech through `uv --python 3.12`; current kernel is {sys.version}."
        )
