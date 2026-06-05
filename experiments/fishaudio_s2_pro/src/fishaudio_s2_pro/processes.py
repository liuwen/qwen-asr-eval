from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .paths import ExperimentPaths


@dataclass(frozen=True)
class ManagedProcess:
    name: str
    pid: int
    command: list[str]
    log_path: Path


def _start_process(
    *,
    name: str,
    command: list[str],
    cwd: str | Path,
    log_path: str | Path,
    env: dict[str, str] | None = None,
    display_command: list[str] | None = None,
) -> ManagedProcess:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    proc_env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("ab") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=Path(cwd),
            env=proc_env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return ManagedProcess(
        name=name,
        pid=proc.pid,
        command=display_command or command,
        log_path=log_path,
    )


def wait_for_http(url: str, *, timeout_seconds: int = 600, interval_seconds: float = 2.0) -> bool:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception as exc:
            last_error = exc
            time.sleep(interval_seconds)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error!r}")


def start_api_server(
    paths: ExperimentPaths,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    compile_model: bool = False,
    half: bool = False,
    api_key: str | None = None,
) -> ManagedProcess:
    command = [
        "uv",
        "run",
        "python",
        "tools/api_server.py",
        "--listen",
        f"{host}:{port}",
        "--llama-checkpoint-path",
        str(paths.checkpoint_dir),
        "--decoder-checkpoint-path",
        str(paths.checkpoint_dir / "codec.pth"),
        "--decoder-config-name",
        "modded_dac_vq",
        "--device",
        "cuda",
        "--workers",
        "1",
    ]
    if compile_model:
        command.append("--compile")
    if half:
        command.append("--half")
    if api_key:
        command.extend(["--api-key", api_key])
    display = ["<redacted>" if part == api_key else part for part in command]
    return _start_process(
        name="fish-api",
        command=command,
        cwd=paths.fish_repo,
        log_path=paths.logs_dir / "fish_api.log",
        env=paths.hf_env(),
        display_command=display,
    )


def start_gradio_webui(
    paths: ExperimentPaths,
    *,
    compile_model: bool = False,
    half: bool = False,
    theme: str = "light",
) -> ManagedProcess:
    command = [
        "uv",
        "run",
        "python",
        "tools/run_webui.py",
        "--llama-checkpoint-path",
        str(paths.checkpoint_dir),
        "--decoder-checkpoint-path",
        str(paths.checkpoint_dir / "codec.pth"),
        "--decoder-config-name",
        "modded_dac_vq",
        "--device",
        "cuda",
        "--theme",
        theme,
    ]
    if compile_model:
        command.append("--compile")
    if half:
        command.append("--half")
    return _start_process(
        name="fish-gradio",
        command=command,
        cwd=paths.fish_repo,
        log_path=paths.logs_dir / "fish_gradio.log",
        env=paths.hf_env(),
    )


def start_cloudflared_named_tunnel(
    *,
    token: str,
    cwd: str | Path,
    log_path: str | Path,
) -> ManagedProcess:
    if not token:
        raise ValueError("Cloudflare tunnel token is required.")
    command = ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", token]
    display = ["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", "REDACTED"]
    return _start_process(
        name="cloudflared",
        command=command,
        cwd=cwd,
        log_path=log_path,
        display_command=display,
    )


def stop_process(process: ManagedProcess, *, timeout_seconds: int = 30) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.5)

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
