#!/usr/bin/env python3
"""Launch full Mandarin Xiaoyuzhou ASR as a detached Colab VM process.

Use this instead of running the full transcription directly through `colab exec`,
because Colab CLI/WebSocket often times out before long cells return.

Run:
    colab exec -s qwen-asr-a100 --file scripts/colab_launch_full_mandarin.py

Then monitor:
    colab exec -s qwen-asr-a100 --file scripts/colab_status_full_mandarin.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PUBLIC_REPO_URL = os.environ.get("PUBLIC_REPO_URL", "https://github.com/liuwen/qwen-asr-eval.git")
PUBLIC_REPO_BRANCH = os.environ.get("PUBLIC_REPO_BRANCH", "main")
REPO_DIR = Path(os.environ.get("REPO_DIR", "/content/qwen-asr-eval"))
DRIVE_ROOT = Path(os.environ.get("ASR_EVAL_DRIVE_ROOT", "/content/drive/MyDrive/asr"))
RUN_ID = os.environ.get("FULL_MANDARIN_RUN_ID", "full_mandarin_long_sushi")
RAW_AUDIO = Path(os.environ.get("FULL_MANDARIN_AUDIO", str(DRIVE_ROOT / "audio/raw_xiaoyuzhou/mandarin_long_sushi.audio")))
OUT_DIR = Path(os.environ.get("FULL_MANDARIN_OUT_DIR", str(DRIVE_ROOT / f"qwen-asr-eval/runs/{RUN_ID}/outputs")))
WORK_DIR = Path(os.environ.get("FULL_MANDARIN_WORK_DIR", f"/content/asr-eval-work/{RUN_ID}"))
LOG_DIR = Path(os.environ.get("FULL_MANDARIN_LOG_DIR", str(DRIVE_ROOT / "qwen-asr-eval/logs")))
HF_CACHE_ROOT = Path(os.environ.get("ASR_EVAL_HF_CACHE", str(DRIVE_ROOT / "hf_cache")))
CHUNK_SECONDS = os.environ.get("FULL_MANDARIN_CHUNK_SECONDS", "300")
OVERLAP_SECONDS = os.environ.get("FULL_MANDARIN_OVERLAP_SECONDS", "5")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check, text=True)


def main() -> None:
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Drive is not mounted. Run: colab drivemount -s qwen-asr-a100 /content/drive")

    if REPO_DIR.exists():
        run(["git", "-C", str(REPO_DIR), "fetch", "origin", PUBLIC_REPO_BRANCH], check=False)
        run(["git", "-C", str(REPO_DIR), "checkout", PUBLIC_REPO_BRANCH], check=False)
        run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", "--depth", "1", "--branch", PUBLIC_REPO_BRANCH, PUBLIC_REPO_URL, str(REPO_DIR)])

    # Fast if already installed; makes launcher robust for fresh VMs.
    run([sys.executable, "-m", "pip", "install", "-q", "-U", "qwen-asr", "huggingface_hub[hf_transfer]", "pandas", "tqdm", "rapidfuzz", "soundfile", "librosa", "pydub", "jiwer", "opencc-python-reimplemented"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR), "--no-deps"])

    if not RAW_AUDIO.exists():
        raise FileNotFoundError(f"Raw audio not found: {RAW_AUDIO}. Run batch/smoke downloader first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{RUN_ID}.log"
    pid_path = LOG_DIR / f"{RUN_ID}.pid"
    status_path = LOG_DIR / f"{RUN_ID}.launch.json"
    shell_path = Path(f"/content/{RUN_ID}.sh")

    shell = f'''#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME={HF_CACHE_ROOT}
export HF_HUB_CACHE={HF_CACHE_ROOT / 'hub'}
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH={REPO_DIR / 'src'}:${{PYTHONPATH:-}}
cd {REPO_DIR}
echo "[start] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "raw_audio={RAW_AUDIO}"
echo "out_dir={OUT_DIR}"
echo "work_dir={WORK_DIR}"
{sys.executable} {REPO_DIR / 'scripts/run_eval.py'} \
  {RAW_AUDIO} \
  --out-dir {OUT_DIR} \
  --work-dir {WORK_DIR} \
  --use-case zh \
  --chunk-seconds {CHUNK_SECONDS} \
  --overlap-seconds {OVERLAP_SECONDS} \
  --batch-size 1 \
  --max-inference-batch-size 1 \
  --max-new-tokens 4096
echo "[done] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
'''
    shell_path.write_text(shell, encoding="utf-8")
    shell_path.chmod(0o755)

    log_file = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(["bash", str(shell_path)], stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    pid_path.write_text(str(proc.pid) + "\n", encoding="utf-8")
    status = {
        "run_id": RUN_ID,
        "pid": proc.pid,
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "raw_audio": str(RAW_AUDIO),
        "out_dir": str(OUT_DIR),
        "work_dir": str(WORK_DIR),
        "log_path": str(log_path),
        "pid_path": str(pid_path),
        "shell_path": str(shell_path),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print("Launched detached. colab exec can now return; job keeps running while runtime lives.")


if __name__ == "__main__":
    main()
