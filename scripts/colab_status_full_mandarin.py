#!/usr/bin/env python3
"""Status check for detached full Mandarin ASR job."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("ASR_EVAL_DRIVE_ROOT", "/content/drive/MyDrive/asr"))
RUN_ID = os.environ.get("FULL_MANDARIN_RUN_ID", "full_mandarin_long_sushi")
OUT_DIR = Path(os.environ.get("FULL_MANDARIN_OUT_DIR", str(DRIVE_ROOT / f"qwen-asr-eval/runs/{RUN_ID}/outputs")))
LOG_DIR = Path(os.environ.get("FULL_MANDARIN_LOG_DIR", str(DRIVE_ROOT / "qwen-asr-eval/logs")))
LOG_PATH = LOG_DIR / f"{RUN_ID}.log"
PID_PATH = LOG_DIR / f"{RUN_ID}.pid"


def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def read_jsonl_count(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    rows = 0
    chars = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows += 1
                try:
                    chars += len(str(json.loads(line).get("text") or ""))
                except Exception:
                    pass
    return rows, chars

print("=== full mandarin status ===")
print("run_id", RUN_ID)
print("out_dir", OUT_DIR)
print("log", LOG_PATH)

pid = None
if PID_PATH.exists():
    try:
        pid = int(PID_PATH.read_text().strip())
    except Exception:
        pid = None
print("pid", pid, "alive", pid_alive(pid) if pid else None)

qwen = OUT_DIR / "qwen_chunks.jsonl"
rows, chars = read_jsonl_count(qwen)
print("qwen_jsonl_exists", qwen.exists(), "rows", rows, "text_chars", chars)
for name in ["chunks_manifest.jsonl", "qwen_chunks.csv", "qwen_transcript_chunked.md", "qwen_transcript_chunked.txt"]:
    p = OUT_DIR / name
    print(name, "exists", p.exists(), "size", p.stat().st_size if p.exists() else None)

print("\n=== gpu ===")
r = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"])
print((r.stdout or r.stderr).strip())
print("\n=== gpu processes ===")
r = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
print((r.stdout or "<none>").strip())

print("\n=== log tail ===")
if LOG_PATH.exists():
    r = run(["tail", "-80", str(LOG_PATH)])
    print(r.stdout)
else:
    print("missing log")
