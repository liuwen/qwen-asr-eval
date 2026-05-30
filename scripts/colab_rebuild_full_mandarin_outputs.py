#!/usr/bin/env python3
"""Rebuild transcript/csv artifacts from full Mandarin qwen_chunks.jsonl.

Use when the long ASR finished enough to write qwen_chunks.jsonl but the final
CSV/Markdown/Text artifacts are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path("/content/qwen-asr-eval")
SRC = REPO_DIR / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asr_eval.reporting import save_transcript  # noqa: E402

OUT_DIR = Path("/content/drive/MyDrive/asr/qwen-asr-eval/runs/full_mandarin_long_sushi/outputs")
QWEN_JSONL = OUT_DIR / "qwen_chunks.jsonl"

if not QWEN_JSONL.exists():
    raise FileNotFoundError(QWEN_JSONL)

rows = []
with QWEN_JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

if not rows:
    raise RuntimeError(f"No rows in {QWEN_JSONL}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.drop(columns=["time_stamps"], errors="ignore").to_csv(OUT_DIR / "qwen_chunks.csv", index=False)
saved = save_transcript(OUT_DIR, "qwen", "Qwen3-ASR full Mandarin transcript", rows)

print(json.dumps({
    "ok": True,
    "out_dir": str(OUT_DIR),
    "rows": len(rows),
    "text_chars": sum(len(str(r.get("text") or "")) for r in rows),
    "first_range": [rows[0].get("start_ts"), rows[0].get("end_ts")],
    "last_range": [rows[-1].get("start_ts"), rows[-1].get("end_ts")],
    "csv": str(OUT_DIR / "qwen_chunks.csv"),
    "md": str(saved["md_path"]),
    "txt": str(saved["txt_path"]),
}, ensure_ascii=False, indent=2))
