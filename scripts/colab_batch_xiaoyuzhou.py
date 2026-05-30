#!/usr/bin/env python3
"""Batch Qwen3-ASR sample workflow for Colab CLI.

This is the practical eval path after the smoke test. It downloads a small eval set,
extracts the first N seconds of each episode, loads Qwen once, transcribes all samples,
and writes durable outputs to Google Drive.

Run after Drive is mounted:
    colab exec -s qwen-asr-a100 --file scripts/colab_batch_xiaoyuzhou.py

Default output:
    /content/drive/MyDrive/asr/qwen-asr-eval/runs/batch_<timestamp>/outputs/<sample_id>/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PUBLIC_REPO_URL = os.environ.get("PUBLIC_REPO_URL", "https://github.com/YOUR_GITHUB_USERNAME/qwen-asr-eval.git")
PUBLIC_REPO_BRANCH = os.environ.get("PUBLIC_REPO_BRANCH", "main")
REPO_DIR = Path(os.environ.get("REPO_DIR", "/content/qwen-asr-eval"))
DRIVE_ROOT = Path(os.environ.get("ASR_EVAL_DRIVE_ROOT", "/content/drive/MyDrive/asr"))
AUDIO_ROOT = Path(os.environ.get("ASR_EVAL_AUDIO_ROOT", str(DRIVE_ROOT / "audio")))
RUNS_ROOT = Path(os.environ.get("ASR_EVAL_RUNS_ROOT", str(DRIVE_ROOT / "qwen-asr-eval" / "runs")))
HF_CACHE_ROOT = Path(os.environ.get("ASR_EVAL_HF_CACHE", str(DRIVE_ROOT / "hf_cache")))
TMP_ROOT = Path(os.environ.get("ASR_EVAL_TMP_ROOT", "/content/asr-eval-work"))
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
SAMPLE_SECONDS = int(os.environ.get("BATCH_SAMPLE_SECONDS", "90"))
CHUNK_SECONDS = int(os.environ.get("BATCH_CHUNK_SECONDS", str(SAMPLE_SECONDS)))
MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "2048"))
MAX_INFERENCE_BATCH_SIZE = int(os.environ.get("QWEN_MAX_INFERENCE_BATCH_SIZE", "1"))
QWEN_BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "1"))

DEFAULT_BATCH_IDS = [
    "cantonese_long_xiaofangjian_ep162",
    "mandarin_long_sushi",
    "english_clear_aee_2618",
    "mixed_zh_en_chinglish_rambles_places",
    "mixed_zh_en_wujimacha_22e01",
]
BATCH_AUDIO_IDS = [
    x.strip()
    for x in os.environ.get("BATCH_AUDIO_IDS", ",".join(DEFAULT_BATCH_IDS)).split(",")
    if x.strip()
]


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check, text=True)


def require_drive() -> None:
    my_drive = Path("/content/drive/MyDrive")
    if my_drive.exists():
        print("Drive mounted:", my_drive)
        return
    raise RuntimeError(
        "Google Drive is not mounted. Run once and authorize in browser:\n"
        "  colab drivemount -s qwen-asr-a100 /content/drive"
    )


def configure_cache() -> None:
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_ROOT / "transformers"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print("HF_HOME:", os.environ["HF_HOME"])


def ensure_repo() -> None:
    if REPO_DIR.exists():
        run(["git", "-C", str(REPO_DIR), "fetch", "origin", PUBLIC_REPO_BRANCH], check=False)
        run(["git", "-C", str(REPO_DIR), "checkout", PUBLIC_REPO_BRANCH], check=False)
        run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", "--depth", "1", "--branch", PUBLIC_REPO_BRANCH, PUBLIC_REPO_URL, str(REPO_DIR)])
    src = str(REPO_DIR / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def install_deps() -> None:
    run(["apt-get", "-qq", "update"])
    run(["apt-get", "-qq", "install", "-y", "ffmpeg", "jq"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "qwen-asr",
            "huggingface_hub[hf_transfer]",
            "feedparser",
            "requests",
            "tqdm",
            "pandas",
            "rapidfuzz",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR), "--no-deps"])


def language_for_item(item: dict[str, Any]) -> str | None:
    hint = item.get("language_hint")
    if hint == "English":
        return "English"
    if hint == "Chinese":
        return "Chinese"
    # Cantonese and mixed/code-switching are left auto to avoid forcing the wrong decoder hint.
    return None


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
    print("=== Qwen3-ASR batch Xiaoyuzhou sample eval ===")
    print("run_id:", run_id)
    print("ids:", BATCH_AUDIO_IDS)
    print("sample seconds:", SAMPLE_SECONDS)
    print("runs root:", RUNS_ROOT)

    require_drive()
    configure_cache()
    ensure_repo()
    install_deps()

    from asr_eval.audio import chunk_wav, ffprobe_duration, fmt_ts, write_jsonl
    from asr_eval.qwen_runner import load_qwen_model, transcribe_chunks
    from asr_eval.reporting import save_transcript
    from asr_eval.xiaoyuzhou import download_file, resolve_xiaoyuzhou_audio_url, select_manifest_items

    items = select_manifest_items(BATCH_AUDIO_IDS)
    batch_root = RUNS_ROOT / run_id
    batch_out = batch_root / "outputs"
    work_root = TMP_ROOT / run_id
    raw_root = AUDIO_ROOT / "raw_xiaoyuzhou"
    sample_root = AUDIO_ROOT / "batch_samples_16k_mono" / f"first_{SAMPLE_SECONDS}s"
    for p in [batch_out, work_root, raw_root, sample_root]:
        p.mkdir(parents=True, exist_ok=True)

    prepared: list[dict[str, Any]] = []
    for item in items:
        print(f"\n=== prepare {item['id']} ===")
        audio_url = resolve_xiaoyuzhou_audio_url(item["episode_url"])
        raw_path = raw_root / f"{item['id']}.audio"
        download_file(audio_url, raw_path)
        sample_wav = sample_root / f"{item['id']}.first_{SAMPLE_SECONDS}s.16k_mono.wav"
        if sample_wav.exists() and sample_wav.stat().st_size > 0:
            print("exists:", sample_wav)
        else:
            run(
                [
                    "ffmpeg",
                    "-y",
                    "-t",
                    str(SAMPLE_SECONDS),
                    "-i",
                    str(raw_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(sample_wav),
                ]
            )
        duration = ffprobe_duration(sample_wav)
        print("sample:", sample_wav, fmt_ts(duration))
        prepared.append({**item, "resolved_audio_url": audio_url, "raw_path": str(raw_path), "sample_wav": str(sample_wav)})

    print("\nLoading Qwen once for all samples...")
    qwen_model = load_qwen_model(
        ASR_MODEL,
        max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        use_forced_aligner=False,
    )

    all_summary: list[dict[str, Any]] = []
    for item in prepared:
        sid = item["id"]
        print(f"\n=== transcribe {sid} ===")
        out_dir = batch_out / sid
        work_dir = work_root / sid
        chunk_dir = work_dir / "chunks"
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks = chunk_wav(item["sample_wav"], chunk_dir, chunk_seconds=CHUNK_SECONDS, overlap_seconds=0)
        write_jsonl(out_dir / "chunks_manifest.jsonl", chunks)
        language = language_for_item(item)
        print("language hint:", language)
        qwen_rows = transcribe_chunks(
            qwen_model,
            chunks,
            language=language,
            batch_size=QWEN_BATCH_SIZE,
            return_time_stamps=False,
            model_name=ASR_MODEL,
        )
        for row in qwen_rows:
            row["eval_id"] = sid
            row["scenario"] = item.get("scenario")
            row["language_hint"] = item.get("language_hint")
            row["sample_wav"] = item["sample_wav"]
        write_jsonl(out_dir / "qwen_chunks.jsonl", qwen_rows)
        saved = save_transcript(out_dir, "qwen", f"Qwen3-ASR transcript: {sid}", qwen_rows)
        meta = {**item, "out_dir": str(out_dir), "qwen_jsonl": str(out_dir / "qwen_chunks.jsonl"), "transcript_md": str(saved["md_path"])}
        (out_dir / "sample_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        all_summary.append(meta)
        print(saved["text"][:1000])

    (batch_out / "batch_manifest.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== BATCH COMPLETE ===")
    print("Batch root:", batch_root)
    print("Batch outputs:", batch_out)
    print("Manifest:", batch_out / "batch_manifest.json")


if __name__ == "__main__":
    main()
