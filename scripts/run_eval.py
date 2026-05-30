#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from asr_eval.audio import chunk_wav, normalize_to_wav, write_jsonl
from asr_eval.metrics import compute_metrics
from asr_eval.qwen_runner import load_qwen_model, qwen_language_for_use_case, transcribe_chunks
from asr_eval.reporting import save_transcript
from asr_eval.whisper_baseline import run_whisper_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR eval on one audio file.")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/local/outputs"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/local/work"))
    parser.add_argument("--use-case", choices=["auto", "zh", "en", "mixed"], default="auto")
    parser.add_argument("--chunk-seconds", type=int, default=300)
    parser.add_argument("--overlap-seconds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_to_wav(args.audio, args.work_dir / "normalized_16k_mono.wav")
    chunks = chunk_wav(normalized, args.work_dir / "chunks", chunk_seconds=args.chunk_seconds, overlap_seconds=args.overlap_seconds)
    write_jsonl(args.out_dir / "chunks_manifest.jsonl", chunks)

    model = load_qwen_model(args.model_id, max_inference_batch_size=args.max_inference_batch_size, max_new_tokens=args.max_new_tokens)
    qwen_rows = transcribe_chunks(
        model,
        chunks,
        language=qwen_language_for_use_case(args.use_case),
        batch_size=args.batch_size,
        model_name=args.model_id,
    )
    write_jsonl(args.out_dir / "qwen_chunks.jsonl", qwen_rows)
    qwen_df = pd.DataFrame(qwen_rows)
    qwen_df.drop(columns=["time_stamps"], errors="ignore").to_csv(args.out_dir / "qwen_chunks.csv", index=False)
    qwen_saved = save_transcript(args.out_dir, "qwen", "Qwen3-ASR transcript", qwen_rows)

    metric_rows = []
    if args.reference:
        reference = args.reference.read_text(encoding="utf-8")
        metric_rows.append({"system": "qwen", **compute_metrics(reference, qwen_saved["text"])})

    if args.whisper:
        whisper_rows = run_whisper_baseline(chunks, model_size=args.whisper_model, use_case=args.use_case)
        write_jsonl(args.out_dir / "whisper_chunks.jsonl", whisper_rows)
        whisper_df = pd.DataFrame(whisper_rows)
        whisper_df.drop(columns=["segments"], errors="ignore").to_csv(args.out_dir / "whisper_chunks.csv", index=False)
        whisper_saved = save_transcript(args.out_dir, "whisper", "faster-whisper transcript", whisper_rows)
        if args.reference:
            metric_rows.append({"system": "whisper", **compute_metrics(reference, whisper_saved["text"])})

    if metric_rows:
        (args.out_dir / "reference_metrics.json").write_text(json.dumps(metric_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(metric_rows).to_csv(args.out_dir / "reference_metrics.csv", index=False)

    print("Output dir:", args.out_dir)


if __name__ == "__main__":
    main()
