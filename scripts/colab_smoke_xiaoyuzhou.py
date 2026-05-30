#!/usr/bin/env python3
"""Colab smoke test for the Qwen3-ASR eval workflow, no notebook required.

It verifies, end-to-end:
1. public repo clone/pull,
2. dependency install,
3. HF_TOKEN from Colab Secrets,
4. Xiaoyuzhou episode audio resolution/download if missing,
5. ffmpeg transcode to 16 kHz mono WAV,
6. Qwen3-ASR model download/load/transcription,
7. google.colab.ai text-only quality triage.

Run from local machine with an authenticated Colab CLI:
    colab exec -s qwen-asr-a100 --file scripts/colab_smoke_xiaoyuzhou.py

Durable artifacts are written to Google Drive by default:
    /content/drive/MyDrive/asr/audio/...
    /content/drive/MyDrive/asr/qwen-asr-eval/runs/...
    /content/drive/MyDrive/asr/hf_cache/...
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PUBLIC_REPO_URL = os.environ.get("PUBLIC_REPO_URL", "https://github.com/YOUR_GITHUB_USERNAME/qwen-asr-eval.git")
PUBLIC_REPO_BRANCH = os.environ.get("PUBLIC_REPO_BRANCH", "main")
REPO_DIR = Path(os.environ.get("REPO_DIR", "/content/qwen-asr-eval"))
DRIVE_ROOT = Path(os.environ.get("ASR_EVAL_DRIVE_ROOT", "/content/drive/MyDrive/asr"))
AUDIO_ROOT = Path(os.environ.get("ASR_EVAL_AUDIO_ROOT", str(DRIVE_ROOT / "audio")))
RUNS_ROOT = Path(os.environ.get("ASR_EVAL_RUNS_ROOT", str(DRIVE_ROOT / "qwen-asr-eval" / "runs")))
HF_CACHE_ROOT = Path(os.environ.get("ASR_EVAL_HF_CACHE", str(DRIVE_ROOT / "hf_cache")))
# Ephemeral working chunks are okay; durable audio, outputs, and model cache live in Drive.
TMP_ROOT = Path(os.environ.get("ASR_EVAL_TMP_ROOT", "/content/asr-eval-work"))
SMOKE_AUDIO_ID = os.environ.get("SMOKE_AUDIO_ID", "mixed_zh_en_wujimacha_22e01")
SMOKE_SECONDS = int(os.environ.get("SMOKE_SECONDS", "45"))
CHUNK_SECONDS = int(os.environ.get("SMOKE_CHUNK_SECONDS", "45"))
ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
COLAB_AI_MODEL = os.environ.get("COLAB_AI_MODEL", "google/gemini-3.5-flash")
REQUIRE_HF_TOKEN = os.environ.get("ASR_EVAL_REQUIRE_HF_TOKEN", "0") == "1"
REQUIRE_COLAB_AI_JUDGE = os.environ.get("ASR_EVAL_REQUIRE_COLAB_AI_JUDGE", "0") == "1"


def run(cmd: list[str], *, cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def mount_drive() -> None:
    """Require Drive to be mounted, but do not start an interactive auth flow inside exec.

    `google.colab.drive.mount()` can block/fail under `colab exec`. For CLI runs, mount Drive
    first with `colab drivemount -s qwen-asr-a100 /content/drive` and finish the browser auth.
    """
    if not str(DRIVE_ROOT).startswith("/content/drive/"):
        print("Drive mount check skipped because ASR_EVAL_DRIVE_ROOT is not under /content/drive:", DRIVE_ROOT)
        return

    my_drive = Path("/content/drive/MyDrive")
    if my_drive.exists():
        print("Drive already mounted:", my_drive)
        return

    if os.environ.get("ASR_EVAL_INTERACTIVE_DRIVE_MOUNT") == "1":
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        if my_drive.exists():
            print("Drive mounted at /content/drive")
            return

    raise RuntimeError(
        "Google Drive is not mounted. For CLI workflow, run this once and authorize in browser:\n"
        "  colab drivemount -s qwen-asr-a100 /content/drive\n"
        "Then rerun:\n"
        "  colab exec -s qwen-asr-a100 --file scripts/colab_smoke_xiaoyuzhou.py\n"
        "If running inside the notebook UI, run drive.mount('/content/drive') first."
    )


def configure_persistent_cache() -> None:
    HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_ROOT / "transformers"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print("HF_HOME:", os.environ["HF_HOME"])
    print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])


def ensure_repo() -> None:
    if REPO_DIR.exists():
        run(["git", "-C", str(REPO_DIR), "fetch", "origin", PUBLIC_REPO_BRANCH], check=False)
        run(["git", "-C", str(REPO_DIR), "checkout", PUBLIC_REPO_BRANCH], check=False)
        run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", "--depth", "1", "--branch", PUBLIC_REPO_BRANCH, PUBLIC_REPO_URL, str(REPO_DIR)])


def install_deps() -> None:
    run(["apt-get", "-qq", "update"])
    run(["apt-get", "-qq", "install", "-y", "ffmpeg", "jq"])
    # Minimal smoke deps: avoid optional faster-whisper to make the first verification faster.
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


def configure_hf_token() -> None:
    """Configure HF_TOKEN when available.

    Colab Secrets are reliably readable from notebook/UI cells, but `google.colab.userdata`
    can time out under `colab exec`. Qwen/Qwen3-ASR-1.7B is public, so the CLI smoke
    workflow treats HF_TOKEN as optional by default. Set ASR_EVAL_REQUIRE_HF_TOKEN=1 to
    make absence fatal.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from google.colab import userdata

            token = userdata.get("HF_TOKEN")
        except Exception as exc:
            print("Could not read Colab Secret HF_TOKEN under colab exec:", repr(exc))
    if not token:
        message = (
            "HF_TOKEN is not available to this CLI execution. Continuing without it because "
            "ASR_EVAL_REQUIRE_HF_TOKEN is not set. Public model downloads should still work; "
            "if Hugging Face rate-limits or rejects the download, run from the Colab UI or set "
            "HF_TOKEN in the runtime environment first."
        )
        if REQUIRE_HF_TOKEN:
            raise RuntimeError(message)
        print("WARNING:", message)
        return

    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("HF_TOKEN configured.")
    except Exception as exc:
        print("HF login failed, but HF_TOKEN env is still set:", repr(exc))


def main() -> None:
    print("=== Qwen3-ASR Colab smoke test ===")
    print("UTC-ish run id:", datetime.utcnow().strftime("%Y%m%d_%H%M%S"))
    print("Repo:", PUBLIC_REPO_URL, PUBLIC_REPO_BRANCH)
    print("Drive root:", DRIVE_ROOT)
    print("Audio root (persistent):", AUDIO_ROOT)
    print("Runs root (persistent):", RUNS_ROOT)
    print("HF cache (persistent):", HF_CACHE_ROOT)
    print("Temp root (ephemeral):", TMP_ROOT)
    print("Require HF_TOKEN:", REQUIRE_HF_TOKEN)
    print("Require Colab AI judge:", REQUIRE_COLAB_AI_JUDGE)

    mount_drive()
    configure_persistent_cache()
    ensure_repo()
    install_deps()
    # `colab exec` runs inside an already-started IPython kernel; after pip install,
    # the current process may not pick up editable-install path changes. Add src directly.
    src_path = str(REPO_DIR / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    configure_hf_token()

    from asr_eval.audio import chunk_wav, ffprobe_duration, fmt_ts, write_jsonl
    from asr_eval.colab_ai_judge import judge_chunks_colab_ai
    from asr_eval.qwen_runner import load_qwen_model, transcribe_chunks
    from asr_eval.reporting import save_transcript
    from asr_eval.xiaoyuzhou import download_file, resolve_xiaoyuzhou_audio_url, select_manifest_items

    run_id = datetime.utcnow().strftime("smoke_%Y%m%d_%H%M%S")
    work_dir = TMP_ROOT / run_id
    out_dir = RUNS_ROOT / run_id / "outputs"
    chunk_dir = work_dir / "chunks"
    work_dir.mkdir(parents=True, exist_ok=True)
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    item = select_manifest_items([SMOKE_AUDIO_ID])[0]
    print("Smoke item:", json.dumps(item, ensure_ascii=False, indent=2))

    audio_url = resolve_xiaoyuzhou_audio_url(item["episode_url"])
    raw_path = AUDIO_ROOT / "raw_xiaoyuzhou" / f"{item['id']}.audio"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    download_file(audio_url, raw_path)

    sample_dir = AUDIO_ROOT / "smoke_samples_16k_mono"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_wav = sample_dir / f"{item['id']}.first_{SMOKE_SECONDS}s.16k_mono.wav"
    if sample_wav.exists() and sample_wav.stat().st_size > 0:
        print("exists:", sample_wav)
    else:
        run(
            [
                "ffmpeg",
                "-y",
                "-t",
                str(SMOKE_SECONDS),
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
    print("Smoke WAV:", sample_wav)
    print("Smoke WAV duration:", fmt_ts(ffprobe_duration(sample_wav)))

    chunks = chunk_wav(sample_wav, chunk_dir, chunk_seconds=CHUNK_SECONDS, overlap_seconds=0)
    chunks = chunks[:1]
    write_jsonl(out_dir / "chunks_manifest.jsonl", chunks)
    print("Chunks:", json.dumps(chunks, ensure_ascii=False, indent=2))

    print("Loading Qwen model. This is the slow step on first run because it downloads model weights.")
    qwen_model = load_qwen_model(
        ASR_MODEL,
        max_inference_batch_size=1,
        max_new_tokens=1024,
        use_forced_aligner=False,
    )
    qwen_rows = transcribe_chunks(
        qwen_model,
        chunks,
        language=None,
        batch_size=1,
        return_time_stamps=False,
        model_name=ASR_MODEL,
    )
    write_jsonl(out_dir / "qwen_chunks.jsonl", qwen_rows)
    saved = save_transcript(out_dir, "qwen_smoke", "Qwen3-ASR smoke transcript", qwen_rows)
    print("Transcript preview:")
    print(saved["text"][:2000])

    print("Running google.colab.ai text-only triage. This does not listen to audio.")
    try:
        judge_reports = judge_chunks_colab_ai(
            qwen_rows,
            whisper_rows=None,
            use_case="mixed",
            model_name=COLAB_AI_MODEL,
            max_chunks=1,
        )
        print("Colab AI report:")
        print(json.dumps(judge_reports, ensure_ascii=False, indent=2))
    except Exception as exc:
        # google.colab.ai depends on Colab's MODEL_PROXY_API_KEY, which is often only
        # available to notebook/UI execution and can time out under `colab exec`.
        judge_reports = [
            {
                "judge_type": "text_only_colab_ai",
                "status": "skipped_or_failed",
                "audio_faithfulness_verified": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": (
                    "Qwen ASR succeeded, but google.colab.ai was unavailable in CLI execution. "
                    "Run the notebook/UI judge cell to complete Colab AI triage, or set "
                    "ASR_EVAL_REQUIRE_COLAB_AI_JUDGE=1 to make this fatal."
                ),
            }
        ]
        if REQUIRE_COLAB_AI_JUDGE:
            raise
        print("WARNING: Colab AI triage unavailable under CLI; wrote skipped report:", repr(exc))
    (out_dir / "colab_ai_text_judge.json").write_text(
        json.dumps(judge_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "ok": True,
        "item": item,
        "raw_path": str(raw_path),
        "sample_wav": str(sample_wav),
        "out_dir": str(out_dir),
        "hf_cache": str(HF_CACHE_ROOT),
        "qwen_jsonl": str(out_dir / "qwen_chunks.jsonl"),
        "judge_json": str(out_dir / "colab_ai_text_judge.json"),
    }
    print("=== SMOKE TEST PASSED ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
