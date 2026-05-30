from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".mp4", ".mov"}


def run_cmd(cmd: list[str], *, check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print("+", " ".join(map(str, cmd)))
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def ffprobe_duration(path: str | Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    raw = res.stdout.strip()
    if not raw:
        raise ValueError(f"ffprobe returned no duration for {path}")
    return float(raw)


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def normalize_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Normalize any ffmpeg-readable media file to mono 16 kHz PCM WAV by default."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def chunk_wav(
    wav_path: str | Path,
    chunk_dir: str | Path,
    *,
    chunk_seconds: int = 300,
    overlap_seconds: int = 5,
    sample_rate: int = 16000,
) -> list[dict[str, Any]]:
    """Split a normalized WAV into timestamped chunk files."""
    wav_path = Path(wav_path)
    chunk_dir = Path(chunk_dir)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration(wav_path)
    chunks: list[dict[str, Any]] = []
    start = 0.0
    i = 0
    while start < duration - 0.05:
        end = min(duration, start + chunk_seconds)
        length = max(0.05, end - start)
        out = chunk_dir / f"chunk_{i:04d}_{int(start):06d}_{int(end):06d}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{length:.3f}",
            "-i",
            str(wav_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append(
            {
                "chunk_id": i,
                "path": str(out),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": length,
                "start_ts": fmt_ts(start),
                "end_ts": fmt_ts(end),
            }
        )
        i += 1
        if end >= duration:
            break
        start = max(0.0, end - overlap_seconds)
    return chunks


def discover_audio_files(audio_dir: str | Path, patterns: Iterable[str] | None = None) -> list[Path]:
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return []
    if patterns is None:
        candidates = [p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    else:
        candidates = []
        for pattern in patterns:
            candidates.extend([p for p in audio_dir.glob(pattern) if p.is_file()])
        candidates = [p for p in candidates if p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(set(candidates), key=lambda p: str(p).lower())


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
