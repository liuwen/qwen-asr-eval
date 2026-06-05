from __future__ import annotations

import re
import shutil
from pathlib import Path

from .artifacts import write_json
from .runtime import run_command


def sanitize_voice_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return slug[:80] or "voice"


def prepare_reference_voice(
    *,
    source_audio: str | Path,
    reference_text: str,
    voice_id: str,
    fish_repo: str | Path,
    run_dir: str | Path,
    start_seconds: float = 0.0,
    max_seconds: float = 10.0,
    sample_rate: int = 44100,
    save_reference_copy_to_drive: bool = False,
) -> dict[str, str | float | int | bool]:
    """Normalize a reference voice into Fish Speech's `references/<id>` layout."""

    source_audio = Path(source_audio)
    if not source_audio.exists():
        raise FileNotFoundError(f"Reference audio does not exist: {source_audio}")
    if not reference_text.strip():
        raise ValueError("reference_text is required for voice cloning.")

    voice_id = sanitize_voice_id(voice_id)
    fish_repo = Path(fish_repo)
    run_dir = Path(run_dir)
    ref_dir = fish_repo / "references" / voice_id
    ref_dir.mkdir(parents=True, exist_ok=True)

    wav_path = ref_dir / "sample.wav"
    lab_path = ref_dir / "sample.lab"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-ss",
        str(start_seconds),
        "-t",
        str(max_seconds),
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-vn",
        str(wav_path),
    ]
    run_command(command)
    lab_path.write_text(reference_text.strip() + "\n", encoding="utf-8")

    copied_to_drive = False
    if save_reference_copy_to_drive:
        drive_ref = run_dir / "references" / voice_id
        drive_ref.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wav_path, drive_ref / wav_path.name)
        shutil.copy2(lab_path, drive_ref / lab_path.name)
        copied_to_drive = True

    manifest = {
        "voice_id": voice_id,
        "source_audio": str(source_audio),
        "fish_reference_dir": str(ref_dir),
        "fish_reference_audio": str(wav_path),
        "fish_reference_text": str(lab_path),
        "start_seconds": start_seconds,
        "max_seconds": max_seconds,
        "sample_rate": sample_rate,
        "saved_reference_copy_to_drive": copied_to_drive,
    }
    write_json(run_dir / "manifests" / f"reference_{voice_id}.json", manifest)
    return manifest
