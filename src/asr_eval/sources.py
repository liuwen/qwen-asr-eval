from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests
from tqdm.auto import tqdm

from .xiaoyuzhou import resolve_xiaoyuzhou_audio_url, xiaoyuzhou_id


def safe_stem(value: str, *, fallback: str = "audio") -> str:
    value = unquote(str(value or "")).strip().rstrip("/")
    if not value:
        return fallback
    stem = Path(urlparse(value).path).stem or Path(value).stem or fallback
    stem = re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", stem).strip("._-")
    return stem[:120] or fallback


def filename_from_url(url: str, *, fallback_stem: str = "audio", default_suffix: str = ".audio") -> str:
    path = Path(urlparse(url).path)
    name = unquote(path.name)
    if name and "." in name:
        return re.sub(r"[^A-Za-z0-9._\-\u4e00-\u9fff]+", "_", name)[:160]
    return safe_stem(fallback_stem) + default_suffix


def download_url(url: str, output_path: str | Path, *, timeout: int = 60, force: bool = False) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not force:
        print(f"exists: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp_path.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp_path.replace(output_path)
    return output_path


def prepare_audio_source(
    *,
    source_type: str,
    drive_audio_path: str | Path | None = None,
    direct_url: str | None = None,
    xiaoyuzhou_url: str | None = None,
    raw_audio_dir: str | Path,
    output_filename: str | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    """Resolve an audio input to a local path.

    source_type values:
    - drive_path: use an existing path in Drive/Colab filesystem
    - direct_url: download a direct URL or any HTTP(S) media URL
    - xiaoyuzhou: resolve episode URL to embedded/RSS audio URL, then download
    """
    raw_audio_dir = Path(raw_audio_dir)
    raw_audio_dir.mkdir(parents=True, exist_ok=True)
    kind = str(source_type or "drive_path").lower()

    if kind == "drive_path":
        if not drive_audio_path:
            raise ValueError("drive_audio_path is required for source_type='drive_path'")
        path = Path(drive_audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return {"source_type": kind, "path": str(path), "resolved_url": None, "source_id": safe_stem(str(path))}

    if kind == "direct_url":
        if not direct_url:
            raise ValueError("direct_url is required for source_type='direct_url'")
        filename = output_filename or filename_from_url(direct_url, fallback_stem="direct_audio")
        path = download_url(direct_url, raw_audio_dir / filename, force=force_download)
        return {"source_type": kind, "path": str(path), "resolved_url": direct_url, "source_id": safe_stem(filename)}

    if kind == "xiaoyuzhou":
        if not xiaoyuzhou_url:
            raise ValueError("xiaoyuzhou_url is required for source_type='xiaoyuzhou'")
        resolved = resolve_xiaoyuzhou_audio_url(xiaoyuzhou_url)
        source_id = safe_stem(output_filename or xiaoyuzhou_id(xiaoyuzhou_url), fallback="xiaoyuzhou_audio")
        filename = output_filename or f"{source_id}.audio"
        path = download_url(resolved, raw_audio_dir / filename, force=force_download)
        return {"source_type": kind, "path": str(path), "resolved_url": resolved, "source_id": source_id}

    raise ValueError("source_type must be one of: drive_path, direct_url, xiaoyuzhou")
