from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import feedparser
import requests
from tqdm.auto import tqdm

RSSHUB_BASE = "https://rsshub.app"

XIAOYUZHOU_EVAL_MANIFEST: list[dict[str, Any]] = [
    {
        "id": "cantonese_long_xiaofangjian_ep162",
        "scenario": "cantonese_long",
        "language_hint": "Cantonese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/697a4d54fed1b9fd8d7389c7",
        "duration_min": 89,
    },
    {
        "id": "cantonese_multispeaker_jianmianqingla_ep27",
        "scenario": "cantonese_multispeaker",
        "language_hint": "Cantonese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/6852da202a38b4d979587371",
        "duration_min": 63,
    },
    {
        "id": "cantonese_music_xiaofangjian_ep175",
        "scenario": "cantonese_music_speech",
        "language_hint": "Cantonese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/6a0dd9c41b7bd50295889ea6",
        "duration_min": 69,
    },
    {
        "id": "cantonese_hk_yuedan_kelly",
        "scenario": "cantonese_hk_discussion",
        "language_hint": "Cantonese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/662486f5c3e09d8f376605b5",
        "duration_min": 56,
    },
    {
        "id": "mandarin_long_sushi",
        "scenario": "mandarin_long",
        "language_hint": "Chinese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/69f08ec360313a2456c966c7",
        "duration_min": 116,
    },
    {
        "id": "mandarin_multispeaker_gushi_fm_e895",
        "scenario": "mandarin_multispeaker",
        "language_hint": "Chinese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/6a0dc4771b7bd502958718f5",
        "duration_min": 48,
    },
    {
        "id": "mandarin_variable_quality_xiaxi",
        "scenario": "mandarin_variable_audio_quality",
        "language_hint": "Chinese",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/6772e7a115a5fd520ed5d71b",
        "duration_min": 99,
    },
    {
        "id": "english_clear_aee_2618",
        "scenario": "english_clear",
        "language_hint": "English",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/6a0a9e0fe9161a38ce487b91",
        "duration_min": 23,
    },
    {
        "id": "english_long_luke_843",
        "scenario": "english_long",
        "language_hint": "English",
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/6501593a70c44b09db9d29ae",
        "duration_min": 90,
    },
    {
        "id": "mixed_zh_en_chinglish_rambles_places",
        "scenario": "mixed_zh_en",
        "language_hint": None,
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/61c9ed69f97e12e3059c167b",
        "duration_min": 44,
    },
    {
        "id": "mixed_zh_en_wujimacha_22e01",
        "scenario": "mixed_zh_en_short",
        "language_hint": None,
        "episode_url": "https://www.xiaoyuzhoufm.com/episode/61d07e0b59a22458f634d41b",
        "duration_min": 14,
    },
]


def xiaoyuzhou_id(url_or_id: str) -> str:
    return url_or_id.rstrip("/").split("/")[-1]


def rsshub_xiaoyuzhou_url(url_or_id: str, *, rsshub_base: str = RSSHUB_BASE) -> str:
    episode_or_podcast_id = xiaoyuzhou_id(url_or_id)
    return f"{rsshub_base.rstrip('/')}/xiaoyuzhou/podcast/{episode_or_podcast_id}"


def scrape_xiaoyuzhou_audio_url(episode_url: str) -> str:
    """Best-effort fallback: Xiaoyuzhou episode pages embed the media URL in HTML/JSON-LD."""
    response = requests.get(
        episode_url,
        timeout=60,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    response.raise_for_status()
    html = response.text
    patterns = [
        r'https?://media\.xyzcdn\.net/[^"\\\']+',
        r'https?://[^"\\\']+?\.(?:mp3|m4a|aac)(?:\?[^"\\\']*)?',
    ]
    for pattern in patterns:
        matches = [m.replace("\\u002F", "/").replace("&amp;", "&") for m in re.findall(pattern, html)]
        if matches:
            return matches[0]
    raise RuntimeError(f"Could not find embedded audio URL in {episode_url}")


def resolve_xiaoyuzhou_audio_url(episode_url: str, *, rsshub_base: str = RSSHUB_BASE) -> str:
    episode_id = xiaoyuzhou_id(episode_url)
    feed_url = rsshub_xiaoyuzhou_url(episode_url, rsshub_base=rsshub_base)
    try:
        feed = feedparser.parse(feed_url)
        if getattr(feed, "bozo", False):
            raise RuntimeError(str(feed.bozo_exception))

        for entry in feed.entries:
            link = getattr(entry, "link", "")
            if episode_id not in link:
                continue

            for enclosure in getattr(entry, "enclosures", []):
                href = enclosure.get("href")
                if href:
                    return href

            for link_obj in getattr(entry, "links", []):
                href = link_obj.get("href")
                rel = link_obj.get("rel")
                typ = link_obj.get("type", "")
                if href and (rel == "enclosure" or typ.startswith("audio/")):
                    return href
    except Exception as exc:
        print(f"RSSHub resolve failed for {feed_url}; falling back to Xiaoyuzhou page scrape: {exc}")

    return scrape_xiaoyuzhou_audio_url(episode_url)


def download_file(url: str, output_path: str | Path, *, timeout: int = 60) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
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


def normalize_to_16k_mono_wav(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"exists: {output_path}")
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def select_manifest_items(ids: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    if not ids:
        return list(XIAOYUZHOU_EVAL_MANIFEST)
    wanted = set(ids)
    items = [item for item in XIAOYUZHOU_EVAL_MANIFEST if item["id"] in wanted]
    missing = wanted - {item["id"] for item in items}
    if missing:
        raise KeyError(f"Unknown Xiaoyuzhou eval ids: {sorted(missing)}")
    return items


def fetch_eval_audio(
    manifest: list[dict[str, Any]],
    *,
    audio_dir: str | Path,
    normalized_dir: str | Path | None = None,
    rsshub_base: str = RSSHUB_BASE,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    raw_dir = Path(audio_dir) / "raw_xiaoyuzhou"
    wav_dir = Path(normalized_dir) if normalized_dir is not None else Path(audio_dir) / "normalized_16k_mono"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for item in manifest[:limit]:
        print(f"\n=== {item['id']} ===")
        audio_url = resolve_xiaoyuzhou_audio_url(item["episode_url"], rsshub_base=rsshub_base)
        raw_path = raw_dir / f"{item['id']}.audio"
        wav_path = wav_dir / f"{item['id']}.16k_mono.wav"
        download_file(audio_url, raw_path)
        normalize_to_16k_mono_wav(raw_path, wav_path)
        results.append(
            {
                **item,
                "resolved_audio_url": audio_url,
                "raw_path": str(raw_path),
                "normalized_wav_path": str(wav_path),
            }
        )
    return results
