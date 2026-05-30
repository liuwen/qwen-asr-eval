from __future__ import annotations

from typing import Any

from .audio import fmt_ts


def whisper_language_for_use_case(use_case: str) -> str | None:
    mapping = {
        "zh": "zh",
        "chinese": "zh",
        "en": "en",
        "english": "en",
        "mixed": None,
        "auto": None,
    }
    return mapping.get(str(use_case).lower(), None)


def run_whisper_baseline(
    chunks: list[dict[str, Any]],
    *,
    model_size: str = "large-v3",
    use_case: str = "auto",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 1,
    vad_filter: bool = True,
) -> list[dict[str, Any]]:
    from faster_whisper import WhisperModel
    from tqdm.auto import tqdm

    language = whisper_language_for_use_case(use_case)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    rows: list[dict[str, Any]] = []
    for meta in tqdm(chunks, desc="Whisper chunks"):
        segments, info = model.transcribe(
            meta["path"],
            language=language,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        segs: list[dict[str, Any]] = []
        texts: list[str] = []
        for s in segments:
            seg = {
                "start_sec": meta["start_sec"] + float(s.start),
                "end_sec": meta["start_sec"] + float(s.end),
                "start_ts": fmt_ts(meta["start_sec"] + float(s.start)),
                "end_ts": fmt_ts(meta["start_sec"] + float(s.end)),
                "text": s.text,
            }
            segs.append(seg)
            texts.append(s.text)
        rows.append(
            {
                **meta,
                "model": f"faster-whisper/{model_size}",
                "detected_language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
                "text": "".join(texts).strip(),
                "segments": segs,
            }
        )
    return rows
