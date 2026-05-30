from __future__ import annotations

from typing import Any


def qwen_language_for_use_case(use_case: str) -> str | None:
    mapping = {
        "zh": "Chinese",
        "chinese": "Chinese",
        "mandarin": "Chinese",
        "cantonese": "Cantonese",
        "yue": "Cantonese",
        "en": "English",
        "english": "English",
        "mixed": None,
        "auto": None,
        "": None,
        "none": None,
    }
    return mapping.get(str(use_case).lower(), None)


def serialize_obj(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_obj(x) for x in obj]
    fields: dict[str, Any] = {}
    for attr in ["text", "start_time", "end_time", "start", "end", "language"]:
        if hasattr(obj, attr):
            try:
                fields[attr] = serialize_obj(getattr(obj, attr))
            except Exception:
                pass
    return fields or str(obj)


def batched(xs: list[Any], n: int):
    n = max(1, int(n))
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def load_qwen_model(
    model_id: str = "Qwen/Qwen3-ASR-1.7B",
    *,
    max_inference_batch_size: int = 4,
    max_new_tokens: int = 4096,
    use_forced_aligner: bool = False,
    forced_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
):
    import gc
    import torch
    from qwen_asr import Qwen3ASRModel

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dtype = torch.bfloat16
        device_map = "cuda:0"
    else:
        dtype = torch.float32
        device_map = "cpu"

    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "max_inference_batch_size": int(max_inference_batch_size),
        "max_new_tokens": int(max_new_tokens),
    }
    if use_forced_aligner:
        load_kwargs.update(
            {
                "forced_aligner": forced_aligner_model,
                "forced_aligner_kwargs": {
                    "dtype": dtype,
                    "device_map": device_map,
                },
            }
        )
    return Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)


def transcribe_chunks(
    model: Any,
    chunks: list[dict[str, Any]],
    *,
    language: str | None = None,
    batch_size: int = 1,
    return_time_stamps: bool = False,
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    rows: list[dict[str, Any]] = []
    batches = list(batched(chunks, max(1, int(batch_size))))
    for batch in tqdm(batches, desc="Qwen chunks"):
        audio_paths = [r["path"] for r in batch]
        if len(audio_paths) == 1:
            audio_arg: str | list[str] = audio_paths[0]
            lang_arg = language
        else:
            audio_arg = audio_paths
            lang_arg = None if language is None else [language] * len(audio_paths)
        results = model.transcribe(
            audio=audio_arg,
            language=lang_arg,
            return_time_stamps=return_time_stamps,
        )
        if not isinstance(results, list):
            results = [results]
        for meta, result in zip(batch, results):
            rows.append(
                {
                    **meta,
                    "model": model_name,
                    "detected_language": getattr(result, "language", None),
                    "text": getattr(result, "text", str(result)),
                    "time_stamps": serialize_obj(getattr(result, "time_stamps", None)),
                }
            )
    return rows
