from __future__ import annotations

import json
import re
from typing import Any

from rapidfuzz import fuzz


def as_records(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        return rows.to_dict("records")
    return list(rows)


def safe_json_loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {"raw_text": text}


def select_eval_chunk_ids(qwen_rows: Any, whisper_rows: Any | None = None, *, max_chunks: int = 6) -> list[int]:
    qrows = as_records(qwen_rows)
    if not qrows:
        return []
    ids = [int(r["chunk_id"]) for r in sorted(qrows, key=lambda r: int(r["chunk_id"]))]
    selected = {ids[0], ids[len(ids) // 2], ids[-1]}

    wrows = as_records(whisper_rows)
    if wrows:
        qmap = {int(r["chunk_id"]): str(r.get("text") or "") for r in qrows}
        wmap = {int(r["chunk_id"]): str(r.get("text") or "") for r in wrows}
        scored = []
        for cid in sorted(set(qmap) & set(wmap)):
            similarity = fuzz.token_set_ratio(qmap[cid], wmap[cid])
            length_ratio = (len(qmap[cid]) + 1) / (len(wmap[cid]) + 1)
            scored.append((similarity, abs(1.0 - length_ratio), cid))
        # Prefer chunks where baseline and Qwen disagree.
        for _, _, cid in sorted(scored, key=lambda x: (x[0], -x[1]))[:max_chunks]:
            selected.add(cid)
            if len(selected) >= max_chunks:
                break

    # Fill with evenly spaced chunks.
    stride = max(1, len(ids) // max(1, max_chunks))
    for cid in ids[::stride]:
        selected.add(cid)
        if len(selected) >= max_chunks:
            break
    return sorted(selected)[:max_chunks]


def _row_by_id(rows: list[dict[str, Any]], chunk_id: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("chunk_id", -1)) == int(chunk_id):
            return row
    raise KeyError(chunk_id)


def build_text_only_judge_prompt(
    qwen_row: dict[str, Any],
    whisper_row: dict[str, Any] | None,
    *,
    use_case: str,
) -> str:
    qwen_text = str(qwen_row.get("text") or "")[:12000]
    whisper_text = str((whisper_row or {}).get("text") or "")[:12000]
    baseline_block = ""
    if whisper_row is not None:
        baseline_block = f"""
## Candidate B: faster-whisper baseline transcript
{whisper_text}
""".strip()
    else:
        baseline_block = "No baseline transcript is available. Judge Candidate A only for text-level quality risks."

    return f"""
You are an ASR output QA analyst. You only receive text transcripts; you do not receive the audio. Be honest about this limitation.

Task context:
- Chunk id: {qwen_row.get('chunk_id')}
- Source time range: {qwen_row.get('start_ts')} to {qwen_row.get('end_ts')}
- Target use case: {use_case}
- Expected cases: pure Chinese, pure English, or Chinese-English code switching.

Evaluate the transcript text for likely ASR quality risks:
1. obvious repetition loops,
2. broken punctuation or segmentation,
3. language inconsistency for the requested use case,
4. suspicious hallucination markers such as generic filler, unrelated boilerplate, or impossible formatting,
5. if a baseline exists, major disagreement between candidates.

Do not claim you verified faithfulness to the audio. You cannot hear the audio.

Return JSON only, with this schema:
{{
  "chunk_id": {qwen_row.get('chunk_id')},
  "judge_type": "text_only_colab_ai",
  "audio_faithfulness_verified": false,
  "language_profile": "Chinese|English|mixed|unclear|other",
  "qwen_text_quality_0_100": 0,
  "qwen_red_flags": ["..."],
  "baseline_disagreement": "none|low|medium|high|not_available",
  "prefer_for_manual_review": true,
  "manual_review_reason": "...",
  "recommended_next_action": "accept|spot_check|rerun_with_forced_language|rerun_with_shorter_chunks|needs_human_audio_audit",
  "verdict": "short practical verdict"
}}

## Candidate A: Qwen3-ASR transcript
{qwen_text}

## Candidate B
{baseline_block}
""".strip()


def judge_chunks_colab_ai(
    qwen_rows: Any,
    whisper_rows: Any | None = None,
    *,
    use_case: str = "auto",
    model_name: str = "google/gemini-2.5-flash",
    max_chunks: int = 6,
) -> list[dict[str, Any]]:
    """Run text-only Colab AI triage over selected chunks.

    This does not send audio and cannot verify ASR faithfulness.
    """
    try:
        from google.colab import ai
    except Exception as exc:  # pragma: no cover - only valid inside Colab
        raise RuntimeError("google.colab.ai is only available inside Colab environments") from exc

    qrecords = as_records(qwen_rows)
    wrecords = as_records(whisper_rows)
    selected_ids = select_eval_chunk_ids(qrecords, wrecords, max_chunks=max_chunks)
    reports: list[dict[str, Any]] = []
    for cid in selected_ids:
        qrow = _row_by_id(qrecords, cid)
        wrow = _row_by_id(wrecords, cid) if wrecords else None
        prompt = build_text_only_judge_prompt(qrow, wrow, use_case=use_case)
        response = ai.generate_text(prompt, model_name=model_name)
        data = safe_json_loads(str(response))
        data.setdefault("chunk_id", cid)
        data["_colab_ai_model"] = model_name
        data["_source_time_range"] = [qrow.get("start_ts"), qrow.get("end_ts")]
        reports.append(data)
    return reports
