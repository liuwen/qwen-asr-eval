from __future__ import annotations

from pathlib import Path
from typing import Any


def longest_suffix_prefix_overlap(a: str, b: str, *, max_chars: int = 300, min_chars: int = 12) -> int:
    """Return exact char overlap length between suffix(a) and prefix(b)."""
    if not a or not b:
        return 0
    a_tail = a[-max_chars:]
    b_head = b[:max_chars]
    max_len = min(len(a_tail), len(b_head))
    for n in range(max_len, min_chars - 1, -1):
        if a_tail[-n:] == b_head[:n]:
            return n
    return 0


def merge_text_rows(rows: Any, *, dedupe_overlap: bool = True) -> str:
    """Merge chunk rows from a pandas DataFrame or list of dicts into a timestamped transcript."""
    if hasattr(rows, "sort_values"):
        iterable = rows.sort_values("chunk_id").to_dict("records")
    else:
        iterable = sorted(list(rows), key=lambda r: int(r.get("chunk_id", 0)))

    parts: list[str] = []
    prev_text = ""
    for row in iterable:
        text = str(row.get("text") or "").strip()
        if dedupe_overlap and prev_text and text:
            n = longest_suffix_prefix_overlap(prev_text, text)
            if n:
                text = text[n:].lstrip()
        parts.append(f"[{row.get('start_ts', '')} - {row.get('end_ts', '')}]\n{text}".strip())
        prev_text = (prev_text + text)[-1000:]
    return "\n\n".join(parts).strip() + "\n"


def save_transcript(out_dir: str | Path, name: str, title: str, rows: Any) -> dict[str, Path | str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = merge_text_rows(rows, dedupe_overlap=True)
    txt = out_dir / f"{name}_transcript_chunked.txt"
    md = out_dir / f"{name}_transcript_chunked.md"
    txt.write_text(text, encoding="utf-8")
    md.write_text(f"# {title}\n\n" + text, encoding="utf-8")
    return {"text": text, "txt_path": txt, "md_path": md}
