from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def read_tail(path: str | Path, lines: int = 60) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    data = target.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(data[-lines:])
