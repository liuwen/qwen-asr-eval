from __future__ import annotations

import re
import unicodedata
from typing import Any

from jiwer import cer, wer

PUNCT_RE = re.compile(r"[\s\u3000\.,!?;:'\"`~@#$%^&*()_+\-=\[\]{}|\\/<>，。！？；：‘’“”、《》【】（）—…·]+")


def normalize_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return PUNCT_RE.sub("", text)


def normalize_for_wer(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_cer = normalize_for_cer(reference)
    hyp_cer = normalize_for_cer(hypothesis)
    ref_wer = normalize_for_wer(reference)
    hyp_wer = normalize_for_wer(hypothesis)
    return {
        "CER_all_chars": cer(ref_cer, hyp_cer) if ref_cer else None,
        "WER_space_tokenized": wer(ref_wer, hyp_wer) if ref_wer else None,
        "ref_chars": len(ref_cer),
        "hyp_chars": len(hyp_cer),
        "ref_words": len(ref_wer.split()),
        "hyp_words": len(hyp_wer.split()),
    }
