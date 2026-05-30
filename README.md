# Qwen3-ASR Colab Evaluation Starter

A small, repo-friendly Colab workflow for testing `Qwen/Qwen3-ASR-1.7B` on A100 GPU runtimes.

Target cases:

1. Long audio tracks, including podcasts up to about 2 hours.
2. Pure Chinese or pure English audio, including possible multi-speaker speech.
3. Chinese-English mixed / code-switching audio.
4. Reference-based metrics when a transcript exists.
5. Optional open ASR baseline with `faster-whisper`.
6. Text-only quality triage with Colab AI via `google.colab.ai`.

## Design decision

The notebook is deliberately thin. Reusable logic lives in `src/asr_eval/`. This keeps future fixes as normal Git patches instead of fragile notebook-cell surgery.

```text
qwen-asr-eval/
  notebooks/qwen3_asr_colab_eval.ipynb
  src/asr_eval/
    audio.py
    qwen_runner.py
    whisper_baseline.py
    metrics.py
    colab_ai_judge.py
    reporting.py
  scripts/run_eval.py
  requirements-colab.txt
  docs/colab_workflow.md
```

## Secrets

Required in Colab Secrets:

```text
HF_TOKEN
```

Not used:

```text
GH_TOKEN      # public repo only
GEMINI_API_KEY # not needed; Colab AI is used for text-only judging
```

Inside Colab, the notebook reads:

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
```

## Important limitation: Colab AI judge is text-only

`google.colab.ai.generate_text(...)` only supports text-to-text input/output. It cannot listen to audio. Therefore it is **not** a true ASR judge.

Use it for:

- language-profile sanity checks,
- repetition / hallucination-style red flags,
- Qwen-vs-Whisper disagreement triage,
- deciding which timestamps deserve manual audit.

Use reference transcripts or an audio-capable evaluator for real faithfulness scoring.

## Generic self-contained Colab notebook

For generic release/use, prefer:

```text
notebooks/qwen3_asr_generic_colab.ipynb
```

Open it in Colab:

```text
https://colab.research.google.com/github/liuwen/qwen-asr-eval/blob/main/notebooks/qwen3_asr_generic_colab.ipynb
```

It supports Drive paths, direct download URLs, and Xiaoyuzhou episode URLs; provides T4/A100/H100 runtime presets; writes checkpointed outputs to Drive; and includes optional Colab GenAI text-only evaluation, high-fidelity proofreading, and enhancement cells.

## Recommended Colab startup

With the official Colab CLI already authenticated locally:

```bash
colab new -s qwen-asr-a100 --gpu A100
colab url -s qwen-asr-a100 --open
```

Then open `notebooks/qwen3_asr_colab_eval.ipynb` in the Colab UI and set:

```text
PUBLIC_REPO_URL = "https://github.com/<you>/qwen-asr-eval.git"
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/asr/qwen-asr-eval"
DRIVE_AUDIO_DIR = "/content/drive/MyDrive/asr/audio"
```

Upload audio files manually to `DRIVE_AUDIO_DIR`.

## First push

```bash
git init
git add .
git commit -m "init qwen asr colab eval"
git branch -M main
# create a public repo yourself, then:
git remote add origin https://github.com/<you>/qwen-asr-eval.git
git push -u origin main
```

## Output files

Each run writes outputs under:

```text
<DRIVE_PROJECT_ROOT>/runs/<timestamp>/outputs/
```

Typical files:

```text
qwen_chunks.jsonl
qwen_chunks.csv
qwen_transcript_chunked.txt
qwen_transcript_chunked.md
whisper_chunks.jsonl           # optional
whisper_transcript_chunked.md   # optional
reference_metrics.csv           # if reference transcript path is configured
colab_ai_text_judge.json        # optional text-only triage
```

## Preferred chunking defaults

```text
ASR_CHUNK_SECONDS = 300
ASR_OVERLAP_SECONDS = 5
TIMESTAMP_CHUNK_SECONDS = 180
```

For a 2-hour podcast, 5-minute chunks produce about 24 chunks. This is easier to retry, inspect, and compare than one giant inference item.
