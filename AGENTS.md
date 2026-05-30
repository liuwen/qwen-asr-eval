# AGENTS.md

## Purpose

This repo evaluates Qwen3-ASR on Colab A100 runtimes. Keep the notebook thin and keep reusable code in `src/asr_eval/`.

## Hard preferences

- Public GitHub repo. Do not require `GH_TOKEN`.
- Use Colab Secrets for `HF_TOKEN`.
- Do not require `GEMINI_API_KEY`.
- Use `google.colab.ai` only for text-to-text quality triage/proofreading/enhancement.
- Run Colab GenAI only from the Colab Web UI; do not call it from `colab exec` scripts.
- Do not try to use `MODEL_PROXY_API_KEY` or `GEMINI_API_KEY` for Colab GenAI.
- Do not claim Colab AI can listen to audio or perform ASR.
- Keep `faster-whisper` baseline optional.
- Always preserve chunk timestamps in outputs.
- Default long-audio chunking: 300 seconds with 5 seconds overlap.
- Use 16 kHz mono WAV as the normalized working format.

## Notebook policy

The notebook should:

1. Mount Drive.
2. Clone/install this public repo.
3. Read `HF_TOKEN` from Colab Secrets.
4. Let the user configure Drive audio paths at startup.
5. Run Qwen ASR.
6. Optionally run Whisper baseline.
7. Optionally run reference metrics.
8. Optionally run Colab AI text-only triage.

## Script-driven Colab policy

The preferred automation path is script-driven and documented in `docs/script_driven_colab_workflow.md`.

- Do not require notebook edits after the public GitHub repo is available.
- Use `scripts/colab_smoke_xiaoyuzhou.py` for an end-to-end smoke test.
- Require Drive to be mounted before `colab exec`; avoid interactive `drive.mount()` inside CLI-run scripts.
- Store durable artifacts in Drive:
  - `/content/drive/MyDrive/asr/audio/`
  - `/content/drive/MyDrive/asr/qwen-asr-eval/runs/`
  - `/content/drive/MyDrive/asr/hf_cache/`
- Treat `/content/...` as ephemeral clone/scratch only.
- Keep Xiaoyuzhou downloaded audio private for ASR evaluation; do not redistribute source audio or full transcripts unless permitted by the publisher.

## Evaluation honesty

Without a reference transcript or audio-capable judge, the system cannot know whether the ASR text is faithful to the audio. Text-only LLM judgment is useful for triage, not final quality measurement.
