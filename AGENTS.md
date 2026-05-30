# AGENTS.md

## Purpose

This repo evaluates Qwen3-ASR on Colab A100 runtimes. Keep the notebook thin and keep reusable code in `src/asr_eval/`.

## Hard preferences

- Public GitHub repo. Do not require `GH_TOKEN`.
- Use Colab Secrets for `HF_TOKEN`.
- Do not require `GEMINI_API_KEY`.
- Use `google.colab.ai` only for text-to-text quality triage.
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

## Evaluation honesty

Without a reference transcript or audio-capable judge, the system cannot know whether the ASR text is faithful to the audio. Text-only LLM judgment is useful for triage, not final quality measurement.
