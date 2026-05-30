# Script-driven Colab workflow

This workflow runs the Qwen3-ASR evaluation from Python scripts through the local `colab` CLI. The notebook remains useful as a visual control panel, but it is not required once the public GitHub repo is available.

## Storage model

Use Google Drive for durable artifacts and `/content` only for disposable runtime work.

| Path | Purpose | Persists after runtime expires? |
| --- | --- | --- |
| `/content/qwen-asr-eval` | Fresh public Git clone | No |
| `/content/asr-eval-work` | Temporary chunks/work scratch | No |
| `/content/drive/MyDrive/asr/audio/raw_xiaoyuzhou` | Downloaded source audio enclosures | Yes |
| `/content/drive/MyDrive/asr/audio/smoke_samples_16k_mono` | Short smoke-test WAVs | Yes |
| `/content/drive/MyDrive/asr/audio/normalized_16k_mono` | Full normalized WAV artifacts | Yes |
| `/content/drive/MyDrive/asr/qwen-asr-eval/runs` | ASR outputs, transcripts, reports | Yes |
| `/content/drive/MyDrive/asr/hf_cache` | Hugging Face model cache | Yes |

The script sets these cache variables before model loading:

```text
HF_HOME=/content/drive/MyDrive/asr/hf_cache
HF_HUB_CACHE=/content/drive/MyDrive/asr/hf_cache/hub
TRANSFORMERS_CACHE=/content/drive/MyDrive/asr/hf_cache/transformers
HF_HUB_ENABLE_HF_TRANSFER=1
```

This avoids re-downloading Qwen weights on every new VM, at the cost of Drive-backed cache I/O.

## Required manual setup

### 1. Authenticate Colab CLI locally

This is assumed already done on the developer machine.

### 2. Create an A100 runtime

```bash
colab new -s qwen-asr-a100 --gpu A100
```

If a session already exists, check it with:

```bash
colab sessions
colab status -s qwen-asr-a100
```

### 3. Mount Google Drive once per runtime

Do this before script execution:

```bash
colab drivemount -s qwen-asr-a100 /content/drive
```

The CLI prints a Google OAuth URL. Open it, grant Drive access, then press Enter in the terminal.

Do not rely on `google.colab.drive.mount()` inside `colab exec`; it may block or fail because it needs interactive browser authorization. The smoke script intentionally checks that Drive is already mounted and fails with instructions if it is not.

### 4. Add Colab Secret

In Colab web UI, add:

```text
HF_TOKEN
```

Notebook/UI cells can read it with:

```python
from google.colab import userdata
HF_TOKEN = userdata.get("HF_TOKEN")
```

Important CLI caveat: `google.colab.userdata.get("HF_TOKEN")` can time out under `colab exec` with:

```text
Secrets can only be fetched when running from the Colab UI.
```

Therefore `scripts/colab_smoke_xiaoyuzhou.py` treats `HF_TOKEN` as optional by default. `Qwen/Qwen3-ASR-1.7B` is public, so public downloads should work without it unless Hugging Face rate-limits or otherwise rejects the request. To make absence fatal, set:

```bash
ASR_EVAL_REQUIRE_HF_TOKEN=1 colab exec -s qwen-asr-a100 --file scripts/colab_smoke_xiaoyuzhou.py
```

Do not add `GH_TOKEN`; the repo is public. Do not add `GEMINI_API_KEY`; `google.colab.ai` is used only for text-to-text triage.

## Run the smoke workflow

From the local machine:

```bash
colab exec -s qwen-asr-a100 --file scripts/colab_smoke_xiaoyuzhou.py
```

The smoke script performs an end-to-end verification:

1. Requires Google Drive to already be mounted.
2. Sets persistent Hugging Face cache paths under Drive.
3. Clones or pulls `https://github.com/liuwen/qwen-asr-eval.git` into `/content/qwen-asr-eval`.
4. Installs a minimal dependency set for smoke testing.
5. Uses `HF_TOKEN` if available. Under `colab exec`, Colab Secrets may be unavailable, so absence is non-fatal by default for public model downloads.
6. Resolves a Xiaoyuzhou episode audio URL. It tries RSSHub first, then falls back to scraping the public episode page for embedded media URLs.
7. Downloads the source audio only if missing.
8. Transcodes the first 45 seconds to 16 kHz mono WAV.
9. Splits the sample into timestamped chunks.
10. Downloads/loads Qwen3-ASR.
11. Runs Qwen ASR on one small chunk.
12. Runs `google.colab.ai` text-only quality triage.
13. Writes JSONL, transcript Markdown/text, and judge JSON into Drive under `runs/<run_id>/outputs/`.

## Useful environment overrides

You can override defaults when needed:

```bash
SMOKE_SECONDS=60 \
SMOKE_AUDIO_ID=english_clear_aee_2618 \
colab exec -s qwen-asr-a100 --file scripts/colab_smoke_xiaoyuzhou.py
```

Supported variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PUBLIC_REPO_URL` | `https://github.com/liuwen/qwen-asr-eval.git` | Public repo to clone/pull |
| `PUBLIC_REPO_BRANCH` | `main` | Branch to checkout |
| `REPO_DIR` | `/content/qwen-asr-eval` | Ephemeral clone path |
| `ASR_EVAL_DRIVE_ROOT` | `/content/drive/MyDrive/asr` | Durable root |
| `ASR_EVAL_AUDIO_ROOT` | `$ASR_EVAL_DRIVE_ROOT/audio` | Durable audio root |
| `ASR_EVAL_RUNS_ROOT` | `$ASR_EVAL_DRIVE_ROOT/qwen-asr-eval/runs` | Durable output runs root |
| `ASR_EVAL_HF_CACHE` | `$ASR_EVAL_DRIVE_ROOT/hf_cache` | Durable HF cache |
| `ASR_EVAL_TMP_ROOT` | `/content/asr-eval-work` | Ephemeral scratch root |
| `SMOKE_AUDIO_ID` | `mixed_zh_en_wujimacha_22e01` | Manifest item to test |
| `SMOKE_SECONDS` | `45` | Seconds to transcode for smoke ASR |
| `SMOKE_CHUNK_SECONDS` | `45` | Chunk length for smoke test |
| `ASR_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Qwen ASR model id |
| `COLAB_AI_MODEL` | `google/gemini-2.5-flash` | Colab AI text judge model |
| `ASR_EVAL_REQUIRE_HF_TOKEN` | `0` | Set `1` to fail if `HF_TOKEN` is unavailable under CLI execution |

## Google Drive account notes

A Colab runtime can mount the Google Drive account that completes the OAuth authorization. With the CLI, the auth URL may include a `login_hint` for the Colab-authenticated account. In practice, use the same Google account for Colab CLI, Colab web UI, and Drive when possible.

If audio or outputs live in another Google account, recommended options are:

1. Share that Drive folder with the Colab account and add a shortcut under `MyDrive/asr`, or
2. Run the Colab CLI/web UI authenticated as that other account, or
3. Download/upload through another storage path explicitly instead of relying on Drive mount.

Do not assume two Google accounts' Drives are both locally mounted at the same time under `/content/drive/MyDrive`.

## Evaluation honesty

`google.colab.ai` receives transcript text only. It cannot listen to audio and cannot verify faithfulness. It is only used to flag text-level risks such as repetition, language mismatch, hallucination-like boilerplate, or disagreement with a baseline when present.
