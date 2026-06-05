# Fish Audio S2 Pro Colab Experiment

This experiment evaluates Fish Audio S2 Pro for TTS and authorized voice cloning on Colab GPU runtimes.

The upstream reference implementation is vendored locally for inspection:

```text
vendor/fish-speech -> https://github.com/fishaudio/fish-speech
```

The Colab notebook does not depend on this repo at runtime. It clones Fish Speech directly into `/content/fish-speech`, installs Fish Speech with `uv`, and defines the small runtime helpers inline. This repo owns the notebook and local helper/reference code for future custom API work.

## Required Colab Setup

Use a Colab runtime with Python 3.12 and an A100/H100-class GPU. Fish Audio's inference docs recommend at least 24 GB VRAM for S2 models.

Add these Colab Secrets:

```text
HF_TOKEN
```

Optional for a named Cloudflare Tunnel:

```text
CLOUDFLARED_TUNNEL_TOKEN
```

No `GH_TOKEN`, `GEMINI_API_KEY`, `MODEL_PROXY_API_KEY`, or Colab GenAI path is required.

## Notebook

Open or upload:

```text
experiments/fishaudio_s2_pro/fishaudio_s2_pro_colab.ipynb
```

The notebook clones `https://github.com/fishaudio/fish-speech.git` directly, checks out the pinned `FISH_SPEECH_REF`, installs the upstream Fish Speech environment with `uv`, downloads `fishaudio/s2-pro`, runs API smoke tests, and can launch the upstream web demos.

There is no `PUBLIC_REPO_URL`, no `PUBLIC_REPO_BRANCH`, and no submodule setup in the notebook.

## Storage Policy

Model artifacts from Hugging Face must stay ephemeral under `/content`:

```text
/content/fishaudio_s2_pro/checkpoints/s2-pro
/content/fishaudio_s2_pro/cache
```

Google Drive is only for our own artifacts:

```text
/content/drive/MyDrive/voice/fishaudio-s2-pro/runs/<run_id>/
  logs/
  outputs/
  manifests/
```

Reference audio copies are not written to Drive by default. The notebook exposes `SAVE_REFERENCE_COPY_TO_DRIVE` for cases where a durable private copy is explicitly wanted.

The HF model download cell writes CLI output to:

```text
logs/hf_download.log
```

If Colab restarts during download, inspect that log and lower `HF_DOWNLOAD_WORKERS`
in the notebook. The default is intentionally conservative (`2`) to reduce
kernel pressure.

## Web Demo and API Shape

The upstream Gradio demo builds a `ServeTTSRequest` from:

- input text,
- optional `reference_id`,
- optional inline reference audio plus reference transcript,
- sampling settings such as `chunk_length`, `top_p`, `temperature`, and `repetition_penalty`.

The upstream engine then:

1. loads reference audio from `references/<voice_id>/sample.wav` plus `sample.lab`, or from inline bytes;
2. encodes reference audio into VQ prompt tokens;
3. sends text plus prompt tokens/text to the semantic model queue;
4. decodes generated VQ codes through the DAC decoder;
5. returns generated audio.

The notebook preserves this flow in its inline helper functions so a future custom wrapped HTTP API can target the same concepts without depending on Gradio.

Upstream WebUI/server paths checked against Fish Speech:

```text
tools/run_webui.py                      # Gradio WebUI
tools/api_server.py                     # Kui/uvicorn API server
tools/server/views.py                   # /ui route for Awesome WebUI
awesome_webui/dist/index.html           # built frontend served at /ui
```

## Cloudflare Tunnel

The notebook follows the named tunnel pattern:

```bash
cloudflared tunnel --no-autoupdate run --token <CLOUDFLARED_TUNNEL_TOKEN>
```

Configure the tunnel in Cloudflare so its origin points to the active local web demo:

```text
Gradio WebUI:  http://127.0.0.1:7860
Awesome WebUI: http://127.0.0.1:8888
```

The notebook does not print the tunnel token.

## Local Development

From this experiment directory:

```bash
uv sync --python 3.12
uv run python -m compileall -q src
```

Before pushing notebook changes, run the mocked Colab notebook executor:

```bash
python3 scripts/validate_colab_notebook.py
```

This parses every code cell, executes the notebook in order with Drive/GPU/git/HF/server mocks, and also executes each non-bootstrap cell in isolation to ensure skipped-cell/kernel-restart failures raise the explicit bootstrap error instead of raw `NameError`s.

Then run the real Jupyter/Papermill dry-run path through the uv project:

```bash
uv run python scripts/papermill_dry_run.py
```

This uses the notebook's Papermill `parameters` cell, injects `DRY_RUN=True`, creates a fake Fish Speech checkout plus reference audio in a temporary workspace, and executes the whole notebook with a project-local ipykernel. It writes the executed notebook to:

```text
runs/papermill/fishaudio_s2_pro_colab.dry_run.ipynb
```

That output is ignored by git. Papermill proves real notebook execution and parameter injection; it still intentionally skips Drive auth, model download, GPU model loading, server startup, and tunnels.

Do not edit `vendor/fish-speech` for helper behavior. Keep custom code in `src/fishaudio_s2_pro/`.

## References

- Fish Speech GitHub: https://github.com/fishaudio/fish-speech
- Fish installation docs: https://speech.fish.audio/install/
- Fish inference docs: https://speech.fish.audio/inference/
- Fish server docs: https://speech.fish.audio/server/
- S2 Pro weights: https://huggingface.co/fishaudio/s2-pro
