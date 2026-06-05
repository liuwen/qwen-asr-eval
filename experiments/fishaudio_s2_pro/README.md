# Fish Audio S2 Pro Colab Experiment

This experiment evaluates Fish Audio S2 Pro for TTS and authorized voice cloning on Colab GPU runtimes. The current notebook is API-first: start one model engine, run fast API inference, optionally expose it, and avoid reloading the model for demos.

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

Optional API bearer token. If absent, the notebook generates a session token:

```text
FISHAUDIO_API_KEY
```

No `GH_TOKEN`, `GEMINI_API_KEY`, `MODEL_PROXY_API_KEY`, or Colab GenAI path is required.

## Notebook

Open or upload:

```text
experiments/fishaudio_s2_pro/fishaudio_s2_pro_colab.ipynb
```

The notebook clones `https://github.com/fishaudio/fish-speech.git` directly, checks out the pinned `FISH_SPEECH_REF`, installs the upstream Fish Speech environment with `uv`, downloads `fishaudio/s2-pro`, starts one long-lived authenticated API server, and runs a minimal TTS showcase through that API.

There is no `PUBLIC_REPO_URL`, no `PUBLIC_REPO_BRANCH`, and no submodule setup in the notebook.

## Storage Policy

Model artifacts from Hugging Face must stay ephemeral under `/content`:

```text
/content/fishaudio_s2_pro/checkpoints/s2-pro
/content/fishaudio_s2_pro/cache
```

Google Drive is not mounted by default. Generated artifacts stay ephemeral unless explicitly downloaded:

```text
/content/fishaudio_s2_pro/artifacts/runs/<run_id>/
  logs/
  outputs/
  manifests/
```

The notebook uses `google.colab.files.upload()` for optional reference audio and `google.colab.files.download()` for output files or a zipped run directory. To upload reference audio for cloning, use cell 7: set `UPLOAD_REFERENCE_AUDIO=True`, fill `REFERENCE_TEXT`, run the cell, then run the reference TTS cell. If `MOUNT_DRIVE=True`, only our own artifacts move to Drive; HF model files still remain under `/content`.

Reference audio copies are not written to artifacts by default. The notebook exposes `SAVE_REFERENCE_COPY_TO_ARTIFACTS` for cases where a durable private copy is explicitly wanted.

The HF model download cell writes CLI output to:

```text
logs/hf_download.log
```

If Colab restarts during download, inspect that log and lower `HF_DOWNLOAD_WORKERS`
in the notebook. The default is intentionally conservative (`2`) to reduce
kernel pressure.

## Upstream Findings

The Fish Speech repo currently has several behaviors that matter on Colab:

- `tools/api_server.py` creates one `TTSInferenceEngine` through `ModelManager`.
- `tools/run_webui.py` creates a separate `TTSInferenceEngine`; starting Gradio after the API server reloads the full model.
- `awesome_webui` is only a frontend. It calls `fetch('/v1/tts')` and is served by `tools/api_server.py` at `/ui` after `npm run build`.
- Both API and Gradio warm up with `max_new_tokens=1024`; this is heavy for a startup smoke.
- `fish_speech/models/text2semantic/inference.py` forces `SDPBackend.MATH` inside the decode loop, which is a poor default for A100-class CUDA attention.
- The inference path clears CUDA cache after requests, which is useful defensively but adds overhead for repeated short calls.

The notebook applies local runtime patches to the cloned Fish Speech tree, not to the vendored submodule:

```text
allow_flash_attention_backends
guard_text2semantic_cuda_cache_clear
guard_engine_cuda_cache_clear
light_warmup_tools/server/model_manager.py
light_warmup_tools/run_webui.py
awesome_default_max_tokens
```

These patches are recorded in each run manifest.

## API Shape

The upstream API builds a `ServeTTSRequest` from:

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
tools/api_server.py                     # Kui/uvicorn API server
tools/server/views.py                   # /ui route for Awesome WebUI
awesome_webui/src/App.tsx               # frontend calls /v1/tts
tools/run_webui.py                      # Gradio WebUI, separate model engine
```

## Cloudflare Tunnel

The API server starts with bearer auth when `ENABLE_API_AUTH=True`. The notebook can expose the same local API server with either a quick or named Cloudflare Tunnel:

```text
PUBLIC_TUNNEL_MODE = none | cloudflare_quick | cloudflare_named
```

Quick tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:7860 --no-autoupdate
```

Named tunnel:

```bash
cloudflared tunnel --no-autoupdate run --token <CLOUDFLARED_TUNNEL_TOKEN>
```

For named tunnels, configure the Cloudflare origin as:

```text
http://127.0.0.1:7860
```

The notebook does not print the Cloudflare tunnel token. It prints the generated API bearer token only when the token was generated in-session or typed as a parameter; tokens loaded from Colab Secrets are not printed.

## Web UI

The speed path is API-only. `START_GRADIO_WEBUI=False` by default because Gradio starts another full model engine.

The API server defaults to port `7860` for Colab compatibility. If Gradio is explicitly enabled, the notebook starts it on `7861` to avoid colliding with the API server.

`BUILD_AWESOME_WEBUI=True` can build and serve `/ui` from the existing API server. The code is in the cloned Fish Speech checkout at `awesome_webui/src/App.tsx`, and the built file is served from `awesome_webui/dist/index.html` through `tools/server/views.py`. Upstream bearer auth also protects the initial `/ui` page load, so for a browser UI either disable upstream API auth and put Cloudflare Access/Tailscale in front, or stay with the authenticated API-only tunnel.

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
