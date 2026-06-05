# Fish Audio S2 Pro Colab Experiment

This experiment evaluates Fish Audio S2 Pro for TTS and authorized voice cloning on Colab GPU runtimes.

The upstream reference implementation is vendored as a submodule:

```text
vendor/fish-speech -> https://github.com/fishaudio/fish-speech
```

This repo owns the helper package, notebook workflow, run manifests, reference preparation, and future custom Colab API wrapping. The upstream repo is used to understand and run the model, API server, and WebUI behavior.

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

The notebook clones this public repo with submodules, installs this helper project with `uv`, installs the upstream Fish Speech environment with `uv`, downloads `fishaudio/s2-pro`, runs API smoke tests, and can launch the upstream web demos.

If this branch has not been merged, keep:

```text
PUBLIC_REPO_BRANCH = "exp/fishaudio"
```

After merge, change it to the branch that contains this experiment.

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

The helper package preserves this flow so a future custom wrapped HTTP API can target the same concepts without depending on Gradio.

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

Do not edit `vendor/fish-speech` for helper behavior. Keep custom code in `src/fishaudio_s2_pro/`.

## References

- Fish Speech GitHub: https://github.com/fishaudio/fish-speech
- Fish installation docs: https://speech.fish.audio/install/
- Fish inference docs: https://speech.fish.audio/inference/
- Fish server docs: https://speech.fish.audio/server/
- S2 Pro weights: https://huggingface.co/fishaudio/s2-pro
