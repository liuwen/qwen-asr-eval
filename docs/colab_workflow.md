# Colab Workflow

## 1. Start an A100 session

```bash
colab new -s qwen-asr-a100 --gpu A100
colab url -s qwen-asr-a100 --open
```

## 2. Add Colab secret

In the Colab UI, add:

```text
HF_TOKEN
```

Use a normal Hugging Face token for model downloading. The notebook does not use `GH_TOKEN` because the repo is public.

## 3. Upload audio to Drive

Recommended Drive structure:

```text
MyDrive/asr/audio/
  podcast_001.m4a
  zh_sample.wav
  mixed_zh_en.mp3

MyDrive/asr/qwen-asr-eval/
  runs/
```

The notebook mounts Drive and lets you configure:

```text
DRIVE_AUDIO_DIR
DRIVE_PROJECT_ROOT
AUDIO_PATH
AUDIO_GLOB
AUDIO_FILE_INDEX
```

## 4. Push repo and clone from notebook

After you push this starter repo publicly, set:

```text
PUBLIC_REPO_URL = "https://github.com/<you>/qwen-asr-eval.git"
```

The notebook clones the repo and installs it editable:

```bash
pip install -e /content/qwen-asr-eval
```

## 5. Evaluation modes

### Reference transcript available

Use `REFERENCE_TEXT_PATH` and compute WER/CER.

### No reference transcript

Run Qwen plus optional Whisper baseline. Then run Colab AI text-only triage to find risky chunks.

### Mixed Chinese-English

Use:

```text
USE_CASE = "mixed"
QWEN_LANGUAGE = None
WHISPER_LANGUAGE = None
```

This lets each model handle language detection/code-switching.

## 6. Colab AI judge caveat

`google.colab.ai` is text-to-text only. It cannot evaluate whether a transcript matches the audio. It can only inspect the produced text and compare transcript candidates.

Use its output as a triage layer, not a final ASR quality score.
