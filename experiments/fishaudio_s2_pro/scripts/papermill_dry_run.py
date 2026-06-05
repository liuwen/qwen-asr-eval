#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import wave
from pathlib import Path

import papermill as pm
from ipykernel.kernelspec import install as install_ipykernel


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "fishaudio_s2_pro_colab.ipynb"
DEFAULT_OUTPUT = ROOT / "runs" / "papermill" / "fishaudio_s2_pro_colab.dry_run.ipynb"


def make_fake_repo(path: Path) -> None:
    required = [
        "pyproject.toml",
        "tools/run_webui.py",
        "tools/api_server.py",
        "tools/server/views.py",
        "awesome_webui/package.json",
        "awesome_webui/src/App.tsx",
        "fish_speech/inference_engine/__init__.py",
        "fish_speech/models/text2semantic/inference.py",
    ]
    for rel in required:
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("mock\n", encoding="utf-8")
    (path / "tools" / "run_webui.py").write_text("max_new_tokens=1024,\n", encoding="utf-8")
    (path / "tools" / "server" / "model_manager.py").parent.mkdir(parents=True, exist_ok=True)
    (path / "tools" / "server" / "model_manager.py").write_text("max_new_tokens=1024,\n", encoding="utf-8")
    (path / "awesome_webui" / "src" / "App.tsx").write_text("  maxNewTokens: 2048,\n", encoding="utf-8")
    (path / "fish_speech" / "inference_engine" / "__init__.py").write_text(
        "import gc\n"
        "        if torch.cuda.is_available():\n"
        "            torch.cuda.empty_cache()\n"
        "            gc.collect()\n",
        encoding="utf-8",
    )
    (path / "fish_speech" / "models" / "text2semantic" / "inference.py").write_text(
        "import os\n"
        "        with sdpa_kernel(SDPBackend.MATH):\n"
        "                if torch.cuda.is_available():\n"
        "                    torch.cuda.empty_cache()\n",
        encoding="utf-8",
    )
    dist = path / "awesome_webui" / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "index.html").write_text("<html></html>\n", encoding="utf-8")


def write_silent_wav(path: Path, *, sample_rate: int = 16000, seconds: float = 0.25) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytes(max(2, int(sample_rate * seconds) * 2))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)


def install_current_python_kernel(kernel_root: Path) -> str:
    kernel_name = "fishaudio-papermill-dry-run"
    prefix = kernel_root / "jupyter-prefix"
    install_ipykernel(
        user=False,
        prefix=str(prefix),
        kernel_name=kernel_name,
        display_name=f"Fish Audio dry run ({Path(sys.executable).name})",
    )
    jupyter_path = str(prefix / "share" / "jupyter")
    existing = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = jupyter_path if not existing else os.pathsep.join([jupyter_path, existing])
    return kernel_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the Fish Audio Colab notebook through Papermill in dry-run mode.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output notebook path.")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep the temporary fake Colab workspace.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fishaudio-papermill-", delete=not args.keep_tmp) as tmp:
        tmp_path = Path(tmp)
        fish_repo = tmp_path / "fish-speech"
        work_root = tmp_path / "content" / "fishaudio_s2_pro"
        drive_root = tmp_path / "drive" / "voice" / "fishaudio-s2-pro"
        reference_audio = tmp_path / "inputs" / "reference.wav"
        make_fake_repo(fish_repo)
        write_silent_wav(reference_audio)

        kernel_name = install_current_python_kernel(tmp_path)
        parameters = {
            "DRY_RUN": True,
            "MOUNT_DRIVE": False,
            "FISH_REPO_PATH": str(fish_repo),
            "WORK_ROOT_PATH": str(work_root),
            "ARTIFACT_ROOT_PATH": str(drive_root),
            "DRIVE_ROOT_PATH": str(drive_root),
            "RUN_ID": "papermill_dry_run",
            "REFERENCE_AUDIO_PATH": str(reference_audio),
            "REFERENCE_TEXT": "Dry run reference transcript.",
            "VOICE_ID": "dry_run_voice",
            "BUILD_AWESOME_WEBUI": False,
            "START_GRADIO_WEBUI": False,
            "PUBLIC_TUNNEL_MODE": "none",
        }

        print("Executing Papermill dry run with parameters:")
        print(json.dumps({k: v for k, v in parameters.items() if "TOKEN" not in k}, indent=2, sort_keys=True))
        pm.execute_notebook(
            str(NOTEBOOK),
            str(args.output),
            parameters=parameters,
            kernel_name=kernel_name,
            progress_bar=True,
            log_output=True,
        )

        manifest_path = drive_root / "runs" / "papermill_dry_run" / "manifests" / "final_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"Papermill dry run completed but final manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("dry_run"):
            raise RuntimeError(f"Papermill manifest did not record dry_run=true: {manifest_path}")

        print("Papermill dry run output notebook:", args.output)
        print("Papermill final manifest:")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.keep_tmp:
            print("Temporary workspace kept:", tmp_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
