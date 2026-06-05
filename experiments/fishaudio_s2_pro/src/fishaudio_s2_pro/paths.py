from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


def default_run_id(prefix: str = "fishaudio_s2_pro") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def resolve_project_root(project_root: str | Path) -> Path:
    root = Path(project_root)
    if (root / "vendor" / "fish-speech").exists():
        return root

    nested = root / "experiments" / "fishaudio_s2_pro"
    if (nested / "vendor" / "fish-speech").exists():
        return nested

    return root


@dataclass(frozen=True)
class ExperimentPaths:
    """Runtime paths for the Colab experiment.

    Model/cache paths intentionally live under ephemeral `/content` by default.
    Only `run_dir` and its children should be under Google Drive.
    """

    project_root: Path
    fish_repo: Path
    work_root: Path
    checkpoint_dir: Path
    cache_root: Path
    drive_root: Path
    run_id: str
    run_dir: Path
    logs_dir: Path
    outputs_dir: Path
    manifests_dir: Path

    @classmethod
    def for_colab(
        cls,
        *,
        project_root: str | Path = "/content/qwen-asr-eval/experiments/fishaudio_s2_pro",
        work_root: str | Path = "/content/fishaudio_s2_pro",
        drive_root: str | Path = "/content/drive/MyDrive/voice/fishaudio-s2-pro",
        run_id: str | None = None,
    ) -> "ExperimentPaths":
        project_root = resolve_project_root(project_root)
        work_root = Path(work_root)
        drive_root = Path(drive_root)
        run_id = run_id or os.environ.get("FISHAUDIO_RUN_ID") or default_run_id()
        run_dir = drive_root / "runs" / run_id
        return cls(
            project_root=project_root,
            fish_repo=project_root / "vendor" / "fish-speech",
            work_root=work_root,
            checkpoint_dir=work_root / "checkpoints" / "s2-pro",
            cache_root=work_root / "cache",
            drive_root=drive_root,
            run_id=run_id,
            run_dir=run_dir,
            logs_dir=run_dir / "logs",
            outputs_dir=run_dir / "outputs",
            manifests_dir=run_dir / "manifests",
        )

    def ensure(self) -> "ExperimentPaths":
        for path in [
            self.work_root,
            self.checkpoint_dir,
            self.cache_root,
            self.drive_root,
            self.run_dir,
            self.logs_dir,
            self.outputs_dir,
            self.manifests_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    def hf_env(self) -> dict[str, str]:
        hf_home = self.cache_root / "hf"
        return {
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }

    def manifest(self) -> dict[str, str]:
        data = asdict(self)
        return {key: str(value) for key, value in data.items()}
