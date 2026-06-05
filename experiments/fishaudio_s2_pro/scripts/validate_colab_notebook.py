#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "fishaudio_s2_pro_colab.ipynb"


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class FakeHTTPResponse:
    status_code = 200
    content = b"RIFFfake-wave"
    text = "ok"


class FakePopen:
    _next_pid = 50000

    def __init__(self, *args: Any, **kwargs: Any):
        self.args = args
        self.kwargs = kwargs
        self.pid = FakePopen._next_pid
        FakePopen._next_pid += 1
        print("[mock subprocess.Popen]", args[0] if args else "")


class FakeRequests(types.SimpleNamespace):
    def post(self, *args: Any, **kwargs: Any) -> FakeHTTPResponse:
        return FakeHTTPResponse()


def load_code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells: list[str] = []
    for index, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"{NOTEBOOK}:cell{index}")
        cells.append(source)
    return cells


def install_colab_mocks() -> None:
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    drive = types.SimpleNamespace(mount=lambda path: print(f"[mock] drive.mount({path})"))
    userdata = types.SimpleNamespace(get=lambda name: f"mock-{name.lower()}")
    colab.drive = drive
    colab.userdata = userdata
    google.colab = colab
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab
    sys.modules["google.colab.drive"] = drive
    sys.modules["google.colab.userdata"] = userdata

    ipython = types.ModuleType("IPython")
    display_mod = types.ModuleType("IPython.display")
    display_mod.Audio = lambda *args, **kwargs: ("Audio", args, kwargs)
    display_mod.display = lambda *args, **kwargs: print("[mock] display")
    ipython.display = display_mod
    sys.modules["IPython"] = ipython
    sys.modules["IPython.display"] = display_mod

    sys.modules["requests"] = FakeRequests()


def make_fake_repo(path: Path) -> None:
    required = [
        "pyproject.toml",
        "tools/run_webui.py",
        "tools/api_server.py",
        "tools/server/views.py",
        "awesome_webui/package.json",
    ]
    for rel in required:
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("mock\n", encoding="utf-8")
    (path / "awesome_webui" / "dist").mkdir(parents=True, exist_ok=True)
    (path / "awesome_webui" / "dist" / "index.html").write_text("<html></html>\n", encoding="utf-8")


def main() -> int:
    os.environ["FISH_NOTEBOOK_VALIDATE"] = "1"
    cells = load_code_cells()
    install_colab_mocks()

    with tempfile.TemporaryDirectory(prefix="fishaudio-notebook-") as tmp:
        tmp_path = Path(tmp)
        fish_repo = tmp_path / "fish-speech"
        work_root = tmp_path / "work"
        drive_root = tmp_path / "drive"
        make_fake_repo(fish_repo)

        env: dict[str, Any] = {
            "__name__": "__main__",
        }

        real_run = subprocess.run
        real_check_output = subprocess.check_output
        real_popen = subprocess.Popen
        real_urlopen = None

        def fake_run(cmd: Any, *args: Any, **kwargs: Any):
            text = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
            print("[mock subprocess.run]", text)
            if "--query-gpu=name,memory.total" in text:
                return subprocess.CompletedProcess(cmd, 0, stdout="NVIDIA A100-SXM4-40GB, 40960\n")
            if text == "nvidia-smi":
                return subprocess.CompletedProcess(cmd, 0, stdout="CUDA Version: 12.8\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="")

        def fake_check_output(cmd: Any, *args: Any, **kwargs: Any):
            text = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
            print("[mock subprocess.check_output]", text)
            return "mock-fish-commit\n"

        import urllib.request

        real_urlopen = urllib.request.urlopen
        urllib.request.urlopen = lambda *args, **kwargs: FakeResponse()
        subprocess.run = fake_run
        subprocess.check_output = fake_check_output
        subprocess.Popen = FakePopen
        try:
            for index, source in enumerate(cells):
                print(f"[mock notebook] executing code cell {index}")
                exec(compile(source, f"{NOTEBOOK}:code-cell-{index}", "exec"), env)
                if index == 0:
                    env["FISH_REPO"] = fish_repo
                    env["WORK_ROOT"] = work_root
                    env["DRIVE_ROOT"] = drive_root
                    env["MOUNT_DRIVE"] = True
                if "CHECKPOINT_DIR" in env:
                    checkpoint = Path(env["CHECKPOINT_DIR"])
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    (checkpoint / "codec.pth").write_text("mock codec\n", encoding="utf-8")
        finally:
            subprocess.run = real_run
            subprocess.check_output = real_check_output
            subprocess.Popen = real_popen
            if real_urlopen is not None:
                urllib.request.urlopen = real_urlopen

    for index, source in enumerate(cells[1:], start=1):
        try:
            exec(compile(source, f"{NOTEBOOK}:isolated-code-cell-{index}", "exec"), {"__name__": "__main__"})
        except RuntimeError as exc:
            if "Notebook bootstrap state is missing" not in str(exc):
                raise
        else:
            raise AssertionError(f"code cell {index} executed without bootstrap guard")

    print("isolated cell bootstrap guards ok")
    print("mock notebook execution ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
