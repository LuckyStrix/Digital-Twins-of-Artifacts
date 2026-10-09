#!/usr/bin/env python3
"""Check that this machine can run the reconstruction app, GPU included.

The native-Windows counterpart of src/check_config.sh (works on Linux too).
Run it with the venv's Python:

    venv\\Scripts\\python check_setup.py        (Windows)
    venv/bin/python3 check_setup.py           (Linux)

Exits 0 when everything needed is present; prints a fix for anything that
isn't.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from run_stage2 import resolve_colmap, resolve_exiftool  # noqa: E402

GUIDE = r"..\WINDOWS_SETUP.md" if sys.platform == "win32" else "../README.md"
OK, WARN, FAIL = "\x1b[32m  ok  \x1b[0m", "\x1b[33m warn \x1b[0m", "\x1b[31m FAIL \x1b[0m"
failures = 0


def report(level: str, what: str, detail: str = "") -> None:
    global failures
    if level == FAIL:
        failures += 1
    print(f"[{level}] {what}" + (f"\n         {detail}" if detail else ""))


def run(cmd: list[str], timeout: float = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)


def check_python() -> None:
    v = sys.version_info
    if (3, 9) <= v[:2] <= (3, 12):
        report(OK, f"Python {v.major}.{v.minor} ({sys.executable})")
    else:
        report(FAIL, f"Python {v.major}.{v.minor}: need 3.9-3.12 (open3d has no wheels for newer)",
               "Install Python 3.12 for your user and recreate the venv.")
    if sys.prefix == sys.base_prefix:
        report(WARN, "not running inside a virtualenv",
               "The app runs stages with venv\\Scripts\\python.exe if it exists; "
               "run this check with that interpreter.")


def check_modules() -> None:
    for mod, pip_name in [("numpy", "numpy"), ("open3d", "open3d"), ("cv2", "opencv"),
                          ("PIL", "pillow"), ("rembg", "rembg[gpu]"),
                          ("onnxruntime", "onnxruntime-gpu"), ("torch", "torch"),
                          ("textual", "textual")]:
        try:
            m = importlib.import_module(mod)
            report(OK, f"{mod} {getattr(m, '__version__', '')}".rstrip())
        except Exception as exc:
            report(FAIL, f"{mod} won't import: {exc}",
                   f"pip install the requirements again (package: {pip_name}).")


def check_gpu() -> None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        report(FAIL, "nvidia-smi not found: no NVIDIA driver?",
               "The NVIDIA driver is the only GPU piece that must already be on the PC.")
    else:
        rc, out = run([smi, "--query-gpu=name,driver_version,memory.total",
                       "--format=csv,noheader"])
        if rc == 0 and out.strip():
            for line in out.strip().splitlines():
                report(OK, f"GPU: {line.strip()}")
        else:
            report(FAIL, "nvidia-smi failed", out.strip()[:300])

    try:
        import torch
        if torch.cuda.is_available():
            report(OK, f"torch sees CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")
        elif torch.version.cuda is None:
            report(FAIL, "torch is the CPU-only build",
                   "Reinstall it from the CUDA index (see the guide, step 3).")
        else:
            report(FAIL, f"torch is a CUDA {torch.version.cuda} build but can't use the GPU",
                   "Update the NVIDIA driver.")
    except Exception:
        pass  # already reported by check_modules

    try:
        import onnxruntime as ort
        from process_photos import _load_cuda_dlls
        _load_cuda_dlls()
        provs = ort.get_available_providers()
        if "CUDAExecutionProvider" in provs:
            report(OK, "onnxruntime has the CUDA provider (rembg can use the GPU)")
        else:
            report(FAIL, f"onnxruntime is CPU-only ({', '.join(provs)})",
                   "pip uninstall -y onnxruntime onnxruntime-gpu, then "
                   "pip install \"onnxruntime-gpu<1.27\".")
    except Exception:
        pass


def check_colmap() -> None:
    colmap = resolve_colmap(dict(os.environ))
    if not colmap:
        report(FAIL, "COLMAP not found",
               "Unzip colmap-x64-windows-cuda.zip so modelingPipeline\\colmap\\COLMAP.bat "
               "exists, or set FIPMESH_COLMAP_BIN.")
        return
    rc, out = run([colmap, "help"])
    first = next((ln.strip() for ln in out.splitlines() if "COLMAP" in ln), "")
    if rc != 0 and not first:
        report(FAIL, f"COLMAP at {colmap} won't run", out.strip()[:300])
    elif "with CUDA" not in out or "without CUDA" in out:
        report(WARN, f"COLMAP is a CPU-only build: {first or colmap}",
               "Use the *-windows-cuda.zip release, not *-windows-nocuda.zip.")
    else:
        report(OK, f"{first} ({colmap})")


def check_exiftool() -> None:
    exiftool = resolve_exiftool(dict(os.environ))
    if not exiftool:
        report(FAIL, "exiftool not found",
               "Put exiftool.exe (renamed from 'exiftool(-k).exe') and its exiftool_files "
               "folder in modelingPipeline\\tools\\, or set FIPMESH_EXIFTOOL_BIN.")
        return
    rc, out = run([exiftool, "-ver"])
    if rc == 0:
        report(OK, f"exiftool {out.strip()} ({exiftool})")
    else:
        report(FAIL, f"exiftool at {exiftool} won't run", out.strip()[:300])


def main() -> int:
    if sys.platform == "win32":
        os.system("")  # turn on ANSI colour handling in the Windows console
    print("Checking the 3D reconstruction setup...\n")
    check_python()
    check_modules()
    check_gpu()
    check_colmap()
    check_exiftool()
    print()
    if failures:
        print(f"{failures} problem(s) found. See {GUIDE} for the fixes.")
        return 1
    print("All good. Start the app with:  venv\\Scripts\\python app.py"
          if sys.platform == "win32" else "All good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
