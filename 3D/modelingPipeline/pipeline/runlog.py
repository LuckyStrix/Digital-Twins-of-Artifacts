"""Per-run records written next to a session's outputs.

Every run started from the app (one stage or all four) adds to two files in
the session folder:

    pipeline.log        every log line (app messages, commands, subprocess
                        output), timestamped; each run appends under a header
    pipeline_runs.json  {"runs": [...]}, one entry per run: all settings,
                        environment, inputs, and per-stage timing/results

The JSON is rewritten after every stage, so a crash still leaves the stages
that finished. Recording must never break a run: any write error disables
the recorder for the rest of that run.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Mapping

from . import SCRIPT_DIR
from .options import IMAGE_EXTS, STAGE_NAMES
from .paths import SessionPaths, detect_sides
from .platform import venv_python

LOG_NAME = "pipeline.log"
RUNS_NAME = "pipeline_runs.json"
SCHEMA_VERSION = 1

# Packages whose versions are recorded (from the venv the stages run in).
PACKAGES = ("open3d", "numpy", "rembg", "onnxruntime", "onnxruntime-gpu", "torch",
            "opencv-python", "opencv-python-headless", "pillow", "trimesh", "pymeshlab")

# Terminal colour/cursor escapes (onnxruntime, tqdm) that would clutter the file.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# Cap on recorded progress steps per stage, so a chatty parser can't bloat the JSON.
MAX_STEPS = 500


def _now() -> datetime:
    return datetime.now().astimezone()


def _run_quiet(cmd: list[str], cwd: Path | None = None, timeout: float = 10) -> str:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _git_commit() -> str:
    """``git describe`` of the checkout, with ``-dirty`` for tracked changes.

    The dirty check is ``git status`` with optional locks off rather than
    ``describe --dirty``, which rewrites .git/index. In Docker the repo is a
    bind mount shared with the host's git, and an index rewritten with the
    container's stat data makes the host's git rescan every file.
    """
    commit = _run_quiet(["git", "describe", "--always", "--abbrev=10"], cwd=SCRIPT_DIR)
    if not commit:
        return ""
    try:
        r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=10,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError):
        return commit
    if r.returncode == 0 and r.stdout.strip():
        commit += "-dirty"
    return commit


def environment() -> dict:
    """Machine and software the run used (best effort; missing parts omitted)."""
    env: dict = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "app_python": sys.version.split()[0],
    }
    commit = _git_commit()
    if commit:
        env["git_commit"] = commit
    script = ("import importlib.metadata as m, json, sys\n"
              "v = {}\n"
              f"for p in {list(PACKAGES)!r}:\n"
              "    try: v[p] = m.version(p)\n"
              "    except Exception: pass\n"
              "print(json.dumps({'python': sys.version.split()[0], 'packages': v}))")
    out = _run_quiet([venv_python(), "-c", script])
    if out:
        try:
            env["stage_python"] = json.loads(out.splitlines()[-1])
        except ValueError:
            pass
    if shutil.which("nvidia-smi"):
        gpus = _run_quiet(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader"])
        if gpus:
            env["gpus"] = gpus.splitlines()
    return env


def _images(d: Path) -> list[Path]:
    try:
        return sorted(f for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXTS and f.is_file())
    except OSError:
        return []


def _image_size(p: Path) -> list[int] | None:
    try:
        from PIL import Image
        with Image.open(p) as im:
            return list(im.size)
    except Exception:
        return None


def _image_summary(d: Path) -> dict:
    imgs = _images(d)
    info: dict = {"images": len(imgs)}
    if imgs:
        size = _image_size(imgs[0])
        if size:
            info["first_image_size"] = size
    return info


def input_summary(settings: Mapping) -> dict:
    """Image counts (and a sample resolution) per side of the input folder."""
    inp = str(settings.get("input_var", "")).strip()
    if not inp:
        return {}
    root = Path(inp)
    sides = detect_sides(root)
    if sides:
        return {"sides": {side: _image_summary(root / side) for side in sides}}
    return {"flat": _image_summary(root)}


def ply_vertex_count(path: Path) -> int | None:
    """Vertex count from a PLY header, without loading the file."""
    try:
        with open(path, "rb") as f:
            if f.readline().strip() != b"ply":
                return None
            for _ in range(200):
                line = f.readline()
                if not line or line.strip() == b"end_header":
                    return None
                parts = line.split()
                if parts[:2] == [b"element", b"vertex"] and len(parts) == 3:
                    return int(parts[2])
    except (OSError, ValueError):
        pass
    return None


def stage_outputs(idx: int, paths: SessionPaths) -> dict:
    """Size of what a stage produced: the numbers later stages' time scales with."""
    s1, s2 = paths.active_sides()
    out: dict = {}
    if idx == 0:
        out["processed_images"] = len(_images(paths.processed_dir()))
    elif idx == 1:
        for side in [s for s in (s1, s2) if s] or [None]:
            n = ply_vertex_count(paths.colmap_dir(side) / "fused.ply")
            if n is not None:
                out[f"fused_points_{side or 'flat'}"] = n
    elif idx == 2:
        n = ply_vertex_count(paths.merged_ply())
        if n is not None:
            out["merged_points"] = n
    else:
        n = ply_vertex_count(paths.input_ply_for_recon())
        if n is not None:
            out["input_points"] = n
    return out


def _jsonable(settings: Mapping) -> dict:
    return {k: v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
            for k, v in settings.items()}


class RunRecorder:
    """Records one run (``mode`` is "all" or "stage N") into ``session_dir``."""

    # Class-level: every RunRecorder writes the same pipeline_runs.json via a
    # read-modify-write, so the lock must be shared across instances (e.g. two
    # back-to-back runs, or a run's own background environment-collection
    # thread saving late) — a per-instance lock lets concurrent saves clobber
    # each other's entries.
    _save_lock = threading.Lock()

    def __init__(self, session_dir: Path, settings: Mapping, mode: str):
        self.dir = Path(session_dir)
        self.settings = settings
        self.paths = SessionPaths(settings)
        self.mode = mode
        self._lock = threading.Lock()
        self._log = None
        self._t0 = time.monotonic()
        self._stage_t0: dict[int, float] = {}
        self.enabled = True
        self.error = ""
        self.run: dict = {
            "schema": SCHEMA_VERSION,
            "id": uuid.uuid4().hex[:12],
            "mode": mode,
            "started": _now().isoformat(timespec="seconds"),
            "finished": None,
            "duration_s": None,
            "result": "running",
            "session_dir": str(self.dir),
            "settings": _jsonable(settings),
            "stages": [],
        }

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Create the log and write the first record. Environment and input
        details take seconds to collect (git and a venv Python on /mnt/c), so
        they are filled in by a background thread instead of delaying the run."""
        self._guard(self._open_log)
        self._guard(self._save)
        threading.Thread(target=self._collect_details, daemon=True).start()

    def _collect_details(self) -> None:
        env = environment()
        inputs = input_summary(self.settings)
        with self._lock:
            self.run["environment"] = env
            self.run["inputs"] = inputs
        self._guard(self._save)

    def finish(self, result: str) -> None:
        with self._lock:
            self.run["result"] = result
            self.run["finished"] = _now().isoformat(timespec="seconds")
            self.run["duration_s"] = round(time.monotonic() - self._t0, 2)
            self._write_line(f"===== run finished: {result} after "
                             f"{self.run['duration_s']:.1f} s =====")
        self._guard(self._save)
        with self._lock:
            if self._log:
                self._log.close()
                self._log = None

    # ── callbacks (worker thread; log() may also come from the UI thread) ─────
    def log(self, text: str, kind: str) -> None:
        if kind == "info" and text[:1] == "[" and text[9:11] == "] ":
            text = text[11:]                         # drop runner's own [HH:MM:SS]
        text = _ANSI.sub("", text)
        prefix = "$ " if kind == "header" and not text.startswith("$") else ""
        with self._lock:
            self._write_line(f"{_now():%H:%M:%S.%f}"[:12] + f"  {prefix}{text}")

    def stage_state(self, idx: int, state: str, stopped: bool = False) -> None:
        if state == "running":
            with self._lock:
                self._stage_t0[idx] = time.monotonic()
                self.run["stages"].append({
                    "stage": idx + 1,
                    "name": STAGE_NAMES[idx].split(". ", 1)[-1],
                    "started": _now().isoformat(timespec="seconds"),
                    "finished": None,
                    "duration_s": None,
                    "result": "running",
                    "steps": [],
                })
            return
        stage = self._stage(idx)
        if stage is None:
            return
        outputs = stage_outputs(idx, self.paths) if state == "done" else {}
        with self._lock:
            stage["result"] = "stopped" if stopped and state == "failed" else state
            stage["finished"] = _now().isoformat(timespec="seconds")
            stage["duration_s"] = round(time.monotonic() - self._stage_t0[idx], 2)
            if outputs:
                stage["outputs"] = outputs
        self._guard(self._save)

    def progress(self, idx: int, pct: int, text: str) -> None:
        stage = self._stage(idx)
        if stage is None:
            return
        steps = stage["steps"]
        if (steps and steps[-1]["text"] == text) or len(steps) >= MAX_STEPS:
            return                                   # only record changes of step
        with self._lock:
            steps.append({"t": round(time.monotonic() - self._stage_t0[idx], 2),
                          "pct": pct, "text": text})

    # ── internals ─────────────────────────────────────────────────────────────
    def _stage(self, idx: int) -> dict | None:
        for st in reversed(self.run["stages"]):
            if st["stage"] == idx + 1:
                return st
        return None

    def _guard(self, fn) -> None:
        if not self.enabled:
            return
        try:
            fn()
        except OSError as e:
            self.enabled = False
            self.error = str(e)

    def _open_log(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = open(self.dir / LOG_NAME, "a", encoding="utf-8", errors="replace")
        self._write_line(f"\n===== run started {self.run['started']} ({self.mode}) =====")

    def _write_line(self, line: str) -> None:
        if not (self.enabled and self._log):
            return
        try:
            self._log.write(line + "\n")
            self._log.flush()
        except OSError as e:
            self.enabled = False
            self.error = str(e)

    def _save(self) -> None:
        # Serialised, so the last write always holds the newest snapshot.
        with self._save_lock:
            self._save_locked()

    def _save_locked(self) -> None:
        path = self.dir / RUNS_NAME
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs = data["runs"] if isinstance(data, dict) and isinstance(data.get("runs"), list) else []
        except (OSError, ValueError):
            runs = []
        with self._lock:
            entry = json.loads(json.dumps(self.run))            # snapshot under the lock
        # Replace this run's earlier snapshot, or append.
        runs = [r for r in runs if not (isinstance(r, dict) and r.get("id") == entry["id"])]
        runs.append(entry)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
        os.replace(tmp, path)
