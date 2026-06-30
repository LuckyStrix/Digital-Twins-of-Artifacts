#!/usr/bin/env python3
"""
run.py
======
Single launcher that ties the four Papyrus sub-projects together:

  backend/capture    - Arduino + camera capture rig (gphoto2/dcraw). Drives the
                        lighting rig and shoots the directional-light TIFFs into
                        the top-level data/<timestamp>/ folder. Requires the
                        physical hardware (camera over USB + Arduino on a serial
                        port).

  backend/modeling   - Python photometric-stereo pipeline.
                        Reads the active folder's scroll scans and writes the
                        render-ready maps into <active folder>/maps/.

  backend/rendering  - Three.js / Vite app. Turns the texture maps into a
                        textured 3D model, saved to <active folder>/model/render.glb.

  backend/website    - Interactive museum-style viewer for render.glb.

The "active working folder" holds one set of scroll scans. The pipeline stores
its outputs inside that same folder, so each scan set is self-contained:

  <active folder>/
    allLight.tiff, ncross.tiff, ... wco.tiff   (the 9 scroll scans)
    maps/        *_render.tiff                  (modeling output)
    model/       render.glb                     (rendering output)

It defaults to the top-level data/ folder (where captures land); a fresh capture
makes its own data/<timestamp>/ folder active, and "Select working image set"
points it at any other scan folder.

Usage:
    python3 run.py

Then click the buttons top to bottom:
     Select working image set - choose which scan folder the pipeline reads from
                              and writes into. This only switches the active
                              folder; nothing runs until you click a step or
                              "Run Everything".
     Open Focus Viewer      - turns on all four lights and shows a live camera
                              preview so you can set focus. Nothing is saved.
  0. Capture Calibration    - shoots the lighting sequence on flat copy paper and
                              stores it straight into backend/calibration/.
                              Done once for a given rig setup.
  1. Capture Scroll         - shoots the scroll TIFFs with the hardware rig
                              (needs the camera + Arduino attached) and stores
                              them in the active working folder.
  2. Run Modeling Pipeline  - generates the texture maps into <active>/maps/.
  3. Build 3D Model (.glb)  - bakes the maps into <active>/model/render.glb.
  4. Open 3D Viewer         - serves backend/website/ and opens it in a browser.

The individual steps 1-4 always run (force re-do). "Run Everything" and "Select
working image set" run the same pipeline but skip stages whose outputs already
exist in the active folder. Run the calibration capture (step 0) once beforehand;
it depends on the physical rig.
"""

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, scrolledtext

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOTER        = Path(__file__).resolve().parent.parent
ROOT          = Path(__file__).resolve().parent
BACKEND       = ROOT / "backend"
CAPTURE_DIR   = BACKEND / "capture"
MODELING_DIR  = BACKEND / "modeling"
RENDERING_DIR = BACKEND / "rendering"
#WEBSITE_DIR   = BACKEND / "website"

CAPTURE_SCRIPT     = CAPTURE_DIR / "arduinoIntegration.py"
FOCUS_SCRIPT       = CAPTURE_DIR / "focusViewer.py"
CAPTURE_DATA_DIR   = ROOTER / "data"
CALIBRATION_IMAGES = BACKEND / "calibration"

TEXTURES_DIR  = RENDERING_DIR / "public" / "textures"

# Files the capture stage produces and the modeling stage consumes.
CAPTURE_TIFFS = [
    "allLight.tiff",
    "ncross.tiff", "ecross.tiff", "scross.tiff", "wcross.tiff",
    "nco.tiff", "eco.tiff", "sco.tiff", "wco.tiff",
]

# Per-working-folder layout. The scroll scans live directly in the working
# folder; the pipeline writes its render-ready maps and the built model into
# these subfolders inside that same folder, so each scan set is self-contained.
MAPS_SUBDIR  = "maps"
MODEL_SUBDIR = "model"
GLB_NAME     = "render.glb"

# The render-ready maps the modeling pipeline produces (Stage 4 output) and the
# renderer consumes. Used both to detect "maps already present" and to copy them
# into the renderer.
RENDER_MAPS = [
    "DiffuseMap_render.tiff",
    "NormalMap_render.tiff",
    "SpecularMap_render.tiff",
    "RoughnessMap_render.tiff",
    "HeightMap_render.tiff",
    "AlphaMask_render.tiff",
]

VITE_PORT = 5173
HTTP_PORT = 8000

# msys2 install that hosts gphoto2 for the capture stage (kept in sync with
# arduinoIntegration.py / focusViewer.py).
MSYS2_SHELL = r"C:\msys64\msys2_shell.cmd"

# Python packages the pipeline needs, as (pip name, import name, used by).
# pillow is also listed in modeling/requirements.txt; pyserial is capture-only
# and is installed alongside the requirements file.
PYTHON_DEPS = [
    ("pyserial",                "serial",       "capture"),
    ("numpy",                   "numpy",        "modeling"),
    ("opencv-python-headless",  "cv2",          "modeling"),
    ("scipy",                   "scipy",        "modeling"),
    ("pillow",                  "PIL",          "modeling + focus viewer"),
    ("tifffile",                "tifffile",     "modeling"),
    ("imagecodecs",             "imagecodecs",  "modeling"),
    ("rembg",                   "rembg",        "modeling (alpha mask)"),
    ("onnxruntime",             "onnxruntime",  "modeling (alpha mask)"),
]


class PipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Papyrus Scroll → 3D Model Pipeline")
        self.root.geometry("820x600")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.busy = False
        self.vite_proc = None
        self.http_proc = None
        self.buttons = []   # all clickable buttons, disabled while a step runs

        # The "active working folder" holds one scroll-scan set and, after
        # processing, its own maps/ and model/ subfolders. Defaults to the
        # top-level data/ folder, where captured scan sets live; the user picks a
        # specific scan folder inside it with "Select working image set".
        CAPTURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.active_dir = CAPTURE_DATA_DIR

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_log_queue)

        self.log("Ready.")
        self.log(f"Active working folder : {self.active_dir}")
        self.log("Use 'Select working image set' to point at a different scan "
                  "folder, or run the steps to use the one above.")

    # ── UI setup ────────────────────────────────────────────────────────────
    def _build_ui(self):
        header = tk.Label(
            self.root,
            text="Papyrus Scroll → 3D Model Pipeline",
            font=("TkDefaultFont", 14, "bold"),
        )
        header.pack(pady=(10, 0))

        subtitle = tk.Label(
            self.root,
            text=(
                "0) Capture calibration   "
                "1) Capture scroll   "
                "2) Run the modeling pipeline   "
                "3) Build the 3D model   "
                "4) View it"
            ),
            font=("TkDefaultFont", 9),
            fg="#555555",
        )
        subtitle.pack(pady=(2, 6))

        # ── Working-folder mini section ──────────────────────────────────────
        active_frame = tk.Frame(self.root, relief=tk.GROOVE, borderwidth=1)
        active_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        path_row = tk.Frame(active_frame)
        path_row.pack(fill=tk.X, padx=8, pady=(6, 2))
        tk.Label(path_row, text="Working folder:", anchor="w",
                 font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT)
        self.active_dir_var = tk.StringVar(value=str(self.active_dir))
        tk.Label(path_row, textvariable=self.active_dir_var, anchor="w",
                 fg="#1a5fb4").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        select_btn = tk.Button(active_frame,
                               text="Select working image set (choose the active scan folder)",
                               command=self.on_select_folder, anchor="w")
        select_btn.pack(fill=tk.X, padx=8, pady=(2, 8))
        self.buttons.append(select_btn)

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10)

        def add_button(text, command, **kwargs):
            b = tk.Button(button_frame, text=text, command=command, anchor="w", **kwargs)
            b.pack(fill=tk.X, pady=2)
            self.buttons.append(b)
            return b

        add_button("Check / Install Dependencies", self.on_install_deps)
        add_button("Open Focus Viewer (set camera focus)", self.on_open_focus_viewer)
        add_button("Step 0 — Capture Calibration (flat copy paper)", self.on_capture_calibration)
        tk.Frame(button_frame, height=1, bg="#cccccc").pack(fill=tk.X, pady=6)
        add_button("Step 1 — Capture Scroll (needs camera + Arduino)", self.on_run_capture)
        add_button("Step 2 — Run Modeling Pipeline", self.on_run_modeling)
        add_button("Step 3 — Build 3D Model (.glb)", self.on_build_model)
        #add_button("Step 4 — Open 3D Viewer", self.on_open_viewer)
        tk.Frame(button_frame, height=1, bg="#cccccc").pack(fill=tk.X, pady=6)
        add_button("Run Everything (skips stages already done)", self.on_run_all,
                   font=("TkDefaultFont", 9, "bold"))

        # Log area
        log_frame = tk.Frame(self.root)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        tk.Label(log_frame, text="Log", anchor="w").pack(fill=tk.X)
        self.text = scrolledtext.ScrolledText(
            log_frame, height=18, state=tk.DISABLED, wrap=tk.WORD,
            font=("TkFixedFont", 9),
        )
        self.text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Idle")
        status = tk.Label(self.root, textvariable=self.status_var, anchor="w",
                           relief=tk.SUNKEN)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Logging helpers (thread-safe) ─────────────────────────────────────────
    def log(self, message: str):
        self.log_queue.put(message)

    def _poll_log_queue(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.text.configure(state=tk.NORMAL)
            self.text.insert(tk.END, message + "\n")
            self.text.see(tk.END)
            self.text.configure(state=tk.DISABLED)
        self.root.after(100, self._poll_log_queue)

    def _set_status(self, text: str):
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_buttons_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED

        def apply():
            for b in self.buttons:
                b.configure(state=state)

        self.root.after(0, apply)

    # ── Background task runner ────────────────────────────────────────────────
    def run_in_background(self, fn, label: str):
        if self.busy:
            self.log("Busy — please wait for the current step to finish.")
            return

        def wrapper():
            self.busy = True
            self._set_status(f"Running: {label}")
            self._set_buttons_enabled(False)
            try:
                fn()
            except Exception as exc:  # surface any unexpected error in the log
                self.log(f"ERROR: {exc}")
            finally:
                self.busy = False
                self._set_status("Idle")
                self._set_buttons_enabled(True)

        threading.Thread(target=wrapper, daemon=True).start()

    # ── Subprocess helpers ─────────────────────────────────────────────────────
    def run_command(self, cmd, cwd=None, env=None) -> int:
        self.log("$ " + " ".join(str(c) for c in cmd))
        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env,
            )
        except FileNotFoundError as exc:
            self.log(f"ERROR: {exc}")
            return 1

        for line in proc.stdout:
            self.log(line.rstrip("\n"))
        return proc.wait()

    def _drain(self, proc, prefix=""):
        for line in proc.stdout:
            self.log(prefix + line.rstrip("\n"))

    def _terminate(self, proc):
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _wait_for_server(self, url: str, timeout: float = 30) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(url, timeout=1)
                return True
            except Exception:
                time.sleep(0.5)
        return False

    # ── Button handlers ─────────────────────────────────────────────────────────
    def on_install_deps(self):
        self.run_in_background(self.step_install_deps, "dependency check & setup")

    def on_select_folder(self):
        # Just point the pipeline at a different folder; nothing runs until the
        # user clicks a step or "Run Everything".
        if self.busy:
            self.log("Busy — please wait for the current step to finish.")
            return
        start = CAPTURE_DATA_DIR if CAPTURE_DATA_DIR.exists() else ROOT
        folder = filedialog.askdirectory(
            initialdir=str(start), title="Select the image set to work with")
        if not folder:
            self.log("Folder selection cancelled.")
            return
        self.active_dir = Path(folder)
        self.active_dir_var.set(str(self.active_dir))
        self.log(f"Active working folder set to: {self.active_dir}")
        have = []
        have.append("scans" if self._has_scans() else "no scans")
        have.append("maps" if self._has_maps() else "no maps")
        have.append(".glb" if self._has_glb() else "no .glb")
        self.log("  Folder status: " + ", ".join(have) + ".")
        self.log("Click a step or 'Run Everything' to process it.")

    def on_open_focus_viewer(self):
        self.run_in_background(self.step_open_focus_viewer, "focus viewer")

    def on_run_capture(self):
        self.run_in_background(self.step_run_capture, "capture")

    def on_capture_calibration(self):
        self.run_in_background(self.step_capture_calibration, "capture calibration")

    def on_run_modeling(self):
        self.run_in_background(self.step_run_modeling, "modeling pipeline")

    def on_build_model(self):
        self.run_in_background(self.step_build_model, "build 3D model")

    #def on_open_viewer(self):
    #    self.run_in_background(self.step_open_viewer, "open 3D viewer")

    def on_run_all(self):
        self.run_in_background(self.step_run_all, "full pipeline")

    # ── Steps ───────────────────────────────────────────────────────────────────
    def _module_present(self, import_name: str) -> bool:
        """True if `import_name` is importable in this Python (no actual import)."""
        check = ("import importlib.util, sys; "
                 "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)")
        try:
            return subprocess.run(
                [sys.executable, "-c", check, import_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
        except Exception:
            return False

    def step_install_deps(self):
        ok = "[ok]"
        no = "[ -]"
        self.log("=" * 64)
        self.log("DEPENDENCY CHECK & SETUP")
        self.log("=" * 64)

        # 1. Python itself ----------------------------------------------------
        ver = sys.version_info
        self.log(f"Python {ver.major}.{ver.minor}.{ver.micro}  ({sys.executable})")
        if (ver.major, ver.minor) < (3, 9):
            self.log("  WARNING: Python 3.9+ is recommended for this pipeline.")
        try:
            import tkinter  # noqa: F401  (already running, but confirm explicitly)
            self.log(f"{ok} tkinter (GUI) available")
        except Exception:
            self.log(f"{no} tkinter missing — reinstall Python with the Tcl/Tk option")

        # 2. Python packages (pip-installable) --------------------------------
        self.log("\nPython packages:")
        missing = []
        for pip_name, import_name, used_by in PYTHON_DEPS:
            present = self._module_present(import_name)
            self.log(f"  {ok if present else no} {pip_name:<24} ({used_by})"
                     + ("" if present else "  -> will install"))
            if not present:
                missing.append(pip_name)

        if missing:
            self.log(f"\nInstalling {len(missing)} missing package(s) via pip "
                      "(modeling/requirements.txt + pyserial)...")
            req = MODELING_DIR / "requirements.txt"
            base = [sys.executable, "-m", "pip", "install", "-r", str(req), "pyserial"]
            rc = self.run_command(base)
            if rc != 0:
                self.log("Retrying with --break-system-packages "
                          "(this Python looks externally managed)...")
                rc = self.run_command(base[:4] + ["--break-system-packages"] + base[4:])
            self.log("Python packages installed." if rc == 0
                      else f"pip install exited with code {rc}.")
        else:
            self.log("\nAll Python packages already present — nothing to install.")

        # 3. Node.js + renderer packages (Build 3D Model step) ----------------
        self.log("\nNode.js (needed for 'Build 3D Model'):")
        node = shutil.which("node")
        npm = shutil.which("npm")
        if node and npm:
            self.log(f"  {ok} node ({node})")
            if (RENDERING_DIR / "node_modules").exists():
                self.log(f"  {ok} renderer node_modules already installed")
            else:
                self.log("  node_modules missing — running 'npm install' "
                          "(downloads three.js + a headless Chromium; may take a while)...")
                rc = self.run_command(["npm", "install"], cwd=RENDERING_DIR)
                self.log("  npm install complete." if rc == 0
                          else f"  npm install exited with code {rc}.")
        else:
            self.log(f"  {no} Node.js not found on PATH.")
            self.log("       Install the LTS from https://nodejs.org/ (tick 'Add to PATH'),")
            self.log("       reopen this app, then click this button again to run npm install.")

        # 4. Capture-rig tools (only needed for capture / focus viewer) -------
        self.log("\nCapture-rig tools (only needed to run Capture / Focus Viewer):")
        if Path(MSYS2_SHELL).exists():
            self.log(f"  {ok} msys2 ({MSYS2_SHELL})")
        else:
            self.log(f"  {no} msys2 not found at {MSYS2_SHELL}")
            self.log("       Install from https://www.msys2.org/ , then in the MSYS2 MINGW64 shell:")
            self.log("         pacman -S mingw-w64-x86_64-gphoto2")
        dcraw = shutil.which("dcraw")
        if dcraw:
            self.log(f"  {ok} dcraw ({dcraw})")
        else:
            self.log(f"  {no} dcraw not found on PATH — install it / add it to PATH "
                      "(used to convert .cr2 -> .tiff)")
        self.log("       gphoto2 runs inside msys2; make sure it's installed there "
                  "(pacman -S mingw-w64-x86_64-gphoto2).")
        self.log("       The Arduino must be flashed with backend/capture/IrisArduinoCode "
                  "and connected (serial port COM3 by default).")

        self.log("\nDone. Re-run this any time to re-check. See SETUP.md for the full guide.")
        self.log("=" * 64)

    def _latest_capture_dir(self):
        """Most recently modified subfolder of the top-level data/, or None."""
        if not CAPTURE_DATA_DIR.exists():
            return None
        subdirs = [d for d in CAPTURE_DATA_DIR.iterdir() if d.is_dir()]
        if not subdirs:
            return None
        return max(subdirs, key=lambda d: d.stat().st_mtime)

    # ── Active working folder ────────────────────────────────────────────────────
    def _maps_dir(self) -> Path:
        return self.active_dir / MAPS_SUBDIR

    def _glb_path(self) -> Path:
        return self.active_dir / MODEL_SUBDIR / GLB_NAME

    def _set_active_dir(self, path):
        """Set the active working folder and refresh its on-screen display.

        Safe to call from a background thread (the label update is marshalled
        back onto the Tk main thread)."""
        self.active_dir = Path(path)
        self.root.after(0, lambda p=str(self.active_dir): self.active_dir_var.set(p))

    def _has_scans(self, d: Path = None) -> bool:
        d = d or self.active_dir
        return all((d / name).exists() for name in CAPTURE_TIFFS)

    def _has_maps(self, d: Path = None) -> bool:
        d = d or self.active_dir
        maps = d / MAPS_SUBDIR
        return all((maps / name).exists() for name in RENDER_MAPS)

    def _has_glb(self, d: Path = None) -> bool:
        d = d or self.active_dir
        return (d / MODEL_SUBDIR / GLB_NAME).exists()

    def _run_capture_script(self) -> int:
        """Run the shared capture script (Arduino + camera). Returns its exit code."""
        if not CAPTURE_SCRIPT.exists():
            self.log(f"ERROR: capture script not found at {CAPTURE_SCRIPT}")
            return 1
        self.log("Running the capture rig (Arduino + camera)...")
        self.log("This requires the camera connected over USB and the Arduino "
                  "on a serial port, plus gphoto2/dcraw installed.")
        rc = self.run_command([sys.executable, "-u", str(CAPTURE_SCRIPT)], cwd=CAPTURE_DIR)
        if rc != 0:
            self.log(f"Capture exited with code {rc}.")
            self.log("If you saw 'ModuleNotFoundError: serial', install pyserial "
                      "(pip install pyserial). Capture only runs on the machine "
                      "wired to the capture rig.")
        return rc

    def _import_capture_into(self, latest: Path, dest: Path, label: str) -> int:
        """Copy the expected capture TIFFs from `latest` into `dest`. Returns count."""
        self.log(f"Importing capture from {latest}")
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        missing = []
        for name in CAPTURE_TIFFS:
            src = latest / name
            if src.exists():
                shutil.copy2(src, dest / name)
                self.log(f"  {name} -> {label}")
                copied += 1
            else:
                missing.append(name)

        if missing:
            self.log("WARNING: capture folder is missing these expected files: "
                      + ", ".join(missing))
        self.log(f"Imported {copied}/{len(CAPTURE_TIFFS)} images into {label}.")
        return copied

    def step_open_focus_viewer(self):
        if not FOCUS_SCRIPT.exists():
            self.log(f"ERROR: focus viewer not found at {FOCUS_SCRIPT}")
            return
        self.log("Opening the focus viewer (turns on all four lights and shows a "
                  "live camera preview)...")
        self.log("Use 'Capture New Photo' in that window to refresh the preview; "
                  "nothing is saved. Close it when the focus looks right.")
        rc = self.run_command([sys.executable, "-u", str(FOCUS_SCRIPT)], cwd=CAPTURE_DIR)
        if rc == 0:
            self.log("Focus viewer closed.")
        else:
            self.log(f"Focus viewer exited with code {rc}.")

    def step_run_capture(self):
        self.log("SCROLL CAPTURE — place the scroll on the stage before "
                  "continuing.")
        rc = self._run_capture_script()
        if rc != 0:
            return
        latest = self._latest_capture_dir()
        if latest is None:
            self.log("Capture reported success but no output folder was found in "
                      f"{CAPTURE_DATA_DIR}.")
            return
        # The capture script already wrote the scans into data/<timestamp>/,
        # which is a self-contained working folder — make it the active one
        # instead of copying the images somewhere else.
        self._set_active_dir(latest)
        self.log(f"Capture finished. Active working folder set to: {latest}")
        if self._has_scans(latest):
            self.log("Scroll scans are in place. Make sure calibration images "
                      "exist too (Step 0 — Capture Calibration), then run the "
                      "modeling pipeline.")
        else:
            missing = [n for n in CAPTURE_TIFFS if not (latest / n).exists()]
            self.log("WARNING: capture folder is missing: " + ", ".join(missing))

    def step_capture_calibration(self):
        self.log("CALIBRATION CAPTURE — place a sheet of flat copy paper (no "
                  "scroll) on the stage before continuing. The same lighting "
                  "sequence is used as for the scroll.")
        rc = self._run_capture_script()
        if rc != 0:
            return
        latest = self._latest_capture_dir()
        if latest is None:
            self.log("Capture reported success but no output folder was found in "
                      f"{CAPTURE_DATA_DIR}.")
            return
        self.log("Capture finished.")
        copied = self._import_capture_into(
            latest, CALIBRATION_IMAGES, "backend/calibration/")
        if copied:
            self.log("Calibration images are in place. Capture the scroll next "
                      "(Step 0 — Capture Scroll), then run the modeling pipeline.")

    def step_run_modeling(self):
        script = MODELING_DIR / "modeling_pipeline.py"
        if not self._has_scans():
            self.log(f"WARNING: {self.active_dir} is missing some scroll scans.")
            self.log("Capture or select a folder with the full scan set "
                      f"({', '.join(CAPTURE_TIFFS)}) before running the pipeline.")

        maps_dir = self._maps_dir()
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Point the modeling pipeline at the active folder's scans and have it
        # write its render-ready maps straight into <active>/maps/.
        env = dict(os.environ)
        env["PAPYRUS_SCROLL_DIR"] = str(self.active_dir)
        env["PAPYRUS_RENDER_OUT"] = str(maps_dir)

        self.log(f"Running modeling pipeline on {self.active_dir} "
                  "(this can take several minutes)...")
        self.log(f"Render-ready maps -> {maps_dir}")
        rc = self.run_command([sys.executable, "-u", script], cwd=MODELING_DIR, env=env)
        if rc == 0:
            self.log("Modeling pipeline finished successfully.")
        else:
            self.log(f"Modeling pipeline exited with code {rc}.")
            self.log("If you saw 'ModuleNotFoundError', click "
                      "'Install Python dependencies' and try again.")

    def step_build_model(self):
        maps_dir = self._maps_dir()
        if not maps_dir.exists() or not any(maps_dir.glob("*_render.tiff")):
            self.log(f"No render-ready textures found in {maps_dir}")
            self.log("Run 'Step 2 — Run Modeling Pipeline' first.")
            return

        TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
        self.log(f"Copying texture maps from {maps_dir} into the renderer...")
        for f in sorted(maps_dir.glob("*_render.tiff")):
            shutil.copy2(f, TEXTURES_DIR / f.name)
            self.log(f"  {f.name} -> backend/rendering/public/textures/")

        if not (RENDERING_DIR / "node_modules").exists():
            self.log("node_modules missing — running npm install "
                      "(this may take a while)...")
            rc = self.run_command(["npm", "install"], cwd=RENDERING_DIR)
            if rc != 0:
                self.log("npm install failed; aborting.")
                return

        if shutil.which("node") is None:
            self.log("ERROR: 'node' was not found on PATH. Install Node.js "
                      "and try again.")
            return

        vite_js = RENDERING_DIR / "node_modules" / "vite" / "bin" / "vite.js"
        if not vite_js.exists():
            self.log(f"ERROR: {vite_js} not found. Try running npm install.")
            return

        self.log("Starting local Vite server...")
        self.vite_proc = subprocess.Popen(
            ["node", str(vite_js), "--port", str(VITE_PORT), "--strictPort"],
            cwd=str(RENDERING_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._drain, args=(self.vite_proc, "[vite] "),
                          daemon=True).start()

        if not self._wait_for_server(f"http://localhost:{VITE_PORT}", timeout=30):
            self.log("Vite server did not become ready in time.")
            self._terminate(self.vite_proc)
            self.vite_proc = None
            return

        self.log("Building textured model -> render.glb (headless browser)...")
        rc = self.run_command(["node", "export-scene.js"], cwd=RENDERING_DIR)

        self._terminate(self.vite_proc)
        self.vite_proc = None

        if rc != 0:
            self.log("Export failed — see log above for details.")
            return

        glb = RENDERING_DIR / "render.glb"
        if glb.exists():
            dest = self._glb_path()
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(glb, dest)
            #shutil.copy2(glb, WEBSITE_DIR / "render.glb")
            self.log(f"Saved model -> {dest}.")
            #self.log(f"Saved model -> {dest} and copied it into the viewer.")
            #self.log("Click 'Step 4 — Open 3D Viewer' to see the result.")
        else:
            self.log("render.glb was not created — see log above for details.")
    """
    def step_open_viewer(self):
        # Make sure the viewer shows the active folder's model.
        active_glb = self._glb_path()
        if active_glb.exists():
            shutil.copy2(active_glb, WEBSITE_DIR / "render.glb")
            self.log(f"Loading {active_glb} into the viewer.")
        elif not (WEBSITE_DIR / "render.glb").exists():
            self.log("No render.glb found for the active folder — build the model "
                      "first (Step 3).")
            return

        if self.http_proc is None or self.http_proc.poll() is not None:
            self.log(f"Starting local web server on port {HTTP_PORT}...")
            self.http_proc = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(HTTP_PORT)],
                cwd=str(WEBSITE_DIR), stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)

        url = f"http://localhost:{HTTP_PORT}/infoboxesweb2.html"
        self.log(f"Opening {url}")
        webbrowser.open(url)
    """
    def _smart_run(self):
        """Run the pipeline on the active folder, skipping stages already done.

        - scroll scans present  -> skip capture
        - render-ready maps present -> skip modeling
        - render.glb present     -> skip render/build
        """
        self.log(f"Processing working folder: {self.active_dir}")

        if self._has_scans():
            self.log("Found a full set of scroll scans — skipping capture.")
        else:
            self.log("No complete scroll-scan set found — running capture.")
            self.step_run_capture()
            if not self._has_scans():
                self.log("Capture did not produce a full scan set; stopping.")
                return

        if self._has_maps():
            self.log(f"Render-ready maps already in {self._maps_dir()} — "
                      "skipping modeling.")
        else:
            self.step_run_modeling()
            if not self._has_maps():
                self.log("Modeling did not produce the expected maps; stopping.")
                return

        if self._has_glb():
            self.log(f"Model already built ({self._glb_path()}) — skipping render.")
        else:
            self.step_build_model()
            if not self._has_glb():
                self.log("Model build did not produce render.glb; stopping.")
                return

        #self.step_open_viewer()

    def step_run_all(self):
        self._smart_run()

    # ── Cleanup ─────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._terminate(self.vite_proc)
        self._terminate(self.http_proc)
        self.root.destroy()


def main():
    root = tk.Tk()
    PipelineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
