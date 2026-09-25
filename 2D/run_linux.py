#!/usr/bin/env python3
"""
run_linux.py
============
Linux build of the Papyrus pipeline launcher (run.py). It behaves identically,
but drives the Linux capture scripts (arduinoIntegration_linux.py /
focusViewer_linux.py) and expects the capture-rig tools — gphoto2 and dcraw — to
be installed natively on PATH (e.g. `sudo apt install gphoto2 dcraw`) rather than
through msys2. The modeling and rendering stages are OS-independent and shared
with run.py.

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
                        textured 3D model, saved to <active folder>.glb.

  backend/website    - Interactive museum-style viewer for render.glb.

The "active working folder" holds one set of scroll scans. The pipeline stores
its outputs inside that same folder, so each scan set is self-contained:

  <active folder>/
    allLight.tiff, ncross.tiff, ... wco.tiff   (the 9 scroll scans)
    maps/        *_render.tiff                  (modeling output)
  <active folder>.glb                           (rendering output, alongside it)

Capture always mints a fresh timestamped folder inside the active folder and
makes it active, so it starts out as the top-level data/ folder and "Select
working image set" points it at any other scan folder.

Optionally you can tick "Scan both sides of the object". Capture then shoots two
scan sets (pausing so you can flip the object) into side1/ and side2/ subfolders
of one working folder, and the modeling and rendering steps run once per side,
each producing its own maps/ and its own .glb next to that side's folder:

  <active folder>/
    side1/  allLight.tiff ... wco.tiff   maps/
    side1.glb
    side2/  allLight.tiff ... wco.tiff   maps/
    side2.glb

When the box is left unticked everything behaves exactly as before (a single
flat scan set with its own maps/ directly in the working folder and its .glb
next to it).

Usage:
    python3 run_linux.py

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
  3. Build 3D Model (.glb)  - bakes the maps into <active>.glb, next to the
                              working folder (side1.glb / side2.glb when both
                              sides were scanned).
  4. Generate Artifact Description (.txt) - prompts for the artifact's name,
                              type and description and writes info.txt into the
                              active working folder.

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
import textwrap
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOTER        = Path(__file__).resolve().parent.parent
ROOT          = Path(__file__).resolve().parent
BACKEND       = ROOT / "backend"
CAPTURE_DIR   = BACKEND / "capture"
MODELING_DIR  = BACKEND / "modeling"
RENDERING_DIR = BACKEND / "rendering"

CAPTURE_SCRIPT     = CAPTURE_DIR / "arduinoIntegration_linux.py"
FOCUS_SCRIPT       = CAPTURE_DIR / "focusViewer_linux.py"
CAPTURE_DATA_DIR   = ROOTER / "data"
CALIBRATION_IMAGES = BACKEND / "calibration"

TEXTURES_DIR  = RENDERING_DIR / "public" / "textures"

# Files the capture stage produces and the modeling stage consumes.
CAPTURE_TIFFS = [
    "allLight.tiff",
    "ncross.tiff", "ecross.tiff", "scross.tiff", "wcross.tiff",
    "nco.tiff", "eco.tiff", "sco.tiff", "wco.tiff",
]

# Sidecar the capture script drops beside the TIFFs recording the dcraw flags
# used, and in particular whether the images are linear or BT.709-encoded. The
# modeling pipeline reads it to decide whether to linearise; when it is absent
# the images are assumed to be BT.709 (the historic dcraw default).
CAPTURE_INFO_NAME = "capture_info.json"

# Calibration lives beside the scan set it belongs to (<working>/cal/), with the
# colour-chart shot in <working>/cal/chart/ and the fitted matrix at
# <working>/cal/ccm.json. Kept in sync with backend/modeling/color_correction.py.
CAL_SUBDIR   = "cal"
CHART_SUBDIR = "chart"
CCM_NAME     = "ccm.json"

# Virtualenv holding color_fit.py's dependencies, looked for beside the repo
# root and then beside 2D/. Separate from the pipeline's environment because
# cv2.mcc needs opencv-contrib while rembg needs plain opencv-python-headless,
# and the two overwrite each other's cv2/ directory. See
# backend/modeling/requirements-colorfit.txt.
COLORFIT_VENV = ".venv-colorfit"
COLORFIT_REQS = MODELING_DIR / "requirements-colorfit.txt"

# Run inside the colour-fit venv by 'Check / Install Dependencies'. Prints the
# space-separated names of whatever color_fit.py needs but cannot import, so an
# empty line means the environment is complete. cv2.mcc is checked as an
# attribute because plain opencv installs a cv2 without it; pip is reported so a
# venv created without it (Debian's python3 with no python3-venv) is rebuilt
# rather than pip-installed into.
COLORFIT_CHECK = """
import importlib.util
missing = [m for m in ("pip", "colour", "tifffile", "imagecodecs", "scipy", "numpy")
           if importlib.util.find_spec(m) is None]
try:
    import cv2
    if not hasattr(cv2, "mcc"):
        missing.append("cv2.mcc")
except Exception:
    missing.append("cv2")
print(" ".join(missing))
"""

# Per-working-folder layout. The scroll scans live directly in the working
# folder and the pipeline writes its render-ready maps into <folder>/maps/, so
# each scan set is self-contained. The built model is saved *next to* the folder
# as <folder>.glb, which is how the website's artifact manifest refers to models
# (e.g. side1.glb / side2.glb, 28-07-26_13-36-25.glb).
MAPS_SUBDIR  = "maps"
# Name the renderer gives its own export inside backend/rendering/ before it is
# copied out to <folder>.glb.
GLB_NAME     = "render.glb"

# Optional two-sided capture. When the user opts to scan both sides of the
# object, the scroll scans (and the maps/ and model/ the pipeline derives from
# them) live in these subfolders of the working folder instead of directly in
# it. Single-sided captures keep the historic flat layout (scans directly in
# the working folder), so nothing changes when only one side is scanned.
SIDE_SUBDIRS = ("side1", "side2")

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

# On Linux the capture-rig tools (gphoto2 / dcraw) are installed natively on
# PATH rather than through msys2, so there is no msys2 shell to point at.

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

# Optional helper script and the artifact-description file the "Generate
# Artifact Description" step writes into the active working folder.
TXT_SCRIPT = BACKEND / "create_artifact_info.py"
TXT_NAME = "info.txt"


class PipelineApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Papyrus Scroll → 3D Model Pipeline (Linux)")
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

        # Whether Capture scans both sides of the object (see SIDE_SUBDIRS).
        self.two_sides_var = tk.BooleanVar(value=False)

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
        select_btn.pack(fill=tk.X, padx=8, pady=(2, 4))
        self.buttons.append(select_btn)

        # When ticked, Capture (and Run Everything) scans both sides of the
        # object into side1/ and side2/ subfolders, and the modeling / rendering
        # steps produce their own outputs for each side. Left unticked, the
        # pipeline behaves exactly as before (single flat scan set).
        two_sides_chk = tk.Checkbutton(
            active_frame,
            text="Scan both sides of the object (stores them in side1/ and side2/)",
            variable=self.two_sides_var, anchor="w")
        two_sides_chk.pack(fill=tk.X, padx=6, pady=(0, 8))
        self.buttons.append(two_sides_chk)

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10)

        def add_button(text, command, **kwargs):
            b = tk.Button(button_frame, text=text, command=command, anchor="w", **kwargs)
            b.pack(fill=tk.X, pady=2)
            self.buttons.append(b)
            return b

        add_button("Check / Install Dependencies", self.on_install_deps)
        add_button("Open Focus Viewer (set camera focus)", self.on_open_focus_viewer)
        add_button("Step 0a — Capture Calibration (flat copy paper)", self.on_capture_calibration)
        add_button("Step 0b — Capture Colour Chart (24-patch ColorChecker)",
                   self.on_capture_chart)
        add_button("Step 0c — Fit Colour Matrix (re-run without re-shooting)",
                   self.on_fit_color)
        tk.Frame(button_frame, height=1, bg="#cccccc").pack(fill=tk.X, pady=6)
        add_button("Step 1 — Capture Scroll (needs camera + Arduino)", self.on_run_capture)
        add_button("Step 2 — Run Modeling Pipeline", self.on_run_modeling)
        add_button("Step 3 — Build 3D Model (.glb)", self.on_build_model)
        add_button("Step 4 — Generate Artifact Description (.txt)", self.on_generate_desc)
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
        start = self.active_dir if self.active_dir.exists() else ROOT
        folder = filedialog.askdirectory(
            initialdir=str(start), title="Select the image set to work with")
        if not folder:
            self.log("Folder selection cancelled.")
            return
        self.active_dir = Path(folder)
        self.active_dir_var.set(str(self.active_dir))
        self.log(f"Active working folder set to: {self.active_dir}")
        for side in self._side_dirs():
            label = self._side_label(side)
            prefix = f"  {label}: " if label else "  Folder status: "
            have = [
                "scans" if self._has_scans(side) else "no scans",
                "maps" if self._has_maps(side) else "no maps",
            ]
            have.append(".glb" if self._has_glb(side) else "no .glb")
            self.log(prefix + ", ".join(have) + ".")
        self.log("  Description: " + (TXT_NAME if self._has_txt() else "no " + TXT_NAME))
        self.log("Click a step or 'Run Everything' to process it.")

    def on_open_focus_viewer(self):
        self.run_in_background(self.step_open_focus_viewer, "focus viewer")

    def on_run_capture(self):
        self.run_in_background(self.step_run_capture, "capture")

    def on_capture_calibration(self):
        self.run_in_background(self.step_capture_calibration, "capture calibration")

    def on_capture_chart(self):
        self.run_in_background(self.step_capture_chart, "capture colour chart")

    def on_fit_color(self):
        self.run_in_background(self.step_fit_color, "fit colour matrix")

    def on_run_modeling(self):
        self.run_in_background(self.step_run_modeling, "modeling pipeline")

    def on_build_model(self):
        self.run_in_background(self.step_build_model, "build 3D model")

    def on_generate_desc(self):
        self.run_in_background(self.step_generate_desc, "generate description")

    def on_run_all(self):
        self.run_in_background(self.step_run_all, "full pipeline")

    # ── Steps ───────────────────────────────────────────────────────────────────
    def wrap_description(self, text, width=70):
        """Wrap long description text to a fixed width, like the example file."""
        paragraphs = text.splitlines() or [""]
        wrapped_lines = []
        for para in paragraphs:
            if para.strip() == "":
                wrapped_lines.append("")
            else:
                wrapped_lines.extend(textwrap.wrap(para, width=width))
        return "\n".join(wrapped_lines)

    def submit(self, root):
        name = self.name_entry.get().strip()
        type_ = self.type_var.get().strip()
        description = self.description_text.get("1.0", tk.END).strip()

        if not name or not type_ or not description:
            messagebox.showwarning(
                "Missing information",
                "Please fill in Name, Type, and Description before submitting."
            )
            return

        save_dir = self.active_dir
        if not save_dir:
            messagebox.showerror(
                "No working folder set",
                "No active working folder is set.\n"
                "Select or capture an image set before generating a description."
            )
            return

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, TXT_NAME)

        content = (
            f"Name: {name}\n"
            f"Type: {type_}\n"
            f"Description: {self.wrap_description(description)}\n"
        )

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            messagebox.showerror("Error saving file", str(e))
            return

        messagebox.showinfo("Saved", f"Saved to:\n{save_path}")
        root.destroy()

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
        elif (ver.major, ver.minor) > (3, 12):
            self.log(f"  WARNING: Python {ver.major}.{ver.minor} is newer than this "
                     "pipeline supports. The pinned numpy<2.0 (and scipy/opencv/"
                     "rembg) have no prebuilt wheels past 3.12, so pip will try to "
                     "compile numpy from source and fail. Install Python 3.9-3.12 "
                     "and launch with it (e.g. python3.12 run_linux.py).")
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

        # 3. Colour-fit virtualenv (Step 0c) ---------------------------------
        self.log("\nColour-fit environment (needed for 'Step 0c — Fit Colour Matrix'):")
        self._ensure_colorfit_env(ok, no)

        # 4. Node.js + renderer packages (Build 3D Model step) ----------------
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

        # 5. Capture-rig tools (only needed for capture / focus viewer) -------
        self.log("\nCapture-rig tools (only needed to run Capture / Focus Viewer):")
        gphoto2 = shutil.which("gphoto2")
        if gphoto2:
            self.log(f"  {ok} gphoto2 ({gphoto2})")
        else:
            self.log(f"  {no} gphoto2 not found on PATH — install it "
                      "(e.g. sudo apt install gphoto2)")
        dcraw = shutil.which("dcraw")
        if dcraw:
            self.log(f"  {ok} dcraw ({dcraw})")
        else:
            self.log(f"  {no} dcraw not found on PATH — install it "
                      "(e.g. sudo apt install dcraw; converts .tmp -> .tiff)")
        self.log("       The Arduino must be flashed with backend/capture/IrisArduinoCode "
                  "and connected. The Linux capture scripts default to serial port "
                  "/dev/ttyACM0 (override with the PAPYRUS_SERIAL_PORT env var).")
        self.log("       You may need to add your user to the 'dialout' group for "
                  "serial access: sudo usermod -a -G dialout $USER (then re-login).")

        self.log("\nDone. Re-run this any time to re-check. See SETUP.md for the full guide.")
        self.log("=" * 64)

    def _new_capture_dir(self) -> Path:
        """A fresh timestamped scan folder inside the active working folder.

        The Linux capture script writes straight into the folder it is handed
        (PAPYRUS_CAPTURE_DIR), unlike the Windows one which mints a timestamped
        subfolder of its own, so the launcher names the folder here. Capturing
        into the *active* folder (rather than always into the top-level data/)
        means selecting e.g. data/realPapyrus and hitting Capture keeps the new
        scan set with the rest of that artifact's sets."""
        return self.active_dir / datetime.now().strftime("%d-%m-%y_%H-%M-%S")

    # ── Active working folder ────────────────────────────────────────────────────
    @property
    def two_sides(self) -> bool:
        """True when the user has opted to scan both sides of the object."""
        return bool(self.two_sides_var.get())

    def _side_dirs(self) -> list:
        """The per-side working directories to process for the active folder.

        - Two-sided mode: <active>/side1 and <active>/side2.
        - Single-sided mode: just the active folder itself (historic flat
          layout, so every existing single-side folder keeps working as before).

        If the active folder already contains side1/side2 subfolders (e.g. the
        user selected an older two-sided set), those are used regardless of the
        checkbox so the modeling / rendering steps still process both sides.
        """
        side_dirs = [self.active_dir / name for name in SIDE_SUBDIRS]
        if self.two_sides:
            return side_dirs
        existing = [d for d in side_dirs if d.is_dir()]
        return existing if existing else [self.active_dir]

    def _side_label(self, side: Path) -> str:
        """Short label for a side directory, empty when it's the flat active folder."""
        return side.name if side != self.active_dir else ""

    def _maps_dir(self, side: Path = None) -> Path:
        return (side or self.active_dir) / MAPS_SUBDIR

    def _glb_path(self, side: Path = None) -> Path:
        """Where the built model for `side` is saved: <side>.glb, alongside the
        folder rather than inside it (side1/ -> side1.glb), matching the naming
        the website's artifact manifest expects."""
        side = side or self.active_dir
        return side.parent / f"{side.name}.glb"

    def _txt_path(self, side: Path = None) -> Path:
        return (side or self.active_dir) / TXT_NAME

    # ── Calibration folders ──────────────────────────────────────────────────
    # Calibration is zoom-specific: the flat-field envelope is a per-pixel
    # spatial fit and the modeling pipeline hard-errors if its shape does not
    # match the scan. So each scan set carries the calibration it was shot with,
    # and only falls back to the shared global folder when it has none.
    def _cal_capture_dir(self) -> Path:
        """Where a NEW calibration capture goes: <working>/cal/."""
        return self.active_dir / CAL_SUBDIR

    def _cal_dir(self, side: Path = None) -> Path:
        """Which calibration set applies to `side`. Mirrors the resolution order
        in backend/modeling/color_correction.py: the scan's own cal/, else the
        shared parent cal/ (so side1/ and side2/ share one set), else global."""
        d = side or self.active_dir
        for candidate in (d / CAL_SUBDIR, d.parent / CAL_SUBDIR):
            if candidate.is_dir():
                return candidate
        return CALIBRATION_IMAGES

    def _has_cal(self, d: Path = None) -> bool:
        """True when a calibration set has the full flat-field paper series."""
        cal = self._cal_dir(d)
        return all((cal / name).exists() for name in CAPTURE_TIFFS)

    def _has_chart(self, d: Path = None) -> bool:
        return (self._cal_dir(d) / CHART_SUBDIR / "allLight.tiff").exists()

    def _has_ccm(self, d: Path = None) -> bool:
        return (self._cal_dir(d) / CCM_NAME).exists()

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

    def _has_glb(self, side: Path = None) -> bool:
        return self._glb_path(side).exists()

    def _has_txt(self, d: Path = None) -> bool:
        d = d or self.active_dir
        return (d / TXT_NAME).exists()

    def _ask_continue(self, title: str, message: str) -> bool:
        """Pop a modal OK/Cancel dialog from a background thread; return True on OK.

        Tk dialogs must run on the main thread, so schedule it there and block
        the worker thread until the user answers."""
        result = {}
        done = threading.Event()

        def ask():
            result["ok"] = messagebox.askokcancel(title, message)
            done.set()

        self.root.after(0, ask)
        done.wait()
        return bool(result.get("ok"))

    def _run_capture_script(self, capture_dir: Path = None) -> int:
        """Run the shared capture script (Arduino + camera). Returns its exit code.

        When capture_dir is given, the script writes the scans straight into it
        (via PAPYRUS_CAPTURE_DIR) instead of minting its own timestamped folder."""
        if not CAPTURE_SCRIPT.exists():
            self.log(f"ERROR: capture script not found at {CAPTURE_SCRIPT}")
            return 1
        self.log("Running the capture rig (Arduino + camera)...")
        self.log("This requires the camera connected over USB and the Arduino "
                  "on a serial port, plus gphoto2/dcraw installed.")
        env = None
        if capture_dir is not None:
            capture_dir.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env["PAPYRUS_CAPTURE_DIR"] = str(capture_dir)
        rc = self.run_command([sys.executable, "-u", str(CAPTURE_SCRIPT)],
                              cwd=CAPTURE_DIR, env=env)
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

        # The sidecar records which transfer curve the TIFFs carry (linear vs
        # BT.709). It has to travel with them: this is the exact moment a
        # capture becomes a calibration set, and if the encoding metadata is
        # dropped here the modeling pipeline falls back to assuming BT.709 and
        # will mis-linearise a linear set without ever saying so.
        info_src = latest / CAPTURE_INFO_NAME
        if info_src.exists():
            shutil.copy2(info_src, dest / CAPTURE_INFO_NAME)
            self.log(f"  {CAPTURE_INFO_NAME} -> {label}")
        else:
            self.log(f"  NOTE: no {CAPTURE_INFO_NAME} in {latest} — the pipeline "
                     "will assume these images are BT.709-encoded.")

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
        if self.two_sides:
            self._run_capture_two_sided()
            return

        self.log("SCROLL CAPTURE — place the scroll on the stage before "
                  "continuing.")
        latest = self._new_capture_dir()
        rc = self._run_capture_script(capture_dir=latest)
        if rc != 0:
            return
        if not latest.is_dir():
            self.log("Capture reported success but no output folder was found in "
                      f"{self.active_dir}.")
            return
        # The capture script wrote the scans into <active>/<timestamp>/, which
        # is a self-contained working folder — make it the active one instead
        # of copying the images somewhere else.
        self._set_active_dir(latest)
        self.log(f"Capture finished. Active working folder set to: {latest}")
        if self._has_scans(latest):
            self.log("Scroll scans are in place. Make sure calibration images "
                      "exist too (Step 0a — Capture Calibration), then run the "
                      "modeling pipeline.")
        else:
            missing = [n for n in CAPTURE_TIFFS if not (latest / n).exists()]
            self.log("WARNING: capture folder is missing: " + ", ".join(missing))

    def _run_capture_two_sided(self):
        """Capture both sides of the object into side1/ and side2/ subfolders of
        one fresh working folder, pausing between them so the user can flip the
        object over."""
        working = self._new_capture_dir()
        self.log("TWO-SIDED SCROLL CAPTURE")
        self.log(f"Working folder: {working}")

        for idx, sub in enumerate(SIDE_SUBDIRS, start=1):
            if idx == 1:
                self.log(f"SIDE {idx} — place the scroll on the stage before "
                          "continuing.")
            else:
                if not self._ask_continue(
                        f"Capture side {idx}",
                        f"Side {idx - 1} captured.\n\nFlip the scroll over to its "
                        f"other side, then click OK to capture side {idx} "
                        "(Cancel to stop)."):
                    self.log("Two-sided capture cancelled after side "
                              f"{idx - 1}.")
                    # Side 1 is still on disk; make its set usable on its own.
                    self._set_active_dir(working)
                    return
                self.log(f"SIDE {idx} — capturing the flipped scroll...")

            dest = working / sub
            rc = self._run_capture_script(capture_dir=dest)
            if rc != 0:
                self.log(f"Capture of {sub} failed (exit code {rc}); stopping.")
                self._set_active_dir(working)
                return
            if not self._has_scans(dest):
                missing = [n for n in CAPTURE_TIFFS if not (dest / n).exists()]
                self.log(f"WARNING: {sub} is missing: " + ", ".join(missing))

        self._set_active_dir(working)
        self.log(f"Two-sided capture finished. Active working folder set to: {working}")
        self.log("Both sides captured into side1/ and side2/. Make sure "
                  "calibration images exist (Step 0a — Capture Calibration), then "
                  "run the modeling pipeline.")

    def _check_exposure(self, folder: Path, label: str) -> None:
        """Report per-channel saturation in a freshly captured set.

        Worth doing at the rig rather than hours later in the pipeline: a
        saturated channel cannot be recovered by any amount of colour
        correction, and red saturates long before an image looks bright.
        """
        try:
            import numpy as np
            import tifffile
        except ImportError:
            return
        self.log(f"Checking {label} exposure...")
        worst = 0.0
        for name in CAPTURE_TIFFS:
            p = folder / name
            if not p.exists():
                continue
            try:
                a = tifffile.imread(str(p))
            except Exception:
                continue
            if a.ndim == 3:
                fr = {c: float((a[:, :, i] >= 65000).mean())
                      for i, c in enumerate("RGB")}     # tifffile -> RGB order
            else:
                fr = {"grey": float((a >= 65000).mean())}
            hot = {c: v for c, v in fr.items() if v > 0.001}
            worst = max(worst, max(fr.values()))
            if hot:
                self.log(f"  {name}: " + ", ".join(
                    f"{c} {v * 100:.1f}% saturated" for c, v in hot.items()))
        if worst > 0.001:
            self.log(f"  WARNING: up to {worst * 100:.1f}% of pixels are pinned at "
                     "full scale. Reduce exposure by roughly "
                     f"{max(1, round(np.log2(1 / max(1 - worst, 0.25)))):.0f}-2 stops "
                     "and re-shoot — clipped channels cannot be colour-corrected.")
        else:
            self.log("  Exposure looks good — nothing significant is clipping.")

    def step_capture_calibration(self):
        self.log("CALIBRATION CAPTURE — place a sheet of flat copy paper (no "
                  "scroll) on the stage before continuing. The same lighting "
                  "sequence is used as for the scroll.")
        self.log("Set the exposure so the paper does NOT clip (brightest pixel "
                  "below ~92% of full scale), and do not change it again until "
                  "the chart and the artifact are both shot.")
        dest = self._cal_capture_dir()
        rc = self._run_capture_script(capture_dir=dest)
        if rc != 0:
            return
        if not self._has_scans(dest):
            missing = [n for n in CAPTURE_TIFFS if not (dest / n).exists()]
            self.log("WARNING: calibration capture is missing: " + ", ".join(missing))
            return
        self.log(f"Calibration captured into {dest}")
        self._check_exposure(dest, "calibration")
        # Also refresh the global set so a scan folder without its own cal/ still
        # has something to fall back on.
        self._import_capture_into(dest, CALIBRATION_IMAGES, "backend/calibration/")
        self.log("Next: Step 0b — Capture Colour Chart, at this same exposure.")

    def step_capture_chart(self):
        """Shoot the 24-patch ColorChecker for the active working folder.

        Reuses the normal capture sequence unchanged and simply points it at
        cal/chart/. That gives allLight.tiff shot CROSS-POLARISED (the lighting
        sequence shoots allLight and the four cross frames before rotating the
        polariser), which is exactly what colorimetry wants: even illumination
        from all four lights with the chart's surface glare suppressed. It also
        matches how the diffuse map is built, and cross-polarised measurement is
        what the chart's published reference values represent.
        """
        cal = self._cal_capture_dir()
        if not self._has_cal(self.active_dir):
            self.log("WARNING: no flat-field copy-paper set found for this "
                     "working folder. Shoot Step 0a FIRST — the colour fit "
                     "divides the chart by the paper envelope, and both must "
                     "share one exposure setting.")
        if not self._ask_continue(
                "Capture colour chart",
                "Place the 24-patch ColorChecker flat on the stage, centred, "
                "filling roughly a third of the frame, at the same height as "
                "the artifact.\n\n"
                "Do NOT change exposure, aperture, ISO or focus from the "
                "copy-paper shot — the fit assumes they match.\n\n"
                "OK to capture, Cancel to stop."):
            self.log("Colour-chart capture cancelled.")
            return

        dest = cal / CHART_SUBDIR
        self.log(f"COLOUR CHART CAPTURE -> {dest}")
        rc = self._run_capture_script(capture_dir=dest)
        if rc != 0:
            return
        if not (dest / "allLight.tiff").exists():
            self.log("WARNING: chart capture produced no allLight.tiff — that is "
                     "the frame the fit uses. Check the camera and re-run.")
            return
        self.log(f"Colour chart captured into {dest}")
        self._check_exposure(dest, "colour chart")
        self.log("A clipped patch biases the whole matrix, so the white patch in "
                 "particular must not be at full scale.")
        self.log("Fitting the colour matrix...")
        self.step_fit_color()

    # ── Colour matrix fit ────────────────────────────────────────────────────
    def _venv_python(self, venv: Path) -> Path:
        return venv / "bin" / "python"

    def _colorfit_missing(self, python: Path):
        """What the colour-fit venv at `python` lacks, as a list (empty when it
        is complete), or None when that interpreter cannot run at all."""
        try:
            res = subprocess.run([str(python), "-c", COLORFIT_CHECK],
                                 capture_output=True, text=True, timeout=300)
        except Exception:
            return None
        if res.returncode != 0:
            return None
        lines = res.stdout.strip().splitlines()
        return lines[-1].split() if lines else []

    def _ensure_colorfit_env(self, ok: str, no: str):
        """Create or repair the colour-fit venv so Step 0c works without manual
        setup. Kept apart from the pipeline's packages for the reason given at
        COLORFIT_VENV; this is the only place that installs into it."""
        venv = next((b / COLORFIT_VENV for b in (ROOTER, ROOT)
                     if self._venv_python(b / COLORFIT_VENV).is_file()),
                    ROOTER / COLORFIT_VENV)
        python = self._venv_python(venv)
        missing = self._colorfit_missing(python) if python.is_file() else None
        if missing == []:
            self.log(f"  {ok} {venv}")
            return

        if missing is None or "pip" in missing:
            ver = sys.version_info
            if not (3, 9) <= (ver.major, ver.minor) <= (3, 12):
                self.log(f"  {no} {venv} — cannot create it with Python "
                         f"{ver.major}.{ver.minor}: colour-science 0.4.x and numpy<2 "
                         "need Python 3.9-3.12. Relaunch with one of those and "
                         "click this button again.")
                return
            self.log(f"  {no} {venv} "
                     + ("is broken" if venv.exists() else "not found")
                     + " -> creating it")
            rc = self.run_command([sys.executable, "-m", "venv", "--clear", str(venv)])
            if rc != 0 or not python.is_file():
                self.log(f"  Could not create {venv} (exit code {rc}).")
                self.log("  On Debian/Ubuntu this usually means the venv module is "
                         "not installed: sudo apt install "
                         f"python{sys.version_info.major}.{sys.version_info.minor}-venv")
                return
        else:
            self.log(f"  {no} {venv} is missing {', '.join(missing)} -> will install")
            if "cv2" in missing or "cv2.mcc" in missing:
                # A plain OpenCV shares cv2/ with the contrib build, so installing
                # contrib over it leaves whichever wrote last. Clear every
                # variant first so the reinstall below lands on a clean cv2/.
                self.run_command([str(python), "-m", "pip", "uninstall", "-y",
                                  "opencv-python", "opencv-python-headless",
                                  "opencv-contrib-python",
                                  "opencv-contrib-python-headless"])

        self.log("  Installing the colour-fit packages (contrib OpenCV + "
                 "colour-science; may take a few minutes)...")
        rc = self.run_command([str(python), "-m", "pip", "install",
                               "-r", str(COLORFIT_REQS)])
        missing = self._colorfit_missing(python)
        if rc == 0 and missing == []:
            self.log(f"  {ok} colour-fit environment ready ({venv})")
        else:
            self.log(f"  {no} colour-fit setup incomplete (pip exit code {rc}"
                     + (f", still missing: {', '.join(missing)}" if missing else "")
                     + "). Step 0c will not run until this succeeds.")

    def _colorfit_python(self) -> Path | None:
        """Interpreter for color_fit.py, or None if its environment is missing.

        The fit needs opencv-contrib (for cv2.mcc) and colour-science, which the
        pipeline does not — and rembg depends on plain opencv-python-headless,
        so installing contrib beside it is a silent last-writer-wins race over
        the same cv2/ directory. It therefore lives in its own virtualenv.
        Falls back to this interpreter if it happens to have cv2.mcc already.
        """
        for base in (ROOTER, ROOT):
            candidate = self._venv_python(base / COLORFIT_VENV)
            if candidate.is_file():
                return candidate
        if self._cv2_mcc_present(sys.executable):
            return Path(sys.executable)
        return None

    def _cv2_mcc_present(self, python: str) -> bool:
        """Whether `python` can do chart detection.

        Deliberately not _module_present: that uses importlib.find_spec, which
        cannot see cv2.mcc because it is a C-extension attribute of cv2 rather
        than an importable submodule.
        """
        try:
            return subprocess.run(
                [str(python), "-c", "import cv2, sys; sys.exit(0 if hasattr(cv2,'mcc') else 1)"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode == 0
        except Exception:
            return False

    def step_fit_color(self):
        """Fit ccm.json from the captured chart. Safe to re-run."""
        cal = self._cal_dir()
        chart = cal / CHART_SUBDIR / "allLight.tiff"
        if not chart.exists():
            self.log(f"No colour chart found at {chart}.")
            self.log("Capture one with 'Step 0b — Capture Colour Chart' first.")
            return
        if not self._has_cal(self.active_dir):
            self.log(f"WARNING: {cal} has no copy-paper set, so the chart cannot "
                     "be flat-fielded the way the pipeline flat-fields the "
                     "artifact. The fit will still run but will be less accurate.")

        python = self._colorfit_python()
        if python is None:
            self.log("Cannot fit: the colour-fit environment is missing.")
            self.log("Click 'Check / Install Dependencies' to create it "
                     "automatically, then run this step again.")
            return

        self.log(f"Using {python}")
        rc = self.run_command([str(python), "-u", str(MODELING_DIR / "color_fit.py"),
                               "fit", "--cal-dir", str(cal)], cwd=MODELING_DIR)
        overlay = cal / "chart_detect_overlay.png"
        if rc == 0:
            self.log(f"Wrote {cal / CCM_NAME}. The modeling pipeline will apply it "
                     "automatically to every scan using this calibration set.")
            if overlay.exists():
                self.log(f"Open {overlay} and confirm the red squares sit inside "
                         "the patches before trusting the numbers.")
        else:
            self.log("Colour fit FAILED — see the log above.")
            self.log("If the chart could not be detected, re-run the command "
                     "shown above by hand adding --box with the chart's four "
                     "corners, which you can read off the overlay image.")
            self.log("Until this succeeds the pipeline runs UNCORRECTED.")

    def step_run_modeling(self):
        sides = self._side_dirs()
        for side in sides:
            label = self._side_label(side)
            if label:
                self.log("=" * 64)
                self.log(f"MODELING — {label}")
                self.log("=" * 64)
            self._run_modeling_for(side)

    def _run_modeling_for(self, side: Path) -> int:
        """Run the modeling pipeline on one side directory, writing its
        render-ready maps into <side>/maps/. Returns the exit code."""
        script = MODELING_DIR / "modeling_pipeline.py"
        if not self._has_scans(side):
            self.log(f"WARNING: {side} is missing some scroll scans.")
            self.log("Capture or select a folder with the full scan set "
                      f"({', '.join(CAPTURE_TIFFS)}) before running the pipeline.")

        maps_dir = self._maps_dir(side)
        maps_dir.mkdir(parents=True, exist_ok=True)

        # Point the modeling pipeline at this side's scans and have it write its
        # render-ready maps straight into <side>/maps/, using the calibration
        # set that belongs to this scan rather than whatever was shot last.
        cal_dir = self._cal_dir(side)
        env = dict(os.environ)
        env["PAPYRUS_SCROLL_DIR"] = str(side)
        env["PAPYRUS_RENDER_OUT"] = str(maps_dir)
        env["PAPYRUS_CAL_DIR"]    = str(cal_dir)

        self.log(f"Running modeling pipeline on {side} "
                  "(this can take several minutes)...")
        self.log(f"Render-ready maps -> {maps_dir}")
        if cal_dir == CALIBRATION_IMAGES:
            self.log(f"Calibration      -> {cal_dir} (shared global set — this scan "
                     "has no cal/ of its own, so the flat-field may not match "
                     "its zoom)")
        else:
            self.log(f"Calibration      -> {cal_dir}")
        if not self._has_ccm(side):
            self.log("No ccm.json for this calibration set — running without "
                     "colour correction. Shoot Step 0b and fit to enable it.")
        rc = self.run_command([sys.executable, "-u", script], cwd=MODELING_DIR, env=env)
        if rc == 0:
            self.log("Modeling pipeline finished successfully.")
        else:
            self.log(f"Modeling pipeline exited with code {rc}.")
            self.log("If you saw 'ModuleNotFoundError', click "
                      "'Install Python dependencies' and try again.")
        return rc

    def step_build_model(self):
        sides = self._side_dirs()
        for side in sides:
            label = self._side_label(side)
            if label:
                self.log("=" * 64)
                self.log(f"BUILD 3D MODEL — {label}")
                self.log("=" * 64)
            self._build_model_for(side)
        self._cleanup_intermediates()

    def _cleanup_intermediates(self):
        """Remove intermediate .tmp/.tiff/.glb files left under backend/ after a
        build. Mirrors backend/delete-tmp-tiff.ps1 (the Windows path used by
        run.py) but done natively in Python so it works on Linux. The 9 capture
        TIFFs are kept, and the working data/ folders (outside backend/) — where
        the copied render.glb, maps/ and info.txt live — are never touched.

        backend/calibration/ is skipped wholesale. Its flat-field TIFFs happen
        to share the 9 capture filenames and so used to survive by coincidence,
        but the colour-chart shot in calibration/chart/ does not — deleting the
        chart would silently destroy the reference the colour matrix is fitted
        from. Exclude the whole tree rather than rely on filenames."""
        keep = set(CAPTURE_TIFFS)
        protected_dir = CALIBRATION_IMAGES.resolve()
        removed = 0
        for pattern in ("*.tmp", "*.tiff", "*.glb"):
            for f in BACKEND.rglob(pattern):
                if protected_dir in f.resolve().parents:
                    continue
                if f.name in keep or not f.is_file():
                    continue
                try:
                    f.unlink()
                    removed += 1
                except OSError as e:
                    self.log(f"  could not delete {f}: {e}")
        if removed:
            self.log(f"Cleanup: removed {removed} intermediate file(s) under backend/.")

    def _build_model_for(self, side: Path):
        maps_dir = self._maps_dir(side)
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

        glb = RENDERING_DIR / GLB_NAME
        if glb.exists():
            dest = self._glb_path(side)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(glb, dest)
            self.log(f"Saved model -> {dest}.")
        else:
            self.log(f"{GLB_NAME} was not created — see log above for details.")

        # Drop the intermediates this build left under backend/ (the copy above
        # is already safe in the working folder). Done here as well as in
        # step_build_model so "Run Everything", which calls this directly, also
        # cleans up.
        self._cleanup_intermediates()

    def step_generate_desc(self):
        done = threading.Event()

        def build_popup():
            popup = tk.Toplevel(self.root)
            popup.title("New Artifact Info")
            popup.geometry("420x440")
            popup.resizable(False, False)

            padding = {"padx": 12, "pady": 6}
            tk.Label(popup, text="Name:").pack(anchor="w", **padding)
            self.name_entry = tk.Entry(popup, width=45)
            self.name_entry.pack(**padding)

            tk.Label(popup, text="Type:").pack(anchor="w", **padding)
            self.type_var = tk.StringVar(popup)
            self.type_var.set("papyrus")
            tk.OptionMenu(popup, self.type_var, "papyrus", "tablet").pack(**padding)

            tk.Label(popup, text="Description:").pack(anchor="w", **padding)
            self.description_text = tk.Text(popup, width=45, height=8, wrap="word")
            self.description_text.pack(**padding)

            def do_submit():
                self.submit(popup)
                done.set()

            tk.Button(popup, text="Generate .txt", command=do_submit).pack(pady=12)
            popup.protocol("WM_DELETE_WINDOW", lambda: (popup.destroy(), done.set()))
            self.name_entry.focus_set()

        self.root.after(0, build_popup)
        done.wait()  # blocks the worker thread until the popup is closed

    def _smart_run(self):
        """Run the pipeline on the active folder, skipping stages already done.

        - scroll scans present  -> skip capture
        - render-ready maps present -> skip modeling
        - render.glb present     -> skip render/build

        In two-sided mode this runs the modeling and build stages once per side
        (side1, side2), skipping whichever stages that side has already done.
        """
        self.log(f"Processing working folder: {self.active_dir}")

        # ── Capture ───────────────────────────────────────────────────────────
        # Capture produces the side folders in two-sided mode, so run it whenever
        # any side is missing its scan set, then re-resolve the side list.
        sides = self._side_dirs()
        if all(self._has_scans(s) for s in sides):
            self.log("Found a full set of scroll scans — skipping capture.")
        else:
            self.log("No complete scroll-scan set found — running capture.")
            self.step_run_capture()
            sides = self._side_dirs()
            if not sides or not all(self._has_scans(s) for s in sides):
                self.log("Capture did not produce a full scan set; stopping.")
                return

        # ── Modeling + build, per side ────────────────────────────────────────
        tag = ""
        for side in sides:
            label = self._side_label(side)
            tag = f" [{label}]" if label else ""

            if self._has_maps(side):
                self.log(f"Render-ready maps already in {self._maps_dir(side)} — "
                          f"skipping modeling{tag}.")
            else:
                self._run_modeling_for(side)
                if not self._has_maps(side):
                    self.log(f"Modeling did not produce the expected maps{tag}; "
                              "stopping.")
                    return

            if self._has_glb(side):
                self.log(f"Model already built ({self._glb_path(side)}) — "
                          f"skipping render{tag}.")
            else:
                self._build_model_for(side)
                if not self._has_glb(side):
                    self.log(f"Model build did not produce render.glb{tag}; "
                              "stopping.")
                    return

        # ── Artifact description (.txt) ───────────────────────────────────────
        if self._has_txt(self.active_dir):
            self.log(f"Description already written ({self._txt_path()}) — "
                     f"skipping description generation.")
        else:
            self.step_generate_desc()
            if not self._has_txt():
                self.log(f"Description generation failed{tag}; stopping.")
                return

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
