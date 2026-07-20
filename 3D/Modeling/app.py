#!/usr/bin/env python3
"""
Automatic Tablet Reconstruction Pipeline GUI

Stages:
  1. Background removal     (process_photos.py)
  2. COLMAP MVS             (run.sh, once per side with FIPMESH_SKIP_RECON=1)
  3. FPFH alignment         (alignment/run.py → output/aligned_cloud/merged_fpfh.ply)
  4. Mesh reconstruction    (src/reconstruct_mesh.py → output/recon/ + output/model.gltf)

Run:  python app.py   (in WSL with WSLg)
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import threading
import subprocess
import queue
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# ── Path constants ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ALIGN_DIR  = SCRIPT_DIR / "alignment"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

STAGE_NAMES = [
    "1. Background Removal",
    "2. COLMAP MVS",
    "3. FPFH Alignment",
    "4. Mesh Reconstruction",
]

# ── Color palette ──────────────────────────────────────────────────────────────
PAL = {
    "bg":      "#f1f3f7",
    "card":    "#ffffff",
    "accent":  "#2563eb",
    "success": "#16a34a",
    "error":   "#dc2626",
    "subtext": "#64748b",
    "border":  "#e2e8f0",
    "text":    "#1e293b",
    "log_bg":  "#0f172a",
    "log_fg":  "#cbd5e1",
    "log_hdr": "#60a5fa",
}


def apply_theme(root: tk.Tk):
    s = ttk.Style(root)
    s.theme_use("clam")

    s.configure(".", background=PAL["bg"], foreground=PAL["text"],
                font=("Segoe UI", 9), borderwidth=0)
    s.configure("TFrame",       background=PAL["bg"])
    s.configure("TLabel",       background=PAL["bg"], foreground=PAL["text"])
    s.configure("TLabelframe",  background=PAL["bg"], bordercolor=PAL["border"],
                relief="flat")
    s.configure("TLabelframe.Label", background=PAL["bg"],
                foreground=PAL["subtext"], font=("Segoe UI", 9, "bold"))

    s.configure("TNotebook",     background=PAL["bg"], borderwidth=0)
    s.configure("TNotebook.Tab", background=PAL["border"],
                foreground=PAL["subtext"], padding=(10, 4),
                font=("Segoe UI", 9))
    s.map("TNotebook.Tab",
          background=[("selected", PAL["card"])],
          foreground=[("selected", PAL["accent"])])

    s.configure("TButton", background=PAL["accent"], foreground="white",
                padding=(10, 5), relief="flat", font=("Segoe UI", 9))
    s.map("TButton",
          background=[("active", "#1d4ed8"), ("disabled", PAL["border"])],
          foreground=[("disabled", PAL["subtext"])])

    s.configure("TEntry",    fieldbackground=PAL["card"],
                bordercolor=PAL["border"], relief="flat", padding=(4, 3))
    s.configure("TCombobox", fieldbackground=PAL["card"],
                bordercolor=PAL["border"])
    s.configure("TSpinbox",  fieldbackground=PAL["card"],
                bordercolor=PAL["border"])
    s.configure("TCheckbutton", background=PAL["bg"])
    s.configure("TSeparator",   background=PAL["border"])

    s.configure("Horizontal.TProgressbar",
                troughcolor=PAL["border"], background=PAL["accent"],
                borderwidth=0, thickness=6)
    s.configure("Success.Horizontal.TProgressbar",
                troughcolor=PAL["border"], background=PAL["success"],
                borderwidth=0, thickness=6)
    s.configure("Error.Horizontal.TProgressbar",
                troughcolor=PAL["border"], background=PAL["error"],
                borderwidth=0, thickness=6)

    root.configure(background=PAL["bg"])


# ══════════════════════════════════════════════════════════════════════════════
#  Progress parsers (stateful; call feed(line) → (pct, text) | None)
# ══════════════════════════════════════════════════════════════════════════════

class PhotosParser:
    def __init__(self):
        self._total = 0

    def feed(self, line: str):
        m = re.search(r'Processing (\d+) image', line)
        if m:
            self._total = int(m.group(1))
            return (2, f"Processing {self._total} images…")
        m = re.search(r'\[(\d+)/(\d+)\]', line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            if n > 0:
                self._total = n
                return (int(5 + 90 * i / n), f"Image {i}/{n}")
        if re.search(r'\[done\]', line, re.IGNORECASE):
            return (100, "Complete")
        return None


class ColmapParser:
    # (pattern, frac_at_start, step_id, label)
    _STEP_DEFS = [
        (r'\[step\] colmap feature_extractor',
         0.05, 'extract', "Feature extraction"),
        (r'\[step\] colmap (?:exhaustive_matcher|vocab_tree_matcher|sequential_matcher)',
         0.20, 'match', "Feature matching"),
        (r'\[step\] colmap mapper',
         0.40, 'map', "Sparse mapping"),
        (r'\[.*?\] colmap image_undistorter',
         0.55, 'undistort', "Image undistortion"),
        (r'\[.*?\] colmap patch_match_stereo',
         0.62, 'pms', "Patch match stereo"),
        (r'\[.*?\] colmap stereo_fusion',
         0.82, 'fusion', "Stereo fusion"),
        (r'dense cloud output:',
         1.00, 'done', "Dense cloud complete"),
    ]

    def __init__(self, lo: int = 0, hi: int = 100):
        self._lo   = lo
        self._hi   = hi
        self._pct  = lo
        self._step = None
        self._frac_start = 0.0
        self._frac_end   = 1.0
        # precompute per-step end fracs (= next step's start)
        defs = self._STEP_DEFS
        self._steps = [
            (pat, frac_s, defs[i + 1][1] if i + 1 < len(defs) else 1.0, sid, lbl)
            for i, (pat, frac_s, sid, lbl) in enumerate(defs)
        ]

    def _scale(self, frac: float) -> int:
        return int(self._lo + (self._hi - self._lo) * frac)

    def feed(self, line: str):
        # Step keyword transitions
        for pat, frac_s, frac_e, step_id, label in self._steps:
            if re.search(pat, line):
                self._step       = step_id
                self._frac_start = frac_s
                self._frac_end   = frac_e
                pct = self._scale(frac_s)
                if pct > self._pct:
                    self._pct = pct
                    return (self._pct, label)
                return None

        # "Processing view X/N" — photometric (pms) and geometric (fusion)
        if self._step in ('pms', 'fusion'):
            m = re.search(r'Processing view\s+(\d+)\s*/\s*(\d+)', line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n > 0:
                    frac = self._frac_start + (self._frac_end - self._frac_start) * i / n
                    pct  = self._scale(frac)
                    if pct > self._pct:
                        self._pct = pct
                        kind = "Photometric" if self._step == 'pms' else "Geometric"
                        return (self._pct, f"{kind}: view {i}/{n}")

        # "Processing file [N/M]" — feature extraction per-image
        if self._step == 'extract':
            m = re.search(r'Processing file \[(\d+)/(\d+)\]', line)
            if m:
                i, n = int(m.group(1)), int(m.group(2))
                if n > 0:
                    frac = self._frac_start + (self._frac_end - self._frac_start) * i / n
                    pct  = self._scale(frac)
                    if pct > self._pct:
                        self._pct = pct
                        return (self._pct, f"Feature extraction: {i}/{n}")

        return None


class AlignParser:
    _STEPS = [
        (r'=== (?:fpfh|opening|collapse) ===', 10, "Initialising alignment…"),
        (r'FPFH on',                            20, "Computing FPFH features…"),
        (r'RANSAC fitness=',                    40, "RANSAC global registration…"),
        (r'refine ',                            60, "Refining alignment (ICP)…"),
        (r'chosen ',                            85, "Selecting best candidate…"),
        (r'Best method:',                       90, "Evaluating methods…"),
        (r'Saved merged',                      100, "Merged cloud saved"),
    ]

    def __init__(self):
        self._pct = 0

    def feed(self, line: str):
        for pattern, pct, text in self._STEPS:
            if re.search(pattern, line):
                if pct > self._pct:
                    self._pct = pct
                    return (pct, text)
        return None


class ReconParser:
    _STEPS = [
        (r'\[step\] random downsample',           4,  "Random downsample…"),
        (r'\[step\] input',                        7,  "Loading input…"),
        (r'\[step\] voxel downsample',            11,  "Voxel downsample…"),
        (r'\[step\] statistical outlier removal',  17,  "Statistical outlier removal…"),
        (r'\[step\] radius outlier removal',       22,  "Radius outlier removal…"),
        (r'\[step\] bbox quantile crop',           26,  "Bounding box crop…"),
        (r'\[step\] cluster cleanup',              32,  "DBSCAN cluster cleanup…"),
        (r'\[step\] optional mirror',              35,  "Mirror check…"),
        (r'\[step\] estimate \+ orient normals',   42,  "Estimating normals…"),
        (r'\[step\] write cleaned cloud',          50,  "Writing cleaned cloud…"),
        (r'\[step\] poisson reconstruction',       57,  "Poisson reconstruction…"),
        (r'\[step\] density trim',                 67,  "Density trim…"),
        (r'\[step\] mesh cleanup',                 73,  "Mesh cleanup…"),
        (r'\[step\] fill mesh holes',               76,  "Filling holes…"),
        (r'\[step\] mesh normal orientation',      79,  "Normal orientation check…"),
        (r'\[step\] optional decimation',          85,  "Decimation…"),
        (r'\[step\] textured output',              90,  "Textured output…"),
        (r'\[step\] write mesh exports',           94,  "Writing mesh exports…"),
        (r'\[step\] simplified export',            97,  "Simplified web-viewer export…"),
        (r'gltf output:',                         100,  "GLTF written"),
    ]

    def __init__(self):
        self._pct = 0

    def feed(self, line: str):
        for pattern, pct, text in self._STEPS:
            if re.search(pattern, line):
                if pct > self._pct:
                    self._pct = pct
                    return (pct, text)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_python(path: Path) -> str:
    return str(path) if path.exists() else sys.executable


def _has_images(d: Path) -> bool:
    return any(f.suffix.lower() in IMAGE_EXTS for f in d.rglob("*") if f.is_file())


def detect_sides(input_dir: Path) -> list[str]:
    if not input_dir.is_dir():
        return []
    return sorted(
        sub.name for sub in input_dir.iterdir()
        if sub.is_dir() and _has_images(sub)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Stage widget
# ══════════════════════════════════════════════════════════════════════════════

class StageWidget(ttk.LabelFrame):
    ICONS = {"waiting": "○", "running": "◉", "done": "✔", "failed": "✘"}

    def __init__(self, parent, stage_idx: int, name: str, view_label: str,
                 run_cb, view_cb):
        super().__init__(parent, text=name, padding=(8, 5))
        self.stage_idx = stage_idx
        self._run_cb   = run_cb
        self._view_cb  = view_cb
        self._state    = "waiting"

        # Status row
        top = ttk.Frame(self)
        top.pack(fill=tk.X)
        self._icon_lbl = tk.Label(top, text=self.ICONS["waiting"],
                                   fg=PAL["subtext"], bg=PAL["bg"],
                                   width=2, font=("Segoe UI", 10))
        self._icon_lbl.pack(side=tk.LEFT)
        self._status_var = tk.StringVar(value="Waiting")
        ttk.Label(top, textvariable=self._status_var, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        # Progress bar
        self._pb = ttk.Progressbar(self, mode="indeterminate",
                                    style="Horizontal.TProgressbar", length=200)
        self._pb.pack(fill=tk.X, pady=(4, 0))

        # Button row
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, pady=(5, 0))
        self._run_btn  = ttk.Button(btn_row, text="Run Stage",
                                     command=self._run_cb, width=11)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._view_btn = ttk.Button(btn_row, text=view_label,
                                     command=self._view_cb,
                                     width=13, state=tk.DISABLED)
        self._view_btn.pack(side=tk.LEFT)

    def set_state(self, state: str, status: str = ""):
        self._state = state
        colors = {
            "waiting": PAL["subtext"],
            "running": PAL["accent"],
            "done":    PAL["success"],
            "failed":  PAL["error"],
        }
        self._icon_lbl.config(text=self.ICONS[state], fg=colors[state])

        if status:
            self._status_var.set(status)
        else:
            defaults = {"waiting": "Waiting", "running": "Running…",
                        "done": "Done", "failed": "Failed"}
            self._status_var.set(defaults[state])

        if state == "running":
            self._pb.config(mode="indeterminate",
                             style="Horizontal.TProgressbar")
            self._pb.start(12)
            self._run_btn.config(state=tk.DISABLED)
            self._view_btn.config(state=tk.DISABLED)
        else:
            self._pb.stop()
            if state == "done":
                self._pb.config(mode="determinate", value=100,
                                 style="Success.Horizontal.TProgressbar")
            elif state == "failed":
                self._pb.config(mode="determinate", value=100,
                                 style="Error.Horizontal.TProgressbar")
            else:
                self._pb.config(mode="determinate", value=0,
                                 style="Horizontal.TProgressbar")
            self._run_btn.config(state=tk.NORMAL)
            if state == "done":
                self._view_btn.config(state=tk.NORMAL)

    @property
    def state(self) -> str:
        return self._state

    def set_progress(self, pct: int, text: str = ""):
        """Switch bar to determinate mode and update value + label."""
        if self._state != "running":
            return
        self._pb.stop()
        self._pb.config(mode="determinate", value=pct,
                         style="Horizontal.TProgressbar")
        if text:
            self._status_var.set(text)

    def set_status_text(self, text: str):
        self._status_var.set(text)


# ══════════════════════════════════════════════════════════════════════════════
#  Config Panel (left notebook)
# ══════════════════════════════════════════════════════════════════════════════

class ConfigPanel(ttk.Frame):
    def __init__(self, parent, app: "App"):
        super().__init__(parent)
        self.app = app
        self._build()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)

        io_tab     = ttk.Frame(nb, padding=8)
        colmap_tab = ttk.Frame(nb, padding=8)
        recon_tab  = ttk.Frame(nb, padding=8)
        align_tab  = ttk.Frame(nb, padding=8)

        nb.add(io_tab,     text="Inputs")
        nb.add(colmap_tab, text="COLMAP")
        nb.add(recon_tab,  text="Reconstruct")
        nb.add(align_tab,  text="Alignment")

        self._build_io(io_tab)
        self._build_colmap(_scroll_frame(colmap_tab))
        self._build_recon(_scroll_frame(recon_tab))
        self._build_align(align_tab)

    # ── IO tab ────────────────────────────────────────────────────────────────
    def _build_io(self, f):
        def browse_dir(var, title):
            d = filedialog.askdirectory(title=title)
            if d:
                var.set(d)

        def path_row(parent, label, var, browse_title):
            r = ttk.Frame(parent)
            r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=11, anchor=tk.W).pack(side=tk.LEFT)
            ttk.Entry(r, textvariable=var).pack(side=tk.LEFT, fill=tk.X,
                                                 expand=True, padx=(2, 2))
            ttk.Button(r, text="…", width=3,
                       command=lambda: browse_dir(var, browse_title)).pack(side=tk.LEFT)

        self.input_var  = tk.StringVar()
        self.output_var = tk.StringVar()
        path_row(f, "Input dir:", self.input_var, "Select input image directory")
        path_row(f, "Output dir:", self.output_var, "Select output directory")
        self.input_var.trace_add("write", self._on_input_changed)
        self.output_var.trace_add("write", self._on_output_changed)

        self._struct_var = tk.StringVar(value="")
        ttk.Label(f, textvariable=self._struct_var, foreground=PAL["subtext"],
                  wraplength=360).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(f, text="Side folders (blank = flat/single-side):").pack(anchor=tk.W)
        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Primary:", width=11).pack(side=tk.LEFT)
        self.side1_var = tk.StringVar(value="side1")
        ttk.Entry(r1, textvariable=self.side1_var, width=14).pack(side=tk.LEFT)

        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Secondary:", width=11).pack(side=tk.LEFT)
        self.side2_var = tk.StringVar(value="side2")
        ttk.Entry(r2, textvariable=self.side2_var, width=14).pack(side=tk.LEFT)
        ttk.Label(r2, text="(blank = skip alignment)",
                  foreground=PAL["subtext"]).pack(side=tk.LEFT, padx=4)

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(f, text="Background removal:",
                  font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)

        rr = ttk.Frame(f); rr.pack(fill=tk.X, pady=2)
        ttk.Label(rr, text="Background:", width=14).pack(side=tk.LEFT)
        self.bg_var = tk.StringVar(value="white")
        ttk.Combobox(rr, textvariable=self.bg_var,
                     values=["white", "black", "transparent"],
                     state="readonly", width=12).pack(side=tk.LEFT)

        rr2 = ttk.Frame(f); rr2.pack(fill=tk.X, pady=2)
        ttk.Label(rr2, text="Black threshold:", width=14).pack(side=tk.LEFT)
        self.black_thresh_var = tk.IntVar(value=0)
        ttk.Spinbox(rr2, from_=0, to=255, textvariable=self.black_thresh_var,
                    width=6).pack(side=tk.LEFT)
        ttk.Label(rr2, text="(0=off)", foreground=PAL["subtext"]).pack(
            side=tk.LEFT, padx=4)

        rr2b = ttk.Frame(f); rr2b.pack(fill=tk.X, pady=2)
        ttk.Label(rr2b, text="White threshold:", width=14).pack(side=tk.LEFT)
        self.white_thresh_var = tk.IntVar(value=0)
        ttk.Spinbox(rr2b, from_=0, to=255, textvariable=self.white_thresh_var,
                    width=6).pack(side=tk.LEFT)
        ttk.Label(rr2b, text="(0=off)", foreground=PAL["subtext"]).pack(
            side=tk.LEFT, padx=4)

        rr2c = ttk.Frame(f); rr2c.pack(fill=tk.X, pady=2)
        ttk.Label(rr2c, text="Value threshold:", width=14).pack(side=tk.LEFT)
        self.value_thresh_var = tk.IntVar(value=0)
        ttk.Spinbox(rr2c, from_=0, to=255, textvariable=self.value_thresh_var,
                    width=6).pack(side=tk.LEFT)
        ttk.Label(rr2c, text="(0=off)", foreground=PAL["subtext"]).pack(
            side=tk.LEFT, padx=4)

        rr2d = ttk.Frame(f); rr2d.pack(fill=tk.X, pady=2)
        ttk.Label(rr2d, text="Edge band (px):", width=14).pack(side=tk.LEFT)
        self.edge_band_var = tk.IntVar(value=0)
        ttk.Spinbox(rr2d, from_=0, to=100, textvariable=self.edge_band_var,
                    width=6).pack(side=tk.LEFT)
        ttk.Label(rr2d, text="(0=whole mask)", foreground=PAL["subtext"]).pack(
            side=tk.LEFT, padx=4)

        rr2e = ttk.Frame(f); rr2e.pack(fill=tk.X, pady=2)
        ttk.Label(rr2e, text="Passes:", width=14).pack(side=tk.LEFT)
        self.passes_var = tk.IntVar(value=1)
        ttk.Spinbox(rr2e, from_=1, to=5, textvariable=self.passes_var,
                    width=6).pack(side=tk.LEFT)
        ttk.Label(rr2e, text="(1=off, slower per extra pass)",
                  foreground=PAL["subtext"]).pack(side=tk.LEFT, padx=4)

        rr2f = ttk.Frame(f); rr2f.pack(fill=tk.X, pady=2)
        ttk.Label(rr2f, text="Seg. quality:", width=14).pack(side=tk.LEFT)
        self.seg_scale_var = tk.IntVar(value=100)
        seg_scale_lbl = ttk.Label(rr2f, text="100%", width=5)
        def _on_seg_scale(val):
            pct = int(round(float(val)))
            self.seg_scale_var.set(pct)
            seg_scale_lbl.config(text=f"{pct}%")
        ttk.Scale(rr2f, from_=10, to=100, orient=tk.HORIZONTAL, length=140,
                  command=_on_seg_scale, value=100).pack(side=tk.LEFT, padx=(0, 6))
        seg_scale_lbl.pack(side=tk.LEFT)
        ttk.Label(
            f,
            text=("Downscales the image fed to the bg-removal model to save memory/time; "
                  "saved output stays full resolution either way. Note: rembg resizes "
                  "every model's input to a fixed working size before running on the GPU, "
                  "so this mainly saves CPU/RAM time, not GPU VRAM."),
            foreground=PAL["subtext"], wraplength=360,
        ).pack(anchor=tk.W, pady=(0, 4))

        rr3 = ttk.Frame(f); rr3.pack(fill=tk.X, pady=2)
        ttk.Label(rr3, text="rembg model:", width=14).pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="birefnet-general")
        _REMBG_MODELS = [
            "birefnet-general",
            "birefnet-general-lite",
            "birefnet-portrait",
            "birefnet-dis",
            "birefnet-hrsod",
            "birefnet-cod",
            "birefnet-massive",
            "isnet-general-use",
            "isnet-anime",
            "silueta",
            "u2net",
            "u2netp",
            "u2net_human_seg",
            "u2net_cloth_seg",
            "bria-rmbg",
            "sam",
            "u2net_custom",
            "dis_custom",
            "ben_custom",
        ]
        ttk.Combobox(rr3, textvariable=self.model_var, values=_REMBG_MODELS,
                     state="readonly", width=22).pack(side=tk.LEFT)

    def _on_input_changed(self, *_):
        inp = self.input_var.get().strip()
        if not inp:
            return
        p = Path(inp)
        if not self.output_var.get():
            self.output_var.set(str(p.parent / (p.name + "_recon")))
        sides = detect_sides(p)
        if len(sides) >= 2:
            self.side1_var.set(sides[0])
            self.side2_var.set(sides[1])
            self._struct_var.set(f"Detected sides: {', '.join(sides)}")
        elif len(sides) == 1:
            self.side1_var.set(sides[0])
            self.side2_var.set("")
            self._struct_var.set(f"Detected: 1 side ({sides[0]}), no alignment stage")
        else:
            self.side1_var.set("")
            self.side2_var.set("")
            self._struct_var.set("Flat structure — images at root, no alignment stage")
        self.app._reconcile_stage_states()

    def _on_output_changed(self, *_):
        self.app._reconcile_stage_states()

    # ── COLMAP tab ────────────────────────────────────────────────────────────
    def _build_colmap(self, f):
        def row(label, widget_fn, *a, **kw):
            r = ttk.Frame(f); r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=22, anchor=tk.W).pack(side=tk.LEFT)
            w = widget_fn(r, *a, **kw)
            w.pack(side=tk.LEFT)
            return w

        _h(f, "COLMAP")
        self.quality_var = tk.StringVar(value="high")
        row("Quality", ttk.Combobox, textvariable=self.quality_var,
            values=["extreme", "high", "medium", "low"],
            state="readonly", width=10)

        self.use_gpu_var = tk.BooleanVar(value=True)
        row("Use GPU", ttk.Checkbutton, variable=self.use_gpu_var)

        self.gpu_index_var = tk.StringVar(value="-1")
        row("GPU index (−1=auto)", ttk.Entry,
            textvariable=self.gpu_index_var, width=6)

        self.img_scale_var = tk.StringVar(value="1")
        row("Image scale (0–1)", ttk.Entry,
            textvariable=self.img_scale_var, width=6)

        self.img_stride_var = tk.IntVar(value=3)
        row("Image stride", ttk.Spinbox, from_=1, to=10,
            textvariable=self.img_stride_var, width=6)

        _h(f, "SIFT")
        self.sift_features_var = tk.IntVar(value=16000)
        row("Max features", ttk.Spinbox, from_=1000, to=65536,
            textvariable=self.sift_features_var, increment=1000, width=8)

        self.sift_peak_var = tk.StringVar(value="0.0045")
        row("Peak threshold", ttk.Entry,
            textvariable=self.sift_peak_var, width=8)

        self.sift_edge_var = tk.StringVar(value="12")
        row("Edge threshold", ttk.Entry,
            textvariable=self.sift_edge_var, width=8)

        self.sift_dsp_var = tk.BooleanVar(value=True)
        row("Domain size pooling", ttk.Checkbutton, variable=self.sift_dsp_var)

        self.sift_affine_var = tk.BooleanVar(value=False)
        row("Estimate affine shape", ttk.Checkbutton,
            variable=self.sift_affine_var)

        _h(f, "Matching")
        self.match_guided_var = tk.BooleanVar(value=True)
        row("Guided matching", ttk.Checkbutton, variable=self.match_guided_var)

        self.match_max_var = tk.IntVar(value=65536)
        row("Max matches", ttk.Spinbox, from_=1000, to=131072,
            textvariable=self.match_max_var, increment=4096, width=8)

        _h(f, "Threads / Cache  (0 = auto)")
        for label, attr in [
            ("Extract threads",   "extract_threads_var"),
            ("Match threads",     "match_threads_var"),
            ("Mapper threads",    "mapper_threads_var"),
            ("Fusion threads",    "fusion_threads_var"),
            ("Patch cache (GB)",  "patch_cache_var"),
            ("Fusion cache (GB)", "fusion_cache_var"),
        ]:
            var = tk.IntVar(value=0)
            setattr(self, attr, var)
            row(label, ttk.Spinbox, from_=0, to=256, textvariable=var, width=6)

        _h(f, "Secondary camera")
        self.sec_rotate_deg_var  = tk.StringVar(value="180")
        self.sec_rotate_axis_var = tk.StringVar(value="primary_frame_x")
        self.sec_extra_x_var     = tk.StringVar(value="0")
        self.sec_extra_y_var     = tk.StringVar(value="0")
        self.sec_extra_z_var     = tk.StringVar(value="0")
        self.sec_translate_x_var = tk.StringVar(value="0")
        self.sec_translate_y_var = tk.StringVar(value="0")
        self.sec_translate_z_var = tk.StringVar(value="0")
        self.sec_align_mode_var  = tk.StringVar(value="auto")

        for label, var in [
            ("Rotate degrees",  self.sec_rotate_deg_var),
            ("Rotate axis",     self.sec_rotate_axis_var),
            ("Extra rotate X",  self.sec_extra_x_var),
            ("Extra rotate Y",  self.sec_extra_y_var),
            ("Extra rotate Z",  self.sec_extra_z_var),
            ("Translate X",     self.sec_translate_x_var),
            ("Translate Y",     self.sec_translate_y_var),
            ("Translate Z",     self.sec_translate_z_var),
        ]:
            row(label, ttk.Entry, textvariable=var, width=10)

        row("Align mode", ttk.Combobox, textvariable=self.sec_align_mode_var,
            values=["auto", "manual"], state="readonly", width=8)

    # ── Reconstruct tab ────────────────────────────────────────────────────────
    def _build_recon(self, f):
        def row(label, var, from_=0, to=10000000, increment=1,
                width=10, is_bool=False, is_float=False):
            r = ttk.Frame(f); r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=26, anchor=tk.W).pack(side=tk.LEFT)
            if is_bool:
                ttk.Checkbutton(r, variable=var).pack(side=tk.LEFT)
            elif is_float:
                ttk.Spinbox(r, from_=from_, to=to, textvariable=var,
                            increment=increment, width=width,
                            format="%.4f").pack(side=tk.LEFT)
            else:
                ttk.Spinbox(r, from_=from_, to=to, textvariable=var,
                            increment=increment, width=width).pack(side=tk.LEFT)

        _h(f, "Point cloud filtering")
        self.r_max_input_pts = tk.IntVar(value=0)
        self.r_outlier_nn    = tk.IntVar(value=32)
        self.r_outlier_std   = tk.DoubleVar(value=0.0)
        self.r_radius_nn     = tk.IntVar(value=0)
        self.r_radius_factor = tk.DoubleVar(value=2.2)
        row("Max input points",       self.r_max_input_pts, to=10000000)
        row("Outlier neighbors",      self.r_outlier_nn,    to=256)
        row("Outlier std ratio",      self.r_outlier_std,   to=20.0,
            increment=0.5, is_float=True)
        row("Radius outlier NB pts",  self.r_radius_nn,     to=256)
        row("Radius outlier factor",  self.r_radius_factor, to=20.0,
            increment=0.1, is_float=True)

        _h(f, "DBSCAN clustering")
        self.r_dbscan_max_pts = tk.IntVar(value=0)
        self.r_dbscan_min_pts = tk.IntVar(value=0)
        self.r_dbscan_eps     = tk.DoubleVar(value=2.2)
        self.r_dbscan_keep    = tk.IntVar(value=1)
        self.r_dbscan_ratio   = tk.DoubleVar(value=0.02)
        row("DBSCAN max points",        self.r_dbscan_max_pts, to=10000000)
        row("DBSCAN min points",        self.r_dbscan_min_pts, to=1000)
        row("DBSCAN eps factor",        self.r_dbscan_eps,     to=20.0,
            increment=0.1, is_float=True)
        row("DBSCAN keep largest",      self.r_dbscan_keep,    to=20)
        row("DBSCAN min cluster ratio", self.r_dbscan_ratio,   to=1.0,
            increment=0.005, is_float=True)

        _h(f, "Normals")
        self.r_normal_max_nn   = tk.IntVar(value=96)
        self.r_normal_orient_k = tk.IntVar(value=64)
        row("Normal max NN",   self.r_normal_max_nn,   to=512)
        row("Normal orient K", self.r_normal_orient_k, to=512)

        _h(f, "Poisson reconstruction")
        self.r_poisson_depth       = tk.IntVar(value=10)
        self.r_poisson_linear      = tk.BooleanVar(value=True)
        self.r_density_trim        = tk.DoubleVar(value=0.02)
        self.r_poisson_crop_scale  = tk.DoubleVar(value=1.05)
        row("Poisson depth",     self.r_poisson_depth,  to=14, from_=5)
        row("Poisson linear fit", self.r_poisson_linear, is_bool=True)
        row("Density trim quantile", self.r_density_trim, to=0.5,
            increment=0.001, is_float=True)
        row("Poisson crop scale",    self.r_poisson_crop_scale, to=2.0,
            increment=0.01, is_float=True)

        _h(f, "Hole reduction")
        self.r_hole_reduction = tk.BooleanVar(value=False)

        # A middle ground between the plain defaults and the earlier aggressive preset:
        # loosening density trim too far let noisy low-confidence surface through in
        # specular/bright spots (which also tend to be low point-density from poor
        # feature matching), hurting geometry there. The hole-filling pass below now
        # picks up the remaining true gaps, so this preset can stay more conservative.
        HOLE_REDUCTION_OFF = {"density_trim": 0.02, "poisson_crop_scale": 1.05, "normal_max_nn": 96}
        HOLE_REDUCTION_ON  = {"density_trim": 0.012, "poisson_crop_scale": 1.08, "normal_max_nn": 80}

        def _apply_hole_reduction():
            preset = HOLE_REDUCTION_ON if self.r_hole_reduction.get() else HOLE_REDUCTION_OFF
            self.r_density_trim.set(preset["density_trim"])
            self.r_poisson_crop_scale.set(preset["poisson_crop_scale"])
            self.r_normal_max_nn.set(preset["normal_max_nn"])

        hr_row = ttk.Frame(f); hr_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(
            hr_row,
            text="Reduce holes near rotation axis / smooth surfaces",
            variable=self.r_hole_reduction,
            command=_apply_hole_reduction,
        ).pack(side=tk.LEFT)
        ttk.Label(
            f,
            text=("Keeps a bit more low-density Poisson surface, widens the crop margin, "
                  "and uses more neighbors for normal estimation, so sparsely sampled "
                  "areas (rotation axis poles, textureless surfaces) are less likely to "
                  "be trimmed away as holes. Overrides density trim quantile, Poisson "
                  "crop scale, and normal max NN above. Pair with hole filling below for "
                  "the remaining gaps."),
            foreground=PAL["subtext"], wraplength=340, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        _h(f, "Hole filling")
        self.r_fill_holes       = tk.BooleanVar(value=False)
        self.r_fill_holes_ratio = tk.DoubleVar(value=0.3)
        self.r_fill_holes_passes = tk.IntVar(value=4)
        row("Fill hole size ratio", self.r_fill_holes_ratio, to=1.0,
            increment=0.01, is_float=True)
        row("Fill hole passes",     self.r_fill_holes_passes, to=10, from_=1)
        fh_row = ttk.Frame(f); fh_row.pack(fill=tk.X, pady=2)
        ttk.Checkbutton(
            fh_row,
            text="Fill remaining holes (boundary triangulation)",
            variable=self.r_fill_holes,
        ).pack(side=tk.LEFT)
        ttk.Label(
            f,
            text=("After cleanup, triangulates any leftover boundary loops to bridge "
                  "gaps directly, rather than relying on loosening the density trim "
                  "threshold. Hole size ratio caps the max hole radius filled, as a "
                  "fraction of the cleaned cloud's bounding-box diagonal — raise it to "
                  "close bigger gaps, lower it to avoid bridging genuinely unscanned "
                  "regions. The fill pass doesn't always converge in one call — "
                  "closing one loop can make a neighboring loop fillable only on the "
                  "next pass — so it repeats up to \"Fill hole passes\" times, "
                  "stopping early once a pass closes nothing more."),
            foreground=PAL["subtext"], wraplength=340, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

        _h(f, "Mesh cleanup")
        self.r_comp_min_ratio = tk.DoubleVar(value=0.01)
        self.r_comp_min_tris  = tk.IntVar(value=1000)
        self.r_comp_max_count = tk.IntVar(value=1)
        self.r_smooth_iters   = tk.IntVar(value=1)
        self.r_decimate_tris  = tk.IntVar(value=300000)
        row("Component min ratio",     self.r_comp_min_ratio, to=1.0,
            increment=0.005, is_float=True)
        row("Component min triangles", self.r_comp_min_tris,  to=100000,
            increment=100)
        row("Component max count",     self.r_comp_max_count, to=50)
        row("Smooth iterations",       self.r_smooth_iters,   to=20)
        row("Decimate target tris",    self.r_decimate_tris,  to=5000000,
            increment=50000)

        _h(f, "Simplified export (for website)")
        self.r_simplified_target_verts = tk.IntVar(value=60000)
        row("Simplified target verts", self.r_simplified_target_verts,
            to=2000000, increment=5000)
        ttk.Label(
            f,
            text=("Always also exports \"<name>_simplified.glb\": a single small "
                  "binary glTF file, decimated to roughly this many vertices, with "
                  "vertex colors baked in (no separate texture/MTL files) — easy to "
                  "drop into a web viewer. Set to 0 to disable."),
            foreground=PAL["subtext"], wraplength=340, justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))

    # ── Alignment tab ──────────────────────────────────────────────────────────
    def _build_align(self, f):
        def browse_ply(var):
            fp = filedialog.askopenfilename(
                title="Select PLY file",
                filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
            if fp:
                var.set(fp)

        _h(f, "FPFH alignment settings")

        r1 = ttk.Frame(f); r1.pack(fill=tk.X, pady=2)
        ttk.Label(r1, text="Method:", width=14).pack(side=tk.LEFT)
        self.align_method_var = tk.StringVar(value="fpfh")
        ttk.Combobox(r1, textvariable=self.align_method_var,
                     values=["opening", "fpfh", "collapse", "all"],
                     state="readonly", width=10).pack(side=tk.LEFT, padx=4)

        r2 = ttk.Frame(f); r2.pack(fill=tk.X, pady=2)
        ttk.Label(r2, text="Voxel (0=auto):", width=14).pack(side=tk.LEFT)
        self.align_voxel_var = tk.StringVar(value="0")
        ttk.Entry(r2, textvariable=self.align_voxel_var,
                  width=8).pack(side=tk.LEFT, padx=4)

        r3 = ttk.Frame(f); r3.pack(fill=tk.X, pady=2)
        ttk.Label(r3, text="Sample points:", width=14).pack(side=tk.LEFT)
        self.align_samples_var = tk.IntVar(value=60000)
        ttk.Spinbox(r3, from_=5000, to=500000,
                    textvariable=self.align_samples_var,
                    increment=5000, width=8).pack(side=tk.LEFT, padx=4)

        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(f, text="PLY overrides (blank = auto from pipeline):",
                  font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(f, text="Useful for re-running alignment with different inputs.",
                  foreground=PAL["subtext"]).pack(anchor=tk.W, pady=(0, 6))

        for label, attr in [("PLY A (fixed):", "align_a_var"),
                             ("PLY B (moving):", "align_b_var")]:
            r = ttk.Frame(f); r.pack(fill=tk.X, pady=2)
            ttk.Label(r, text=label, width=14).pack(side=tk.LEFT)
            var = tk.StringVar()
            setattr(self, attr, var)
            ttk.Entry(r, textvariable=var).pack(side=tk.LEFT, fill=tk.X,
                                                  expand=True, padx=(2, 2))
            ttk.Button(r, text="…", width=3,
                       command=lambda v=var: browse_ply(v)).pack(side=tk.LEFT)

    # ── data extraction helpers ────────────────────────────────────────────────
    def get_colmap_env(self) -> dict:
        env = os.environ.copy()
        env["FIPMESH_COLMAP_QUALITY"]               = self.quality_var.get()
        env["FIPMESH_COLMAP_USE_GPU"]               = "1" if self.use_gpu_var.get() else "0"
        env["FIPMESH_COLMAP_GPU_INDEX"]             = self.gpu_index_var.get()
        env["FIPMESH_COLMAP_IMAGE_SCALE"]           = self.img_scale_var.get()
        # Stride is applied by process_photos.py (stage 1), so COLMAP sees
        # only the already-filtered images and must use them all.
        env["FIPMESH_COLMAP_IMAGE_STRIDE"]          = "1"
        env["FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES"] = str(self.sift_features_var.get())
        env["FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD"]   = self.sift_peak_var.get()
        env["FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD"]   = self.sift_edge_var.get()
        env["FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING"]   = "1" if self.sift_dsp_var.get() else "0"
        env["FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE"] = "1" if self.sift_affine_var.get() else "0"
        env["FIPMESH_COLMAP_MATCH_GUIDED"]           = "1" if self.match_guided_var.get() else "0"
        env["FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES"]  = str(self.match_max_var.get())
        env["FIPMESH_SKIP_RECON"]                    = "1"

        for attr, env_key in [
            ("extract_threads_var", "FIPMESH_COLMAP_EXTRACT_THREADS"),
            ("match_threads_var",   "FIPMESH_COLMAP_MATCH_THREADS"),
            ("mapper_threads_var",  "FIPMESH_COLMAP_MAPPER_THREADS"),
            ("fusion_threads_var",  "FIPMESH_COLMAP_FUSION_THREADS"),
            ("patch_cache_var",     "FIPMESH_COLMAP_PATCH_CACHE_SIZE"),
            ("fusion_cache_var",    "FIPMESH_COLMAP_FUSION_CACHE_SIZE"),
        ]:
            val = getattr(self, attr).get()
            if val > 0:
                env[env_key] = str(val)

        env["FIPMESH_COLMAP_SECONDARY_ROTATE_DEG"]     = self.sec_rotate_deg_var.get()
        env["FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS"]    = self.sec_rotate_axis_var.get()
        env["FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X"] = self.sec_extra_x_var.get()
        env["FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y"] = self.sec_extra_y_var.get()
        env["FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z"] = self.sec_extra_z_var.get()
        env["FIPMESH_COLMAP_SECONDARY_TRANSLATE_X"]    = self.sec_translate_x_var.get()
        env["FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y"]    = self.sec_translate_y_var.get()
        env["FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z"]    = self.sec_translate_z_var.get()
        env["FIPMESH_COLMAP_SECONDARY_ALIGN_MODE"]     = self.sec_align_mode_var.get()
        return env

    def get_recon_cmd(self, input_ply: str, out_obj: str,
                      clean_cloud: str, decimated: str) -> list[str]:
        py = _find_python(SCRIPT_DIR / "venv" / "bin" / "python3")
        cmd = [
            py,
            str(SCRIPT_DIR / "src" / "reconstruct_mesh.py"),
            "--input",                    input_ply,
            "--output",                   out_obj,
            "--output-clean-cloud",       clean_cloud,
            "--output-decimated",         decimated,
            "--max-input-points",         str(self.r_max_input_pts.get()),
            "--dbscan-max-points",        str(self.r_dbscan_max_pts.get()),
            "--outlier-nb-neighbors",     str(self.r_outlier_nn.get()),
            "--outlier-std-ratio",        str(self.r_outlier_std.get()),
            "--radius-outlier-nb-points", str(self.r_radius_nn.get()),
            "--radius-outlier-radius-factor", str(self.r_radius_factor.get()),
            "--dbscan-min-points",        str(self.r_dbscan_min_pts.get()),
            "--dbscan-eps-factor",        str(self.r_dbscan_eps.get()),
            "--dbscan-keep-largest",      str(self.r_dbscan_keep.get()),
            "--dbscan-min-cluster-ratio", str(self.r_dbscan_ratio.get()),
            "--normal-max-nn",            str(self.r_normal_max_nn.get()),
            "--normal-orient-k",          str(self.r_normal_orient_k.get()),
            "--poisson-depth",            str(self.r_poisson_depth.get()),
            "--density-trim-quantile",    str(self.r_density_trim.get()),
            "--poisson-crop-scale",       str(self.r_poisson_crop_scale.get()),
            "--fill-holes",               "1" if self.r_fill_holes.get() else "0",
            "--fill-holes-max-size-ratio", str(self.r_fill_holes_ratio.get()),
            "--fill-holes-passes",        str(self.r_fill_holes_passes.get()),
            "--simplified-target-vertices", str(self.r_simplified_target_verts.get()),
            "--component-min-ratio",      str(self.r_comp_min_ratio.get()),
            "--component-min-triangles",  str(self.r_comp_min_tris.get()),
            "--component-max-count",      str(self.r_comp_max_count.get()),
            "--smooth-iters",             str(self.r_smooth_iters.get()),
            "--decimate-target-triangles", str(self.r_decimate_tris.get()),
        ]
        if self.r_poisson_linear.get():
            cmd.append("--poisson-linear-fit")
        return cmd


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline Panel (right side)
# ══════════════════════════════════════════════════════════════════════════════

class PipelinePanel(ttk.Frame):
    VIEW_LABELS = [
        "View Photos",
        "View Cloud",
        "View Merged",
        "View Mesh",
    ]

    def __init__(self, parent, app: "App"):
        super().__init__(parent)
        self.app = app
        self._stages: list[StageWidget] = []
        self._build()

    def _build(self):
        for i, (name, view_lbl) in enumerate(zip(STAGE_NAMES, self.VIEW_LABELS)):
            sw = StageWidget(
                self, i, name, view_lbl,
                run_cb=lambda idx=i: self.app.run_stage(idx),
                view_cb=lambda idx=i: self.app.view_stage(idx),
            )
            sw.pack(fill=tk.X, padx=4, pady=3)
            self._stages.append(sw)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=6)

        bf = ttk.Frame(self)
        bf.pack(fill=tk.X, padx=4, pady=(4, 4))
        ttk.Button(bf, text="▶  Run All Stages",
                   command=self.app.run_all).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bf, text="■  Stop",
                   command=self.app.stop_all).pack(side=tk.LEFT)

    def stage(self, idx: int) -> StageWidget:
        return self._stages[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Photo gallery viewer (Stage 1 view)
# ══════════════════════════════════════════════════════════════════════════════

class PhotoGallery(tk.Toplevel):
    THUMB = 160

    def __init__(self, parent, processed_dir: Path):
        super().__init__(parent)
        self.title("Processed Photos")
        self.geometry("960x620")
        self._images: list = []

        canvas = tk.Canvas(self, bg="#1e293b")
        sb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        inner = tk.Frame(canvas, bg="#1e293b")
        canvas.create_window((0, 0), window=inner, anchor=tk.NW)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        try:
            from PIL import Image, ImageTk
            files = sorted(
                f for f in processed_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            )[:200]
            cols = max(1, 960 // (self.THUMB + 8))
            for idx, fp in enumerate(files):
                try:
                    img = Image.open(fp)
                    img.thumbnail((self.THUMB, self.THUMB))
                    tk_img = ImageTk.PhotoImage(img)
                    self._images.append(tk_img)
                    col = idx % cols
                    row = idx // cols
                    tk.Label(inner, image=tk_img, bg="#1e293b",
                             cursor="hand2").grid(
                        row=row * 2, column=col, padx=4, pady=4)
                    tk.Label(inner, text=fp.name[:20], bg="#1e293b",
                             fg="#94a3b8", font=("Courier", 7)).grid(
                        row=row * 2 + 1, column=col, sticky="n")
                except Exception:
                    pass
            if not files:
                tk.Label(inner, text="No images found", bg="#1e293b",
                         fg="#64748b", font=("Segoe UI", 14)).pack(
                    padx=40, pady=40)
        except ImportError:
            tk.Label(inner,
                     text="Install Pillow (pip install pillow) to view photos.",
                     bg="#1e293b", fg="#64748b",
                     font=("Segoe UI", 12)).pack(padx=40, pady=40)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Automatic Tablet Reconstruction")
        self.minsize(1050, 740)

        apply_theme(self)

        self._log_q:      queue.Queue[str] = queue.Queue()
        self._proc:       subprocess.Popen | None = None
        self._stop_req:   bool = False
        self._run_thread: threading.Thread | None = None

        self._build_menu()
        self._build_layout()
        self.after(80, self._poll_log)
        self._reconcile_stage_states()

    # ── menu ──────────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = tk.Menu(self)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open session folder",
                       command=self._open_session_folder)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        mb.add_cascade(label="File", menu=fm)
        self.config(menu=mb)

    # ── layout (body + resizable log) ─────────────────────────────────────────
    def _build_layout(self):
        # Vertical paned window: upper body / lower log (draggable sash)
        vpw = ttk.PanedWindow(self, orient=tk.VERTICAL)
        vpw.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Upper pane: left config notebook + right pipeline panel
        upper = ttk.Frame(vpw)
        vpw.add(upper, weight=3)

        pw = ttk.PanedWindow(upper, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(pw, width=420)
        self._cfg = ConfigPanel(left_frame, self)
        self._cfg.pack(fill=tk.BOTH, expand=True)
        pw.add(left_frame, weight=2)

        right_frame = ttk.Frame(pw)
        self._pipe = PipelinePanel(right_frame, self)
        self._pipe.pack(fill=tk.BOTH, expand=True)
        pw.add(right_frame, weight=3)

        # Lower pane: output log
        log_frame = ttk.LabelFrame(vpw, text="Output Log", padding=4)
        vpw.add(log_frame, weight=1)

        self._log_widget = scrolledtext.ScrolledText(
            log_frame, height=11, font=("Courier", 8),
            state=tk.DISABLED,
            background=PAL["log_bg"], foreground=PAL["log_fg"],
        )
        self._log_widget.tag_configure("header",
                                        foreground=PAL["log_hdr"],
                                        font=("Courier", 8, "bold"))
        self._log_widget.pack(fill=tk.BOTH, expand=True)

    # ── logging (thread-safe) ─────────────────────────────────────────────────
    def log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_q.put(f"[{ts}] {msg}")

    def _poll_log(self):
        try:
            while True:
                line = self._log_q.get_nowait()
                self._log_widget.config(state=tk.NORMAL)
                if line.startswith("\x00HEADER\x00"):
                    text = line[len("\x00HEADER\x00"):]
                    self._log_widget.insert(tk.END, "\n── " + text + "\n", "header")
                else:
                    self._log_widget.insert(tk.END, line + "\n")
                self._log_widget.see(tk.END)
                self._log_widget.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    # ── stage state helpers ───────────────────────────────────────────────────
    def _set_stage(self, idx: int, state: str, status: str = ""):
        self.after(0, lambda: self._pipe.stage(idx).set_state(state, status))

    def _stage_status(self, idx: int, text: str):
        self.after(0, lambda: self._pipe.stage(idx).set_status_text(text))

    def _expected_output_for_stage(self, idx: int) -> Path:
        s1, _ = self._active_sides()
        if idx == 0:
            return self._processed_dir()
        if idx == 1:
            return (self._colmap_dir(s1) if s1 else self._colmap_dir()) / "fused.ply"
        if idx == 2:
            return self._merged_ply()
        return self._recon_dir() / "recon_mesh_recon.obj"

    def _stage_output_ready(self, idx: int) -> bool:
        p = self._expected_output_for_stage(idx)
        try:
            if idx == 0:
                return p.is_dir() and _has_images(p)
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False

    def _reconcile_stage_states(self):
        for idx in range(4):
            if self._pipe.stage(idx).state == "running":
                continue
            if self._stage_output_ready(idx):
                self._set_stage(idx, "done")

    # ── path helpers ──────────────────────────────────────────────────────────
    def _session_dir(self) -> Path:
        out = self._cfg.output_var.get().strip()
        if out:
            return Path(out)
        inp = self._cfg.input_var.get().strip()
        if inp:
            p = Path(inp)
            return p.parent / (p.name + "_recon")
        return Path.home() / "tablet_recon"

    def _processed_dir(self) -> Path:
        return self._session_dir() / "processed"

    def _colmap_dir(self, side: str | None = None) -> Path:
        if side:
            return self._session_dir() / f"colmap_{side}"
        return self._session_dir() / "colmap_out"

    def _merged_ply(self) -> Path:
        return self._session_dir() / "aligned_cloud" / "merged_fpfh.ply"

    def _recon_dir(self) -> Path:
        return self._session_dir() / "recon"

    def _active_sides(self) -> tuple[str, str]:
        return (self._cfg.side1_var.get().strip(),
                self._cfg.side2_var.get().strip())

    def _input_ply_for_recon(self) -> Path:
        s1, s2 = self._active_sides()
        if s2:
            return self._merged_ply()
        return self._colmap_dir(s1 if s1 else None) / "fused.ply"

    def _open_session_folder(self):
        d = self._session_dir()
        if sys.platform == "win32":
            os.startfile(str(d))
        else:
            subprocess.Popen(["xdg-open", str(d)])

    # ── subprocess runner ─────────────────────────────────────────────────────
    def _run_proc(self, cmd: list, cwd: Path,
                  env: dict | None = None, on_line=None) -> int:
        """Run subprocess, stream all output to log. on_line(line) per line."""
        self._log_q.put(f"\x00HEADER\x00$ {' '.join(str(c) for c in cmd)}")
        e = {**(env or os.environ.copy()), "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=e,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in self._proc.stdout:
            if self._stop_req:
                self._proc.kill()
                break
            stripped = line.rstrip()
            self._log_q.put(stripped)
            if on_line and stripped:
                on_line(stripped)
        self._proc.wait()
        rc = self._proc.returncode
        self._proc = None
        return rc

    # ── Stop ──────────────────────────────────────────────────────────────────
    def stop_all(self):
        self._stop_req = True
        if self._proc:
            try:
                self._proc.kill()
            except Exception:
                pass
        self.log("[pipeline] stopped by user")

    # ── View ──────────────────────────────────────────────────────────────────
    def view_stage(self, idx: int):
        if idx == 0:
            PhotoGallery(self, self._processed_dir())
        else:
            self._spawn_viewer(self._expected_output_for_stage(idx))

    def _spawn_viewer(self, path: Path):
        if not path.exists():
            messagebox.showwarning("File not found", f"Not found:\n{path}")
            return
        viewer_py = _find_python(SCRIPT_DIR / "venv" / "bin" / "python3")
        try:
            proc = subprocess.Popen(
                [viewer_py, str(SCRIPT_DIR / "viewer.py"), str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            self.log(f"[viewer] interpreter not found: {viewer_py!r}")
            messagebox.showerror("Viewer failed",
                                  f"Could not launch viewer: interpreter not found:\n{viewer_py}")
            return

        def _watch():
            err = proc.stderr.read()
            proc.wait()
            if proc.returncode != 0 and err.strip():
                self.log(f"[viewer] {err.strip()}")
                self.after(0, lambda: messagebox.showerror(
                    "Viewer failed", f"Viewer exited with an error:\n{err.strip()}"))

        threading.Thread(target=_watch, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage dispatch
    # ══════════════════════════════════════════════════════════════════════════

    def run_stage(self, idx: int):
        if self._run_thread and self._run_thread.is_alive():
            messagebox.showinfo("Busy", "Another stage is already running.")
            return
        self._stop_req = False
        self._run_thread = threading.Thread(
            target=self._stage_worker, args=(idx,), daemon=True)
        self._run_thread.start()

    def run_all(self):
        if self._run_thread and self._run_thread.is_alive():
            messagebox.showinfo("Busy", "Pipeline is already running.")
            return
        self._stop_req = False
        self._run_thread = threading.Thread(
            target=self._run_all_worker, daemon=True)
        self._run_thread.start()

    def _run_all_worker(self):
        for idx in range(4):
            if self._stop_req:
                break
            ok = self._stage_worker(idx)
            if not ok:
                self.log(f"[pipeline] stopped after stage {idx + 1} failed")
                break

    def _stage_worker(self, idx: int) -> bool:
        runners = [
            self._run_stage_1_bg_removal,
            self._run_stage_2_colmap,
            self._run_stage_3_alignment,
            self._run_stage_4_reconstruction,
        ]
        sw = self._pipe.stage(idx)
        self._set_stage(idx, "running")

        def on_progress(pct: int, text: str = ""):
            self.after(0, sw.set_progress, pct, text)

        try:
            ok = runners[idx](on_progress)
        except Exception as exc:
            self.log(f"[error] stage {idx + 1}: {exc}")
            ok = False

        self._set_stage(idx, "done" if ok else "failed")
        return ok

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 1 — Background removal
    # ══════════════════════════════════════════════════════════════════════════
    def _run_stage_1_bg_removal(self, on_progress) -> bool:
        inp = self._cfg.input_var.get().strip()
        if not inp:
            self.log("[error] No input directory specified")
            return False

        out = self._processed_dir()
        out.mkdir(parents=True, exist_ok=True)

        py = _find_python(SCRIPT_DIR / "venv" / "bin" / "python3")
        cmd = [
            py,
            str(SCRIPT_DIR / "process_photos.py"),
            "--input",      inp,
            "--output",     str(out),
            "--model",      self._cfg.model_var.get(),
            "--background", self._cfg.bg_var.get(),
        ]
        thresh = self._cfg.black_thresh_var.get()
        if thresh > 0:
            cmd += ["--black-threshold", str(thresh)]
        white_thresh = self._cfg.white_thresh_var.get()
        if white_thresh > 0:
            cmd += ["--white-threshold", str(white_thresh)]
        value_thresh = self._cfg.value_thresh_var.get()
        if value_thresh > 0:
            cmd += ["--value-threshold", str(value_thresh)]
        edge_band = self._cfg.edge_band_var.get()
        if edge_band > 0:
            cmd += ["--edge-band", str(edge_band)]
        passes = self._cfg.passes_var.get()
        if passes > 1:
            cmd += ["--passes", str(passes)]
        stride = self._cfg.img_stride_var.get()
        if stride > 1:
            cmd += ["--stride", str(stride)]
        seg_scale_pct = self._cfg.seg_scale_var.get()
        if seg_scale_pct < 100:
            cmd += ["--seg-scale-pct", str(seg_scale_pct)]

        self.log(f"[stage 1] Background removal: {inp} → {out}")
        parser = PhotosParser()

        def on_line(line: str):
            r = parser.feed(line)
            if r:
                on_progress(*r)

        rc = self._run_proc(cmd, cwd=SCRIPT_DIR, on_line=on_line)
        if rc == 0:
            on_progress(100, "Complete")
        return rc == 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 2 — COLMAP MVS
    # ══════════════════════════════════════════════════════════════════════════
    def _run_stage_2_colmap(self, on_progress) -> bool:
        s1, s2 = self._active_sides()
        processed = self._processed_dir()
        env = self._cfg.get_colmap_env()

        if s1 and s2:
            self.log(f"[stage 2] COLMAP side 1: {s1}")
            parser1 = ColmapParser(0, 50)

            def on_line1(line: str):
                r = parser1.feed(line)
                if r:
                    on_progress(*r)

            ok = self._colmap_one_side(s1, processed, env, on_line=on_line1)
            if not ok or self._stop_req:
                return False

            self.log(f"[stage 2] COLMAP side 2: {s2}")
            parser2 = ColmapParser(50, 100)

            def on_line2(line: str):
                r = parser2.feed(line)
                if r:
                    on_progress(*r)

            ok = self._colmap_one_side(s2, processed, env, on_line=on_line2)
            return ok

        elif s1:
            self.log(f"[stage 2] COLMAP single side: {s1}")
            parser = ColmapParser(0, 100)

            def on_line(line: str):
                r = parser.feed(line)
                if r:
                    on_progress(*r)

            return self._colmap_one_side(s1, processed, env, on_line=on_line)

        else:
            self.log("[stage 2] COLMAP flat image set")
            parser = ColmapParser(0, 100)

            def on_line(line: str):
                r = parser.feed(line)
                if r:
                    on_progress(*r)

            return self._colmap_flat(processed, env, on_line=on_line)

    def _colmap_one_side(self, side: str, processed_root: Path,
                          env: dict, on_line=None) -> bool:
        img_dir = processed_root / side
        out_dir = self._colmap_dir(side)
        out_dir.mkdir(parents=True, exist_ok=True)
        # run.sh prepends its own SCRIPT_DIR to -i/-o, so pass relative paths
        rel_img = os.path.relpath(img_dir, SCRIPT_DIR)
        rel_out = os.path.relpath(out_dir, SCRIPT_DIR)
        cmd = ["bash", str(SCRIPT_DIR / "run.sh"),
               "-i", rel_img, "-o", rel_out, "-v"]
        rc = self._run_proc(cmd, cwd=SCRIPT_DIR, env=env, on_line=on_line)
        return rc == 0

    def _colmap_flat(self, processed_dir: Path, env: dict,
                      on_line=None) -> bool:
        out_dir = self._colmap_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        rel_img = os.path.relpath(processed_dir, SCRIPT_DIR)
        rel_out = os.path.relpath(out_dir, SCRIPT_DIR)
        cmd = ["bash", str(SCRIPT_DIR / "run.sh"),
               "-i", rel_img, "-o", rel_out, "-v"]
        rc = self._run_proc(cmd, cwd=SCRIPT_DIR, env=env, on_line=on_line)
        return rc == 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 3 — FPFH Alignment
    # ══════════════════════════════════════════════════════════════════════════
    def _run_stage_3_alignment(self, on_progress) -> bool:
        s1, s2 = self._active_sides()

        ply_a = self._cfg.align_a_var.get().strip()
        ply_b = self._cfg.align_b_var.get().strip()

        if not ply_a and not ply_b:
            if not s2:
                self.log("[stage 3] Single-side scan — alignment skipped")
                on_progress(100, "Skipped (single side)")
                return True
            ply_a = str(self._colmap_dir(s1) / "fused.ply")
            ply_b = str(self._colmap_dir(s2) / "fused.ply")

        for p in (ply_a, ply_b):
            if not Path(p).exists():
                self.log(f"[error] stage 3: PLY not found: {p}")
                return False

        merged_out = self._merged_ply()
        merged_out.parent.mkdir(parents=True, exist_ok=True)

        align_py = _find_python(SCRIPT_DIR / "venv" / "bin" / "python3")
        cmd = [
            align_py, str(ALIGN_DIR / "run.py"),
            ply_a, ply_b,
            "-o", str(merged_out),
            "--method", self._cfg.align_method_var.get(),
            "--samples", str(self._cfg.align_samples_var.get()),
        ]
        voxel = self._cfg.align_voxel_var.get().strip()
        if voxel and voxel != "0":
            cmd += ["--voxel", voxel]

        self.log(f"[stage 3] Aligning {Path(ply_a).name} + {Path(ply_b).name}")
        parser = AlignParser()

        def on_line(line: str):
            r = parser.feed(line)
            if r:
                on_progress(*r)

        rc = self._run_proc(cmd, cwd=ALIGN_DIR, on_line=on_line)
        if rc == 0:
            on_progress(100, "Alignment complete")
        return rc == 0

    # ══════════════════════════════════════════════════════════════════════════
    #  Stage 4 — Mesh Reconstruction
    # ══════════════════════════════════════════════════════════════════════════
    def _run_stage_4_reconstruction(self, on_progress) -> bool:
        input_ply = self._input_ply_for_recon()
        if not input_ply.exists():
            self.log(f"[error] stage 4: input PLY not found: {input_ply}")
            return False

        recon_dir = self._recon_dir()
        recon_dir.mkdir(parents=True, exist_ok=True)
        out_obj       = recon_dir / "recon_mesh_recon.obj"
        clean_cloud   = recon_dir / "clean_cloud.ply"
        decimated_obj = recon_dir / "recon_mesh_recon_decimated.obj"

        cmd = self._cfg.get_recon_cmd(
            str(input_ply), str(out_obj), str(clean_cloud), str(decimated_obj))
        self.log(f"[stage 4] Reconstructing mesh from {input_ply.name}…")
        parser = ReconParser()

        def on_line(line: str):
            r = parser.feed(line)
            if r:
                on_progress(*r)

        rc = self._run_proc(cmd, cwd=SCRIPT_DIR, on_line=on_line)
        if rc != 0:
            return False

        # Copy auto-generated GLTF to the session root for easy access
        src_gltf  = recon_dir / "recon_mesh_recon.gltf"
        dest_gltf = self._session_dir() / "model.gltf"
        if src_gltf.exists():
            shutil.copy2(src_gltf, dest_gltf)
            self.log(f"[stage 4] GLTF → {dest_gltf}")
        else:
            self.log(f"[stage 4] Warning: GLTF not found at {src_gltf}")

        # Copy the decimated web-viewer GLB to the session root too
        src_simplified_glb  = recon_dir / "recon_mesh_recon_simplified.glb"
        dest_simplified_glb = self._session_dir() / "model_simplified.glb"
        if src_simplified_glb.exists():
            shutil.copy2(src_simplified_glb, dest_simplified_glb)
            self.log(f"[stage 4] Simplified GLB → {dest_simplified_glb}")

        on_progress(100, "Reconstruction complete")
        return True


# ══════════════════════════════════════════════════════════════════════════════
#  Layout helpers
# ══════════════════════════════════════════════════════════════════════════════

def _scroll_frame(parent: tk.Widget) -> ttk.Frame:
    canvas = tk.Canvas(parent, highlightthickness=0, background=PAL["bg"])
    sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(fill=tk.BOTH, expand=True)
    inner = ttk.Frame(canvas)
    win_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>",
                lambda e: canvas.itemconfig(win_id, width=e.width))

    def _scroll(e):
        canvas.yview_scroll(
            -1 * (e.delta // 120 if e.delta else (-1 if e.num == 4 else 1)),
            "units")
    canvas.bind("<MouseWheel>", _scroll)
    canvas.bind("<Button-4>", _scroll)
    canvas.bind("<Button-5>", _scroll)
    return inner


def _h(parent: tk.Widget, text: str):
    ttk.Label(parent, text=text,
              font=("Segoe UI", 9, "bold"),
              foreground=PAL["subtext"]).pack(anchor=tk.W, pady=(8, 1))


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
