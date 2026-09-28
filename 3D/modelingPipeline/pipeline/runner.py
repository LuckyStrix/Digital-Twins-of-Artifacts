"""Stage runners: build each stage's command line, run it, report progress.

``PipelineRunner`` is UI-independent. It reports through three callbacks,
all called from the worker thread (a UI must marshal them onto its own
thread, e.g. Textual's ``app.call_from_thread``):

    on_log(text, kind)              kind: "info"   – app message, timestamped
                                          "header" – "$ command ..." line
                                          "output" – subprocess stdout/stderr
    on_stage_state(idx, state, status)    state: waiting/running/done/failed
    on_progress(idx, pct, text)

Runs started with ``run_stage``/``run_all`` are also recorded in the session
folder (``pipeline.log`` and ``pipeline_runs.json``; see ``runlog``).
"""

from __future__ import annotations

import copy
import os
import shutil
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping

from . import ALIGN_DIR, SCRIPT_DIR
from .parsers import AlignParser, ColmapParser, PhotosParser, ReconParser
from .paths import SessionPaths
from .platform import venv_python
from .runlog import RunRecorder

LogFn = Callable[[str, str], None]
StateFn = Callable[[int, str, str], None]
ProgressFn = Callable[[int, int, str], None]

# After SIGTERM, how long a stopped process group gets before SIGKILL.
STOP_GRACE_SECONDS = 3.0

_POSIX = os.name == "posix"


def _flag(v) -> str:
    return "1" if v else "0"


# ── command / env construction (pure functions of the settings) ───────────────

def colmap_env(s: Mapping, base_env: Mapping | None = None) -> dict:
    env = dict(os.environ if base_env is None else base_env)
    env["FIPMESH_COLMAP_QUALITY"]               = s["quality_var"]
    env["FIPMESH_COLMAP_USE_GPU"]               = _flag(s["use_gpu_var"])
    env["FIPMESH_COLMAP_GPU_INDEX"]             = s["gpu_index_var"]
    env["FIPMESH_COLMAP_IMAGE_SCALE"]           = s["img_scale_var"]
    # Stride is applied by process_photos.py (stage 1), so COLMAP sees
    # only the already-filtered images and must use them all.
    env["FIPMESH_COLMAP_IMAGE_STRIDE"]          = "1"
    env["FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES"] = str(s["sift_features_var"])
    env["FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD"]   = s["sift_peak_var"]
    env["FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD"]   = s["sift_edge_var"]
    env["FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING"]   = _flag(s["sift_dsp_var"])
    env["FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE"] = _flag(s["sift_affine_var"])
    env["FIPMESH_COLMAP_MATCH_GUIDED"]           = _flag(s["match_guided_var"])
    env["FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES"]  = str(s["match_max_var"])
    env["FIPMESH_SKIP_RECON"]                    = "1"

    for key, env_key in [
        ("extract_threads_var", "FIPMESH_COLMAP_EXTRACT_THREADS"),
        ("match_threads_var",   "FIPMESH_COLMAP_MATCH_THREADS"),
        ("mapper_threads_var",  "FIPMESH_COLMAP_MAPPER_THREADS"),
        ("fusion_threads_var",  "FIPMESH_COLMAP_FUSION_THREADS"),
        ("patch_cache_var",     "FIPMESH_COLMAP_PATCH_CACHE_SIZE"),
        ("fusion_cache_var",    "FIPMESH_COLMAP_FUSION_CACHE_SIZE"),
    ]:
        val = s[key]
        if val > 0:
            env[env_key] = str(val)

    env["FIPMESH_COLMAP_SECONDARY_ROTATE_DEG"]     = s["sec_rotate_deg_var"]
    env["FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS"]    = s["sec_rotate_axis_var"]
    env["FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X"] = s["sec_extra_x_var"]
    env["FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y"] = s["sec_extra_y_var"]
    env["FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z"] = s["sec_extra_z_var"]
    env["FIPMESH_COLMAP_SECONDARY_TRANSLATE_X"]    = s["sec_translate_x_var"]
    env["FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y"]    = s["sec_translate_y_var"]
    env["FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z"]    = s["sec_translate_z_var"]
    env["FIPMESH_COLMAP_SECONDARY_ALIGN_MODE"]     = s["sec_align_mode_var"]
    cam_model = s["camera_model_var"].strip()
    if cam_model and not cam_model.startswith("("):
        env["FIPMESH_COLMAP_CAMERA_MODEL"] = cam_model
    return env


def photos_cmd(s: Mapping, inp: str, out: Path) -> list[str]:
    cmd = [
        venv_python(),
        str(SCRIPT_DIR / "process_photos.py"),
        "--input",      inp,
        "--output",     str(out),
        "--model",      s["model_var"],
        "--background", s["bg_var"],
    ]
    if not s["hard_mask_var"]:
        cmd += ["--no-hard-mask"]
    for key, flag in [
        ("black_thresh_var",  "--black-threshold"),
        ("white_thresh_var",  "--white-threshold"),
        ("value_thresh_var",  "--value-threshold"),
        ("chroma_thresh_var", "--chroma-threshold"),
        ("edge_band_var",     "--edge-band"),
        ("erode_px_var",      "--erode-px"),
    ]:
        if s[key] > 0:
            cmd += [flag, str(s[key])]
    grow_chroma = s["grow_chroma_var"]
    if grow_chroma > 0:
        cmd += ["--grow-chroma", str(grow_chroma)]
        if s["grow_hull_fill_var"]:
            cmd += ["--grow-hull-fill"]
    passes = s["passes_var"]
    if passes > 1:
        cmd += ["--passes", str(passes)]
    stride = s["img_stride_var"]
    if stride > 1:
        cmd += ["--stride", str(stride)]
    seg_scale_pct = s["seg_scale_var"]
    if seg_scale_pct < 100:
        cmd += ["--seg-scale-pct", str(seg_scale_pct)]
    return cmd


def run_sh_cmd(img_dir: Path, out_dir: Path, mask_dir: Path | None) -> list[str]:
    # run.sh prepends its own SCRIPT_DIR to -i/-o/-m, so pass relative paths
    cmd = ["bash", str(SCRIPT_DIR / "run.sh"),
           "-i", os.path.relpath(img_dir, SCRIPT_DIR),
           "-o", os.path.relpath(out_dir, SCRIPT_DIR), "-v"]
    if mask_dir is not None:
        cmd += ["-m", os.path.relpath(mask_dir, SCRIPT_DIR)]
    return cmd


def align_cmd(s: Mapping, ply_a: str, ply_b: str, merged_out: Path,
              report_path: Path) -> list[str]:
    def text(key: str) -> str:
        return str(s[key]).strip()

    cmd = [
        venv_python(), str(ALIGN_DIR / "run.py"),
        ply_a, ply_b,
        "-o", str(merged_out),
        "--method", s["align_method_var"],
        "--samples", str(s["align_samples_var"]),
    ]
    voxel = text("align_voxel_var")
    if voxel and voxel != "0":
        cmd += ["--voxel", voxel]
    if s["align_seam_var"]:
        cmd += [
            "--resolve-seam",
            "--seam-mode", s["align_seam_mode_var"],
            "--seam-tau", text("align_seam_tau_var") or "0.3",
            "--seam-min-conf", text("align_seam_minconf_var") or "0",
            "--seam-window", text("align_seam_window_var") or "10",
            "--seam-min-count", str(s["align_seam_minc_var"]),
            "--seam-passes", str(s["align_seam_passes_var"]),
        ]
    if s["align_refine_var"]:
        cmd += [
            "--refine",
            "--refine-thresholds", text("align_thresholds_var"),
            "--refine-iters", str(s["align_iters_var"]),
            "--refine-points", str(s["align_points_var"]),
            "--refine-tol", text("align_tol_var") or "1e-7",
            "--refine-band", text("align_band_var") or "0",
            "--refine-robust", text("align_robust_var") or "0",
            "--report", str(report_path),
        ]
    return cmd


def recon_cmd(s: Mapping, input_ply: str, out_obj: str, clean_cloud: str,
              decimated: str, side1_camera_centers: str = "") -> list[str]:
    # --background transparent has no single flat fill colour to prune by
    # (the "background" there is the real original backdrop photo, not a
    # flat fill) — only white/black produce a colour worth pruning against.
    bg = s["bg_var"]
    prune_fill_color = bg if bg in ("white", "black") else "none"
    cmd = [
        venv_python(),
        str(SCRIPT_DIR / "src" / "reconstruct_mesh.py"),
        "--input",                    input_ply,
        "--output",                   out_obj,
        "--output-clean-cloud",       clean_cloud,
        "--output-decimated",         decimated,
        "--prune-fill-color",         prune_fill_color,
        "--max-input-points",         str(s["r_max_input_pts"]),
        "--dbscan-max-points",        str(s["r_dbscan_max_pts"]),
        "--outlier-nb-neighbors",     str(s["r_outlier_nn"]),
        "--outlier-std-ratio",        str(s["r_outlier_std"]),
        "--radius-outlier-nb-points", str(s["r_radius_nn"]),
        "--radius-outlier-radius-factor", str(s["r_radius_factor"]),
        "--dbscan-min-points",        str(s["r_dbscan_min_pts"]),
        "--dbscan-eps-factor",        str(s["r_dbscan_eps"]),
        "--dbscan-keep-largest",      str(s["r_dbscan_keep"]),
        "--dbscan-min-cluster-ratio", str(s["r_dbscan_ratio"]),
        "--normal-max-nn",            str(s["r_normal_max_nn"]),
        "--normal-orient-k",          str(s["r_normal_orient_k"]),
        "--poisson-depth",            str(s["r_poisson_depth"]),
        "--density-trim-quantile",    str(s["r_density_trim"]),
        "--poisson-crop-scale",       str(s["r_poisson_crop_scale"]),
        "--fill-holes",               _flag(s["r_fill_holes"]),
        "--fill-holes-max-size-ratio", str(s["r_fill_holes_ratio"]),
        "--fill-holes-passes",        str(s["r_fill_holes_passes"]),
        "--fill-holes-smooth",        _flag(s["r_fill_smooth"]),
        "--simplified-target-vertices", str(s["r_simplified_target_verts"]),
        "--component-min-ratio",      str(s["r_comp_min_ratio"]),
        "--component-min-triangles",  str(s["r_comp_min_tris"]),
        "--component-max-count",      str(s["r_comp_max_count"]),
        "--smooth-iters",             str(s["r_smooth_iters"]),
        "--decimate-target-triangles", str(s["r_decimate_tris"]),
        "--normalize-pose",           _flag(s["r_normalize_pose"]),
    ]
    if s["r_poisson_linear"]:
        cmd.append("--poisson-linear-fit")
    if side1_camera_centers:
        cmd += ["--side1-camera-centers", side1_camera_centers]
    return cmd


# ── runner ────────────────────────────────────────────────────────────────────

class PipelineRunner:
    def __init__(self, settings: Mapping,
                 on_log: LogFn | None = None,
                 on_stage_state: StateFn | None = None,
                 on_progress: ProgressFn | None = None):
        # Runs started with run_stage()/run_all() work on a snapshot taken at
        # start, so edits made in the UI mid-run can't move the session
        # folder or change flags under stages that are still to come.
        self._live_settings = settings
        self.settings = settings
        self.paths = SessionPaths(settings)
        self._ui_log = on_log or (lambda text, kind: None)
        self._ui_stage_state = on_stage_state or (lambda idx, state, status: None)
        self._ui_progress = on_progress or (lambda idx, pct, text: None)
        self._recorder: RunRecorder | None = None

        self._proc: subprocess.Popen | None = None
        self._stop_req = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ── public API ────────────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def run_stage(self, idx: int) -> bool:
        """Run one stage in a background thread. False if already running."""
        return self._start(lambda: self.execute_stage(idx), f"stage {idx + 1}")

    def run_all(self) -> bool:
        """Run all stages in a background thread. False if already running."""
        return self._start(self.execute_all, "all")

    def stop(self) -> None:
        self._stop_req = True
        proc = self._proc
        if proc:
            self._terminate(proc)
        self.log("[pipeline] stopped by user")

    def execute_all(self) -> bool:
        """Run stages 1-4 in order on the calling thread; stop at a failure."""
        for idx in range(4):
            if self._stop_req:
                return False
            if not self.execute_stage(idx):
                self.log(f"[pipeline] stopped after stage {idx + 1} failed")
                return False
        return True

    def execute_stage(self, idx: int) -> bool:
        """Run one stage on the calling thread, reporting state + progress."""
        runners = [
            self._run_stage_1_bg_removal,
            self._run_stage_2_colmap,
            self._run_stage_3_alignment,
            self._run_stage_4_reconstruction,
        ]
        self._on_stage_state(idx, "running", "")

        def on_progress(pct: int, text: str = ""):
            self._on_progress(idx, pct, text)

        try:
            ok = runners[idx](on_progress)
        except Exception as exc:
            self.log(f"[error] stage {idx + 1}: {exc}")
            ok = False

        self._on_stage_state(idx, "done" if ok else "failed", "")
        return ok

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._on_log(f"[{ts}] {msg}", "info")

    # ── callbacks: to the UI, and to the run record while one is open ─────────
    def _on_log(self, text: str, kind: str) -> None:
        self._ui_log(text, kind)
        rec = self._recorder
        if rec:
            rec.log(text, kind)

    def _on_stage_state(self, idx: int, state: str, status: str) -> None:
        self._ui_stage_state(idx, state, status)
        rec = self._recorder
        if rec:
            rec.stage_state(idx, state, stopped=self._stop_req)

    def _on_progress(self, idx: int, pct: int, text: str) -> None:
        self._ui_progress(idx, pct, text)
        rec = self._recorder
        if rec:
            rec.progress(idx, pct, text)

    def _recorded(self, target: Callable[[], object], mode: str) -> None:
        """Run `target`, recording it in the session folder when there is one.

        With neither an input nor an output folder set there is nothing to
        run on (stage 1 fails at once), so nothing is written to the default
        folder in the home directory."""
        rec = None
        if any(str(self.settings.get(k, "")).strip() for k in ("output_var", "input_var")):
            rec = RunRecorder(self.paths.session_dir(), self.settings, mode)
            rec.start()
            if rec.enabled:
                self._recorder = rec
            else:
                self.log(f"[pipeline] Warning: can't write the run log ({rec.error})")
        ok = False
        try:
            ok = bool(target())
        finally:
            if self._recorder:
                self._recorder = None
                rec.finish("done" if ok else "stopped" if self._stop_req else "failed")
                if not rec.enabled:
                    self.log(f"[pipeline] Warning: run log incomplete ({rec.error})")

    # ── threading / processes ─────────────────────────────────────────────────
    def _start(self, target: Callable[[], object], mode: str) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self._stop_req = False
            self.settings = copy.copy(self._live_settings)
            self.paths = SessionPaths(self.settings)
            self._thread = threading.Thread(target=self._recorded, args=(target, mode),
                                            daemon=True)
            self._thread.start()
        return True

    def wait(self, timeout: float | None = None) -> None:
        t = self._thread
        if t is not None:
            t.join(timeout)

    def shutdown(self, grace: float = STOP_GRACE_SECONDS) -> None:
        """Stop the running process group and wait for it (SIGKILL after
        `grace` seconds). For app exit: stop()'s SIGKILL timer is a daemon
        thread and would die with the interpreter."""
        self._stop_req = True
        proc = self._proc
        if proc is None:
            return
        self._terminate(proc)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        if _POSIX:
            try:
                os.killpg(proc.pid, signal.SIGKILL)   # pgid == pid (own session)
            except (ProcessLookupError, OSError):
                pass
        else:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """Stop `proc` and everything it started.

        Stage 2 runs ``bash run.sh``, whose COLMAP children would survive a
        kill of bash alone, so each subprocess gets its own process group
        (session) and the whole group is signalled: SIGTERM, then SIGKILL
        after a grace period.
        """
        if not _POSIX:
            try:
                proc.kill()
            except Exception:
                pass
            return
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            return
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return

        def _kill():
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass

        t = threading.Timer(STOP_GRACE_SECONDS, _kill)
        t.daemon = True
        t.start()

    def _run_proc(self, cmd: list, cwd: Path,
                  env: dict | None = None, on_line=None) -> int:
        """Run subprocess, stream all output to the log. on_line(line) per line."""
        self._on_log(f"$ {' '.join(str(c) for c in cmd)}", "header")
        if self._stop_req:
            return -1
        e = {**(env or os.environ.copy()), "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=e,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, errors="replace",
            start_new_session=_POSIX,
        )
        self._proc = proc
        if self._stop_req:          # stop() came in while the process was starting
            self._terminate(proc)
        try:
            for line in proc.stdout:
                if self._stop_req:
                    self._terminate(proc)
                    break
                stripped = line.rstrip()
                self._on_log(stripped, "output")
                if on_line and stripped:
                    on_line(stripped)
            proc.wait()
        finally:
            if proc.stdout:
                proc.stdout.close()
            self._proc = None
        return proc.returncode

    @staticmethod
    def _parsed(parser, on_progress):
        def on_line(line: str):
            r = parser.feed(line)
            if r:
                on_progress(*r)
        return on_line

    # ── Stage 1 — Background removal ──────────────────────────────────────────
    def _run_stage_1_bg_removal(self, on_progress) -> bool:
        inp = str(self.settings["input_var"]).strip()
        if not inp:
            self.log("[error] No input directory specified")
            return False

        out = self.paths.processed_dir()
        out.mkdir(parents=True, exist_ok=True)
        cmd = photos_cmd(self.settings, inp, out)

        self.log(f"[stage 1] Background removal: {inp} → {out}")
        rc = self._run_proc(cmd, cwd=SCRIPT_DIR,
                            on_line=self._parsed(PhotosParser(), on_progress))
        if rc == 0:
            on_progress(100, "Complete")
        return rc == 0

    # ── Stage 2 — COLMAP MVS ──────────────────────────────────────────────────
    def _run_stage_2_colmap(self, on_progress) -> bool:
        s1, s2 = self.paths.active_sides()
        processed = self.paths.processed_dir()
        env = colmap_env(self.settings)

        if s1 and s2:
            self.log(f"[stage 2] COLMAP side 1: {s1}")
            ok = self._colmap_one_side(s1, processed, env, role="primary",
                                       on_line=self._parsed(ColmapParser(0, 50), on_progress))
            if not ok or self._stop_req:
                return False

            self.log(f"[stage 2] COLMAP side 2: {s2}")
            return self._colmap_one_side(s2, processed, env, role="secondary",
                                         on_line=self._parsed(ColmapParser(50, 100), on_progress))

        if s1:
            self.log(f"[stage 2] COLMAP single side: {s1}")
            return self._colmap_one_side(s1, processed, env, role="primary",
                                         on_line=self._parsed(ColmapParser(0, 100), on_progress))

        self.log("[stage 2] COLMAP flat image set")
        return self._colmap_flat(processed, env,
                                 on_line=self._parsed(ColmapParser(0, 100), on_progress))

    def _colmap_one_side(self, side: str, processed_root: Path,
                         env: dict, on_line=None, role: str = "primary") -> bool:
        env = dict(env)
        user_file = str(self.settings["intr_file_var"]).strip()
        if user_file:
            env["FIPMESH_COLMAP_INTRINSICS_IN"] = user_file
            self.log(f"[stage 2] using fixed intrinsics from {user_file}")
        if role == "primary":
            # record what this side's reconstruction refined its cameras to
            env["FIPMESH_COLMAP_INTRINSICS_OUT"] = str(self.paths.intrinsics_path())
        elif self.settings["share_intr_var"] and not user_file:
            p = self.paths.intrinsics_path()
            if p.is_file():
                env["FIPMESH_COLMAP_INTRINSICS_IN"] = str(p)
                self.log(f"[stage 2] side {side}: reusing side 1's camera intrinsics (fixed) from {p.name}")
            else:
                self.log(f"[stage 2] side {side}: no saved intrinsics at {p} (run side 1 first); "
                         "estimating this side's own")
        img_dir = processed_root / side
        out_dir = self.paths.colmap_dir(side)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Each side is run through run.sh independently (as its own -i, no
        # -s), so only -m (primary mask) applies here, pointed at this side's
        # mask subfolder — not the shared mask root, which mirrors processed/'s
        # side1/side2 layout.
        side_mask_dir = self.paths.masks_dir() / side
        cmd = run_sh_cmd(img_dir, out_dir, side_mask_dir if side_mask_dir.is_dir() else None)
        rc = self._run_proc(cmd, cwd=SCRIPT_DIR, env=env, on_line=on_line)
        return rc == 0

    def _colmap_flat(self, processed_dir: Path, env: dict, on_line=None) -> bool:
        out_dir = self.paths.colmap_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        masks = self.paths.masks_dir()
        cmd = run_sh_cmd(processed_dir, out_dir, masks if masks.is_dir() else None)
        rc = self._run_proc(cmd, cwd=SCRIPT_DIR, env=env, on_line=on_line)
        return rc == 0

    # ── Stage 3 — FPFH Alignment ──────────────────────────────────────────────
    def _run_stage_3_alignment(self, on_progress) -> bool:
        s1, s2 = self.paths.active_sides()

        ply_a = str(self.settings["align_a_var"]).strip()
        ply_b = str(self.settings["align_b_var"]).strip()

        if not ply_a and not ply_b:
            if not s2:
                self.log("[stage 3] Single-side scan — alignment skipped")
                on_progress(100, "Skipped (single side)")
                return True
            ply_a = str(self.paths.colmap_dir(s1) / "fused.ply")
            ply_b = str(self.paths.colmap_dir(s2) / "fused.ply")

        for p in (ply_a, ply_b):
            if not Path(p).exists():
                self.log(f"[error] stage 3: PLY not found: {p}")
                return False

        merged_out = self.paths.merged_ply()
        merged_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = align_cmd(self.settings, ply_a, ply_b, merged_out, self.paths.icp_report_path())

        self.log(f"[stage 3] Aligning {Path(ply_a).name} + {Path(ply_b).name}")
        rc = self._run_proc(cmd, cwd=ALIGN_DIR,
                            on_line=self._parsed(AlignParser(), on_progress))
        if rc == 0:
            on_progress(100, "Alignment complete")
        return rc == 0

    # ── Stage 4 — Mesh Reconstruction ─────────────────────────────────────────
    def _run_stage_4_reconstruction(self, on_progress) -> bool:
        input_ply = self.paths.input_ply_for_recon()
        if not input_ply.exists():
            self.log(f"[error] stage 4: input PLY not found: {input_ply}")
            return False

        recon_dir = self.paths.recon_dir()
        recon_dir.mkdir(parents=True, exist_ok=True)
        out_obj       = recon_dir / "recon_mesh_recon.obj"
        clean_cloud   = recon_dir / "clean_cloud.ply"
        decimated_obj = recon_dir / "recon_mesh_recon_decimated.obj"

        side1_cam_centers = self.paths.side1_camera_centers_path()
        cmd = recon_cmd(
            self.settings,
            str(input_ply), str(out_obj), str(clean_cloud), str(decimated_obj),
            side1_camera_centers=str(side1_cam_centers) if side1_cam_centers.exists() else "")
        self.log(f"[stage 4] Reconstructing mesh from {input_ply.name}…")
        rc = self._run_proc(cmd, cwd=SCRIPT_DIR,
                            on_line=self._parsed(ReconParser(), on_progress))
        if rc != 0:
            return False

        session = self.paths.session_dir()

        # Copy auto-generated GLTF to the session root for easy access.
        # copyfile, not copy2: copying metadata fails on network drives.
        src_gltf  = recon_dir / "recon_mesh_recon.gltf"
        dest_gltf = session / "model.gltf"
        if src_gltf.exists():
            shutil.copyfile(src_gltf, dest_gltf)
            self.log(f"[stage 4] GLTF → {dest_gltf}")
        else:
            self.log(f"[stage 4] Warning: GLTF not found at {src_gltf}")

        # Copy the decimated web-viewer GLB to the session root too
        src_simplified_glb  = recon_dir / "recon_mesh_recon_simplified.glb"
        dest_simplified_glb = session / "model_simplified.glb"
        if src_simplified_glb.exists():
            shutil.copyfile(src_simplified_glb, dest_simplified_glb)
            self.log(f"[stage 4] Simplified GLB → {dest_simplified_glb}")

        # Write the website's info.txt metadata file, if a name was given
        name = str(self.settings["meta_name_var"]).strip()
        if name:
            from src.artifact_info import write_info_txt
            try:
                info_path = write_info_txt(
                    session,
                    name=name,
                    type_=str(self.settings["meta_type_var"]).strip() or "tablet",
                    description=str(self.settings["meta_desc_text"]).strip(),
                    link=str(self.settings["meta_link_var"]).strip(),
                    link_label=str(self.settings["meta_link_label_var"]).strip(),
                )
                self.log(f"[stage 4] info.txt → {info_path}")
            except ValueError as e:
                self.log(f"[stage 4] Warning: skipped info.txt ({e})")
        else:
            self.log("[stage 4] No artifact name set — skipping info.txt")

        on_progress(100, "Reconstruction complete")
        return True
