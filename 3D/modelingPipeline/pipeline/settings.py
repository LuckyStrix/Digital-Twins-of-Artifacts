"""Registry of every app setting, plus load/save of app_defaults.json.

One ``Setting`` per value the user can edit. The UI builds its forms from
``REGISTRY`` (in order, grouped by ``tab`` then ``section``); the runner reads
plain values out of a ``Settings`` dict keyed by ``Setting.key``.

Keys are the attribute names the Tk GUI used (``ConfigPanel._PERSISTED_VARS``)
so existing app_defaults.json files load unchanged. Value types also match
what Tk stored: some numeric-looking fields (GPU index, image scale, SIFT
thresholds, ...) were StringVars and stay strings, since the runner passes
them through verbatim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import DEFAULTS_PATH
from .options import (
    ALIGN_METHODS, ARTIFACT_TYPES, BACKGROUNDS, CAMERA_MODEL_DEFAULT, CAMERA_MODELS,
    COLMAP_QUALITIES, REMBG_MODELS, SEAM_MODES, SEC_ALIGN_MODES,
)

# Setting.type values
INT, FLOAT, STR, BOOL, CHOICE, PATH, TEXT = "int", "float", "str", "bool", "choice", "path", "text"


@dataclass(frozen=True)
class Setting:
    key: str
    type: str                      # one of INT/FLOAT/STR/BOOL/CHOICE/PATH/TEXT
    default: Any
    tab: str
    section: str
    label: str
    hint: str = ""                 # short grey text next to the field
    help: str = ""                 # long grey paragraph under the field
    min: float | None = None
    max: float | None = None
    step: float | None = None      # spinbox increment in the Tk GUI
    choices: tuple = ()
    editable: bool = False         # CHOICE only: free text allowed, choices are suggestions
    persisted: bool = True         # saved by "Save current settings as default"
    path_kind: str = ""            # PATH only: "dir" or "file"
    file_glob: str = ""            # PATH/file only: e.g. "*.json"
    validate: str = ""             # STR only: "int" / "float" / "float_list" hint for the UI
    widget: str = ""               # UI hint, e.g. "slider"


def _S(key, type_, default, tab, section, label, **kw) -> Setting:
    return Setting(key, type_, default, tab, section, label, **kw)


_IN, _CM, _RC, _AL = "Inputs", "COLMAP", "Reconstruct", "Alignment"

REGISTRY: list[Setting] = [
    # ── Inputs ────────────────────────────────────────────────────────────────
    _S("input_var", PATH, "", _IN, "Folders", "Input dir", path_kind="dir", persisted=False),
    _S("output_var", PATH, "", _IN, "Folders", "Output dir", path_kind="dir", persisted=False),

    _S("meta_name_var", STR, "", _IN, "Artifact info (for info.txt, optional)", "Name",
       persisted=False),
    _S("meta_type_var", CHOICE, "tablet", _IN, "Artifact info (for info.txt, optional)", "Type",
       choices=tuple(ARTIFACT_TYPES), editable=True, persisted=False),
    _S("meta_desc_text", TEXT, "", _IN, "Artifact info (for info.txt, optional)", "Description",
       persisted=False),
    _S("meta_link_var", STR, "", _IN, "Artifact info (for info.txt, optional)", "Link",
       persisted=False),
    _S("meta_link_label_var", STR, "", _IN, "Artifact info (for info.txt, optional)", "Link Label",
       persisted=False, help="Leave Name blank to skip writing info.txt."),

    _S("side1_var", STR, "side1", _IN, "Side folders (blank = flat/single-side)", "Primary"),
    _S("side2_var", STR, "side2", _IN, "Side folders (blank = flat/single-side)", "Secondary",
       hint="(blank = skip alignment)"),

    _S("bg_var", CHOICE, "white", _IN, "Background removal", "Background",
       choices=tuple(BACKGROUNDS)),
    _S("hard_mask_var", BOOL, True, _IN, "Background removal", "Hard mask cutoff",
       help=("Thresholds rembg's soft alpha matte to a clean binary mask instead "
             "of a blended gradient at the silhouette edge. COLMAP never sees "
             "alpha, only RGB, so a soft edge leaves real background colour "
             "blended into the tablet's boundary pixels and can reconstruct as "
             "speckled noise along the mesh edges. Recommended on.")),
    _S("black_thresh_var", INT, 0, _IN, "Background removal", "Black threshold",
       min=0, max=255, hint="(0=off)"),
    _S("white_thresh_var", INT, 0, _IN, "Background removal", "White threshold",
       min=0, max=255, hint="(0=off)"),
    _S("value_thresh_var", INT, 0, _IN, "Background removal", "Value threshold",
       min=0, max=255, hint="(0=off)"),
    _S("chroma_thresh_var", INT, 0, _IN, "Background removal", "Chroma threshold",
       min=0, max=255, hint="(0=off, removes low-colour bleed — try 15-30)"),
    _S("edge_band_var", INT, 0, _IN, "Background removal", "Edge band (px)",
       min=0, max=100, hint="(0=whole mask)"),
    _S("erode_px_var", INT, 0, _IN, "Background removal", "Erode mask (px)",
       min=0, max=100, hint="(0=off, shrinks mask inward)"),
    _S("grow_chroma_var", INT, 0, _IN, "Background removal", "Grow by colour",
       min=0, max=255, hint="(0=off, chroma threshold — try 40-60)",
       help=("Grows the mask onto any region touching the tablet that has real "
             "colour, regardless of hue — fixes rembg dropping an attached, "
             "differently-coloured piece (e.g. a mounting board) as background. "
             "Only works against a neutral (black/white/grey) backdrop.")),
    _S("grow_hull_fill_var", BOOL, False, _IN, "Background removal",
       "Convex-hull fill grown region",
       help=("Bridges low-colour detail sitting on the grown region (e.g. a "
             "printed label) even where it touches the region's own edge. "
             "Assumes the attached piece is basically convex (a standard "
             "rectangular/oval mounting board) — off by default since most "
             "tablets aren't board-backed.")),
    _S("passes_var", INT, 1, _IN, "Background removal", "Passes",
       min=1, max=5, hint="(1=off, slower per extra pass)"),
    _S("seg_scale_var", INT, 100, _IN, "Background removal", "Seg. quality",
       min=10, max=100, hint="%", widget="slider",
       help=("Downscales the image fed to the bg-removal model to save memory/time; "
             "saved output stays full resolution either way. Note: rembg resizes "
             "every model's input to a fixed working size before running on the GPU, "
             "so this mainly saves CPU/RAM time, not GPU VRAM.")),
    _S("model_var", CHOICE, "birefnet-general", _IN, "Background removal", "rembg model",
       choices=tuple(REMBG_MODELS)),

    # ── COLMAP ────────────────────────────────────────────────────────────────
    _S("quality_var", CHOICE, "high", _CM, "COLMAP", "Quality", choices=tuple(COLMAP_QUALITIES)),
    _S("use_gpu_var", BOOL, True, _CM, "COLMAP", "Use GPU"),
    _S("gpu_index_var", STR, "-1", _CM, "COLMAP", "GPU index (−1=auto)", validate="int"),
    _S("img_scale_var", STR, "1", _CM, "COLMAP", "Image scale (0–1)", validate="float"),
    _S("img_stride_var", INT, 3, _CM, "COLMAP", "Image stride", min=1, max=10),

    _S("sift_features_var", INT, 16000, _CM, "SIFT", "Max features",
       min=1000, max=65536, step=1000),
    _S("sift_peak_var", STR, "0.0045", _CM, "SIFT", "Peak threshold", validate="float"),
    _S("sift_edge_var", STR, "12", _CM, "SIFT", "Edge threshold", validate="float"),
    _S("sift_dsp_var", BOOL, True, _CM, "SIFT", "Domain size pooling"),
    _S("sift_affine_var", BOOL, False, _CM, "SIFT", "Estimate affine shape"),

    _S("match_guided_var", BOOL, True, _CM, "Matching", "Guided matching"),
    _S("match_max_var", INT, 65536, _CM, "Matching", "Max matches",
       min=1000, max=131072, step=4096),

    *[_S(key, INT, 0, _CM, "Threads / Cache  (0 = auto)", label, min=0, max=256)
      for label, key in [
          ("Extract threads",   "extract_threads_var"),
          ("Match threads",     "match_threads_var"),
          ("Mapper threads",    "mapper_threads_var"),
          ("Fusion threads",    "fusion_threads_var"),
          ("Patch cache (GB)",  "patch_cache_var"),
          ("Fusion cache (GB)", "fusion_cache_var"),
      ]],

    _S("share_intr_var", BOOL, True, _CM, "Camera intrinsics", "Share between sides",
       help=("Both sides come from the same camera and lens, so side 1's refined "
             "calibration (focal length, lens distortion) is saved and side 2 reuses it "
             "FIXED instead of re-estimating its own. Separately estimated calibrations "
             "differ slightly from side to side (focal length and distortion trade off "
             "against each other), which shows up as the two surfaces not quite agreeing. "
             "With several cameras, each camera folder (cam1, cam2, ...) is matched to "
             "the same folder on the other side. Untick if the sides used different lenses.")),
    _S("camera_model_var", CHOICE, CAMERA_MODEL_DEFAULT, _CM, "Camera intrinsics", "Camera model",
       choices=tuple(CAMERA_MODELS)),
    _S("intr_file_var", PATH, "", _CM, "Camera intrinsics", "Intrinsics file",
       path_kind="file", file_glob="*.json", persisted=False,
       help=("Optional: a camera_intrinsics.json (e.g. from an earlier session or a "
             "separate calibration) to use, fixed, for BOTH sides. Overrides sharing.")),

    _S("sec_rotate_deg_var", STR, "180", _CM, "Secondary camera", "Rotate degrees", validate="float"),
    _S("sec_rotate_axis_var", STR, "primary_frame_x", _CM, "Secondary camera", "Rotate axis"),
    _S("sec_extra_x_var", STR, "0", _CM, "Secondary camera", "Extra rotate X", validate="float"),
    _S("sec_extra_y_var", STR, "0", _CM, "Secondary camera", "Extra rotate Y", validate="float"),
    _S("sec_extra_z_var", STR, "0", _CM, "Secondary camera", "Extra rotate Z", validate="float"),
    _S("sec_translate_x_var", STR, "0", _CM, "Secondary camera", "Translate X", validate="float"),
    _S("sec_translate_y_var", STR, "0", _CM, "Secondary camera", "Translate Y", validate="float"),
    _S("sec_translate_z_var", STR, "0", _CM, "Secondary camera", "Translate Z", validate="float"),
    _S("sec_align_mode_var", CHOICE, "auto", _CM, "Secondary camera", "Align mode",
       choices=tuple(SEC_ALIGN_MODES)),

    # ── Reconstruct ───────────────────────────────────────────────────────────
    _S("r_max_input_pts", INT, 0, _RC, "Point cloud filtering", "Max input points",
       min=0, max=10000000),
    _S("r_outlier_nn", INT, 32, _RC, "Point cloud filtering", "Outlier neighbors", min=0, max=256),
    _S("r_outlier_std", FLOAT, 0.0, _RC, "Point cloud filtering", "Outlier std ratio",
       min=0, max=20.0, step=0.5),
    _S("r_radius_nn", INT, 0, _RC, "Point cloud filtering", "Radius outlier NB pts", min=0, max=256),
    _S("r_radius_factor", FLOAT, 2.2, _RC, "Point cloud filtering", "Radius outlier factor",
       min=0, max=20.0, step=0.1),

    _S("r_dbscan_max_pts", INT, 0, _RC, "DBSCAN clustering", "DBSCAN max points",
       min=0, max=10000000),
    _S("r_dbscan_min_pts", INT, 0, _RC, "DBSCAN clustering", "DBSCAN min points", min=0, max=1000),
    _S("r_dbscan_eps", FLOAT, 2.2, _RC, "DBSCAN clustering", "DBSCAN eps factor",
       min=0, max=20.0, step=0.1),
    _S("r_dbscan_keep", INT, 1, _RC, "DBSCAN clustering", "DBSCAN keep largest", min=0, max=20),
    _S("r_dbscan_ratio", FLOAT, 0.02, _RC, "DBSCAN clustering", "DBSCAN min cluster ratio",
       min=0, max=1.0, step=0.005),

    _S("r_normal_max_nn", INT, 96, _RC, "Normals", "Normal max NN", min=0, max=512),
    _S("r_normal_orient_k", INT, 64, _RC, "Normals", "Normal orient K", min=0, max=512),

    _S("r_poisson_depth", INT, 10, _RC, "Poisson reconstruction", "Poisson depth", min=5, max=14),
    _S("r_poisson_linear", BOOL, True, _RC, "Poisson reconstruction", "Poisson linear fit"),
    _S("r_density_trim", FLOAT, 0.02, _RC, "Poisson reconstruction", "Density trim quantile",
       min=0, max=0.5, step=0.001),
    _S("r_poisson_crop_scale", FLOAT, 1.05, _RC, "Poisson reconstruction", "Poisson crop scale",
       min=0, max=2.0, step=0.01),

    _S("r_hole_reduction", BOOL, False, _RC, "Hole reduction",
       "Reduce holes near rotation axis / smooth surfaces",
       help=("Keeps a bit more low-density Poisson surface, widens the crop margin, "
             "and uses more neighbors for normal estimation, so sparsely sampled "
             "areas (rotation axis poles, textureless surfaces) are less likely to "
             "be trimmed away as holes. Overrides density trim quantile, Poisson "
             "crop scale, and normal max NN above. Pair with hole filling below for "
             "the remaining gaps.")),

    _S("r_fill_holes_ratio", FLOAT, 0.3, _RC, "Hole filling", "Fill hole size ratio",
       min=0, max=1.0, step=0.01),
    _S("r_fill_holes_passes", INT, 4, _RC, "Hole filling", "Fill hole passes", min=1, max=10),
    _S("r_fill_holes", BOOL, False, _RC, "Hole filling",
       "Fill remaining holes (boundary triangulation)"),
    _S("r_fill_smooth", BOOL, False, _RC, "Hole filling",
       "Smooth the filled patches (experimental)",
       help=("Hole patches are always re-wound to match the surrounding surface "
             "(Open3D's filler leaves most of them inverted, which shows as dark or "
             "oddly shaded flat panels). This option additionally subdivides and "
             "fairs each well-formed patch into a smooth membrane. Slower, and can "
             "leave a few dark slivers on thin patches.\n\n"
             "After cleanup, triangulates any leftover boundary loops to bridge "
             "gaps directly, rather than relying on loosening the density trim "
             "threshold. Hole size ratio caps the max hole radius filled, as a "
             "fraction of the cleaned cloud's bounding-box diagonal — raise it to "
             "close bigger gaps, lower it to avoid bridging genuinely unscanned "
             "regions. The fill pass doesn't always converge in one call — "
             "closing one loop can make a neighboring loop fillable only on the "
             "next pass — so it repeats up to \"Fill hole passes\" times, "
             "stopping early once a pass closes nothing more.")),

    _S("r_comp_min_ratio", FLOAT, 0.01, _RC, "Mesh cleanup", "Component min ratio",
       min=0, max=1.0, step=0.005),
    _S("r_comp_min_tris", INT, 1000, _RC, "Mesh cleanup", "Component min triangles",
       min=0, max=100000, step=100),
    _S("r_comp_max_count", INT, 1, _RC, "Mesh cleanup", "Component max count", min=0, max=50),
    _S("r_smooth_iters", INT, 1, _RC, "Mesh cleanup", "Smooth iterations", min=0, max=20),
    _S("r_decimate_tris", INT, 300000, _RC, "Mesh cleanup", "Decimate target tris",
       min=0, max=5000000, step=50000),

    _S("r_simplified_target_verts", INT, 60000, _RC, "Simplified export (for website)",
       "Simplified target verts", min=0, max=2000000, step=5000,
       help=("Always also exports \"<name>_simplified.glb\": a single small "
             "binary glTF file, decimated to roughly this many vertices, with "
             "vertex colors baked in (no separate texture/MTL files) — easy to "
             "drop into a web viewer. Set to 0 to disable.")),

    _S("r_normalize_pose", BOOL, False, _RC, "Pose normalization",
       "Center at origin & flatten (largest cross-section on XY plane)",
       help=("Recenters the mesh's centroid at (0,0,0) and rotates it (via PCA "
             "on the mesh vertices) so its two widest axes span X/Y and its "
             "thinnest axis (tablet thickness) lies along Z — like a coin lying "
             "flat on a table. Applied once to the cleaned cloud and mesh right "
             "after cleanup, so every exported variant (OBJ/PLY/glTF/simplified "
             "glb) shares the same pose.")),

    # ── Alignment ─────────────────────────────────────────────────────────────
    _S("align_method_var", CHOICE, "fpfh", _AL, "FPFH alignment settings", "Method",
       choices=tuple(ALIGN_METHODS)),
    _S("align_voxel_var", STR, "0", _AL, "FPFH alignment settings", "Voxel (0=auto)",
       validate="float"),
    _S("align_samples_var", INT, 60000, _AL, "FPFH alignment settings", "Sample points",
       min=5000, max=500000, step=5000),

    _S("align_refine_var", BOOL, False, _AL, "ICP refinement (coarse-to-fine)",
       "Refine with multi-stage ICP after alignment",
       help=("Runs ICP on dense clouds at a shrinking correspondence distance "
             "so the two halves are pulled onto each other at sub-voxel scale. "
             "Off = the original single-pass result.")),
    _S("align_thresholds_var", STR, "3,1,0.5,0.25", _AL, "ICP refinement (coarse-to-fine)",
       "Stage dists", validate="float_list",
       hint="Correspondence distance per stage, in voxels. Add a smaller value to go tighter."),
    _S("align_iters_var", INT, 100, _AL, "ICP refinement (coarse-to-fine)", "Iters / stage",
       min=10, max=2000, step=10,
       hint="Max ICP iterations per stage; a stage stops early once converged."),
    _S("align_points_var", INT, 300000, _AL, "ICP refinement (coarse-to-fine)", "Dense points",
       min=50000, max=3000000, step=50000,
       hint="Points sampled per side for refinement (separate from Sample points)."),
    _S("align_tol_var", STR, "1e-7", _AL, "ICP refinement (coarse-to-fine)", "Tolerance",
       validate="float", hint="Relative fitness/RMSE change that counts as converged."),
    _S("align_band_var", STR, "0", _AL, "ICP refinement (coarse-to-fine)", "Seam band (0=off)",
       validate="float",
       hint=("If >0, refine only points within band x stage-distance of the "
             "other side, i.e. the overlap near the seam.")),
    _S("align_robust_var", STR, "0", _AL, "ICP refinement (coarse-to-fine)", "Robust σ (0=off)",
       validate="float",
       hint=("If >0, Tukey robust loss (σ in voxels) so outliers and the "
             "wrong-sheet points near the seam are down-weighted.")),

    _S("align_seam_var", BOOL, True, _AL, "Seam resolution (drop low-confidence points)",
       "Where the two sides disagree, keep the more confident one",
       help=("Confidence per point = COLMAP view count x how head-on the views were "
             "(from each side's .vis file + camera poses). Where both sides cover a "
             "surface but sit apart as separate sheets, the less confident side's "
             "points are dropped. Points are never moved. Writes "
             "merged_fpfh_seam_audit.ply (dropped points shown black/orange) and a "
             "seam report next to the merged cloud.")),
    _S("align_seam_mode_var", CHOICE, "point", _AL, "Seam resolution (drop low-confidence points)",
       "Mode", choices=tuple(SEAM_MODES),
       hint="point = stronger; patch = compare whole sheets (gentler)"),
    _S("align_seam_tau_var", STR, "0.3", _AL, "Seam resolution (drop low-confidence points)",
       "Conflict gap (vox)", validate="float",
       hint="Sheets closer than this count as agreeing. Lower = more drops."),
    _S("align_seam_minconf_var", STR, "0.35", _AL, "Seam resolution (drop low-confidence points)",
       "Confidence floor", validate="float",
       hint=("Also drop points below this confidence (median is about 0.5) on the less "
             "confident side wherever the other side covers the surface. 0 = off.")),
    _S("align_seam_window_var", STR, "10", _AL, "Seam resolution (drop low-confidence points)",
       "Window (vox)", validate="float",
       hint="Neighbourhood used to compare the sides' confidence."),
    _S("align_seam_minc_var", INT, 20, _AL, "Seam resolution (drop low-confidence points)",
       "Min. other pts",
       hint=("The other side needs this many points nearby, so nothing is dropped "
             "where it has no coverage.")),
    _S("align_seam_passes_var", INT, 3, _AL, "Seam resolution (drop low-confidence points)",
       "Passes", hint="Re-evaluate after dropping, up to this many times."),

    _S("align_a_var", PATH, "", _AL, "PLY overrides (blank = auto from pipeline)", "PLY A (fixed)",
       path_kind="file", file_glob="*.ply", persisted=False),
    _S("align_b_var", PATH, "", _AL, "PLY overrides (blank = auto from pipeline)", "PLY B (moving)",
       path_kind="file", file_glob="*.ply", persisted=False,
       help="Useful for re-running alignment with different inputs."),
]

BY_KEY: dict[str, Setting] = {s.key: s for s in REGISTRY}
assert len(BY_KEY) == len(REGISTRY), "duplicate setting key"

TABS = ["Inputs", "COLMAP", "Reconstruct", "Alignment"]

# Save order of app_defaults.json: exactly the Tk GUI's _PERSISTED_VARS.
PERSISTED_KEYS = [
    "side1_var", "side2_var",
    "bg_var", "hard_mask_var", "black_thresh_var", "white_thresh_var", "value_thresh_var",
    "chroma_thresh_var",
    "edge_band_var", "erode_px_var", "grow_chroma_var", "grow_hull_fill_var",
    "passes_var", "seg_scale_var", "model_var",
    "quality_var", "use_gpu_var", "gpu_index_var", "img_scale_var",
    "img_stride_var",
    "sift_features_var", "sift_peak_var", "sift_edge_var",
    "sift_dsp_var", "sift_affine_var",
    "match_guided_var", "match_max_var",
    "extract_threads_var", "match_threads_var", "mapper_threads_var",
    "fusion_threads_var", "patch_cache_var", "fusion_cache_var",
    "sec_rotate_deg_var", "sec_rotate_axis_var",
    "sec_extra_x_var", "sec_extra_y_var", "sec_extra_z_var",
    "sec_translate_x_var", "sec_translate_y_var", "sec_translate_z_var",
    "sec_align_mode_var",
    "share_intr_var", "camera_model_var",
    "r_max_input_pts", "r_outlier_nn", "r_outlier_std",
    "r_radius_nn", "r_radius_factor",
    "r_dbscan_max_pts", "r_dbscan_min_pts", "r_dbscan_eps",
    "r_dbscan_keep", "r_dbscan_ratio",
    "r_normal_max_nn", "r_normal_orient_k",
    "r_poisson_depth", "r_poisson_linear", "r_density_trim",
    "r_poisson_crop_scale", "r_hole_reduction",
    "r_fill_holes", "r_fill_holes_ratio", "r_fill_holes_passes", "r_fill_smooth",
    "r_comp_min_ratio", "r_comp_min_tris", "r_comp_max_count",
    "r_smooth_iters", "r_decimate_tris", "r_simplified_target_verts",
    "r_normalize_pose",
    "align_method_var", "align_voxel_var", "align_samples_var",
    "align_refine_var", "align_thresholds_var", "align_iters_var",
    "align_points_var", "align_tol_var", "align_band_var", "align_robust_var",
    "align_seam_var", "align_seam_mode_var", "align_seam_tau_var",
    "align_seam_window_var", "align_seam_minc_var", "align_seam_passes_var",
    "align_seam_minconf_var",
]
assert set(PERSISTED_KEYS) == {s.key for s in REGISTRY if s.persisted}

# "Reduce holes" checkbox presets, applied to the three linked fields whenever
# it is toggled. A middle ground between the plain defaults and the earlier
# aggressive preset: loosening density trim too far let noisy low-confidence
# surface through in specular/bright spots (which also tend to be low
# point-density from poor feature matching), hurting geometry there. The
# hole-filling pass now picks up the remaining true gaps, so this preset can
# stay more conservative.
HOLE_REDUCTION_OFF = {"r_density_trim": 0.02, "r_poisson_crop_scale": 1.05, "r_normal_max_nn": 96}
HOLE_REDUCTION_ON = {"r_density_trim": 0.003, "r_poisson_crop_scale": 1.08, "r_normal_max_nn": 80}


def hole_reduction_preset(on: bool) -> dict[str, Any]:
    return dict(HOLE_REDUCTION_ON if on else HOLE_REDUCTION_OFF)


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def coerce(setting: Setting, value: Any) -> Any:
    """Convert ``value`` to ``setting``'s type; raise ValueError if it can't be."""
    t = setting.type
    if t == BOOL:
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            return value.strip().lower() in _TRUE
        raise ValueError(f"{setting.key}: not a boolean: {value!r}")
    if t == INT:
        if isinstance(value, bool):
            raise ValueError(f"{setting.key}: not an integer: {value!r}")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"{setting.key}: not an integer: {value!r}")
            return int(value)
        if isinstance(value, (int, str)):
            return int(str(value).strip())
        raise ValueError(f"{setting.key}: not an integer: {value!r}")
    if t == FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError(f"{setting.key}: not a number: {value!r}")
        return float(value)
    if t == CHOICE:
        if not isinstance(value, str):
            raise ValueError(f"{setting.key}: not a string: {value!r}")
        if not setting.editable and value not in setting.choices:
            raise ValueError(f"{setting.key}: {value!r} not one of {setting.choices}")
        return value
    # STR / PATH / TEXT
    if isinstance(value, (dict, list)) or value is None:
        raise ValueError(f"{setting.key}: not a string: {value!r}")
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


class Settings(dict):
    """Current value of every setting, keyed by ``Setting.key``."""

    @classmethod
    def defaults(cls) -> "Settings":
        return cls({s.key: s.default for s in REGISTRY})

    @classmethod
    def load(cls, path: Path = DEFAULTS_PATH) -> "Settings":
        """Code defaults overlaid with the persisted values saved at ``path``."""
        s = cls.defaults()
        s.update_from_file(path)
        return s

    def update_from_file(self, path: Path = DEFAULTS_PATH) -> None:
        """Apply saved defaults; unknown keys and bad values are skipped."""
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for key, value in data.items():
            setting = BY_KEY.get(key)
            if setting is None or not setting.persisted:
                continue
            try:
                self[key] = coerce(setting, value)
            except (ValueError, TypeError):
                pass

    def persisted(self) -> dict[str, Any]:
        return {k: self[k] for k in PERSISTED_KEYS}

    def save(self, path: Path = DEFAULTS_PATH) -> None:
        """Write the persisted settings. Raises OSError on failure."""
        Path(path).write_text(json.dumps(self.persisted(), indent=2))

    def text(self, key: str) -> str:
        """String value, stripped (like ``var.get().strip()`` in the Tk GUI)."""
        return str(self[key]).strip()
