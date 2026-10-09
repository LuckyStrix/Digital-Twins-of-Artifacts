"""Golden tests: the exact commands / env vars each stage builds.

Expected values were taken from the Tk GUI (app.py before the Textual port)
for the same settings, and cross-checked against it with randomised settings.
"""

import os
from pathlib import Path

import pytest

from conftest import SCRIPT_DIR, RecordingRunner
from pipeline.platform import venv_python

PY = venv_python()
RUN_SH = str(SCRIPT_DIR / "run.sh")


def rel(p: Path) -> str:
    return os.path.relpath(p, SCRIPT_DIR)


@pytest.fixture
def s(settings, two_side_scan, tmp_path):
    settings["input_var"] = str(two_side_scan)
    settings["output_var"] = str(tmp_path / "out")
    return settings


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def test_stage1_defaults(s, tmp_path):
    r = RecordingRunner(s)
    assert r.execute_stage(0)
    (cmd, cwd, env), = r.calls
    assert cmd == [
        PY, str(SCRIPT_DIR / "process_photos.py"),
        "--input", s["input_var"],
        "--output", str(tmp_path / "out" / "processed"),
        "--model", "birefnet-general",
        "--background", "white",
        "--stride", "3",
    ]
    assert cwd == str(SCRIPT_DIR) and env is None
    assert (tmp_path / "out" / "processed").is_dir()
    assert r.info() == [f"[stage 1] Background removal: {s['input_var']} → {tmp_path / 'out' / 'processed'}"]
    assert r.states == [(0, "running"), (0, "done")]
    assert r.progress[-1] == (0, 100, "Complete")


def test_stage1_all_options(s, tmp_path):
    s.update({
        "bg_var": "black", "model_var": "u2net", "hard_mask_var": False,
        "black_thresh_var": 10, "white_thresh_var": 20, "value_thresh_var": 30,
        "chroma_thresh_var": 40, "edge_band_var": 5, "erode_px_var": 6,
        "grow_chroma_var": 50, "grow_hull_fill_var": True, "passes_var": 3,
        "img_stride_var": 1, "seg_scale_var": 50,
    })
    r = RecordingRunner(s)
    r.execute_stage(0)
    assert r.calls[0][0] == [
        PY, str(SCRIPT_DIR / "process_photos.py"),
        "--input", s["input_var"],
        "--output", str(tmp_path / "out" / "processed"),
        "--model", "u2net",
        "--background", "black",
        "--no-hard-mask",
        "--black-threshold", "10",
        "--white-threshold", "20",
        "--value-threshold", "30",
        "--chroma-threshold", "40",
        "--edge-band", "5",
        "--erode-px", "6",
        "--grow-chroma", "50",
        "--grow-hull-fill",
        "--passes", "3",
        "--seg-scale-pct", "50",
    ]


def test_stage1_hull_fill_needs_grow_chroma(s):
    s.update({"grow_chroma_var": 0, "grow_hull_fill_var": True, "img_stride_var": 1})
    r = RecordingRunner(s)
    r.execute_stage(0)
    assert "--grow-hull-fill" not in r.calls[0][0]
    assert "--stride" not in r.calls[0][0]


def test_stage1_no_input(settings):
    r = RecordingRunner(settings)
    assert not r.execute_stage(0)
    assert r.calls == []
    assert r.info() == ["[error] No input directory specified"]
    assert r.states == [(0, "running"), (0, "failed")]


def test_stage1_parser_progress(s):
    r = RecordingRunner(s, output={"process_photos": ["Processing 2 images", "[1/2] a", "[2/2] b"]})
    r.execute_stage(0)
    assert r.progress == [(0, 2, "Processing 2 images…"), (0, 50, "Image 1/2"),
                          (0, 95, "Image 2/2"), (0, 100, "Complete")]


def test_stage1_failure(s):
    r = RecordingRunner(s, rc=1)
    assert not r.execute_stage(0)
    assert r.states[-1] == (0, "failed")


# ── Stage 2 ───────────────────────────────────────────────────────────────────

GOLDEN_COLMAP_ENV_DEFAULTS = {
    "FIPMESH_COLMAP_QUALITY": "high",
    "FIPMESH_COLMAP_USE_GPU": "1",
    "FIPMESH_COLMAP_GPU_INDEX": "-1",
    "FIPMESH_COLMAP_IMAGE_SCALE": "1",
    "FIPMESH_COLMAP_IMAGE_STRIDE": "1",
    "FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES": "16000",
    "FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD": "0.0045",
    "FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD": "12",
    "FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING": "1",
    "FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE": "0",
    "FIPMESH_COLMAP_MATCH_GUIDED": "1",
    "FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES": "65536",
    "FIPMESH_SKIP_RECON": "1",
    "FIPMESH_COLMAP_SECONDARY_ROTATE_DEG": "180",
    "FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS": "primary_frame_x",
    "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X": "0",
    "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y": "0",
    "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z": "0",
    "FIPMESH_COLMAP_SECONDARY_TRANSLATE_X": "0",
    "FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y": "0",
    "FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z": "0",
    "FIPMESH_COLMAP_SECONDARY_ALIGN_MODE": "auto",
}


def fipmesh(env: dict) -> dict:
    # FIPMESH_PYTHON comes from the Docker image, not the runner (conftest
    # keeps it so PY above stays right).
    return {k: v for k, v in env.items() if k.startswith("FIPMESH_") and k != "FIPMESH_PYTHON"}


def test_stage2_two_sides_defaults(s, tmp_path):
    out = tmp_path / "out"
    (out / "processed_masks" / "side1").mkdir(parents=True)   # side2 has no masks
    r = RecordingRunner(s, output={
        "colmap_side1": ["[step] colmap mapper", "dense cloud output: x"],
        "colmap_side2": ["[step] colmap mapper", "dense cloud output: x"],
    })
    assert r.execute_stage(1)
    (c1, cwd1, e1), (c2, cwd2, e2) = r.calls
    assert c1 == ["bash", RUN_SH, "-i", rel(out / "processed" / "side1"),
                  "-o", rel(out / "colmap_side1"), "-v",
                  "-m", rel(out / "processed_masks" / "side1")]
    assert c2 == ["bash", RUN_SH, "-i", rel(out / "processed" / "side2"),
                  "-o", rel(out / "colmap_side2"), "-v"]
    assert cwd1 == cwd2 == str(SCRIPT_DIR)
    intr = str(out / "camera_intrinsics.json")
    assert fipmesh(e1) == {**GOLDEN_COLMAP_ENV_DEFAULTS, "FIPMESH_COLMAP_INTRINSICS_OUT": intr}
    # side 1 hasn't written intrinsics (recorded run), so side 2 estimates its own
    assert fipmesh(e2) == GOLDEN_COLMAP_ENV_DEFAULTS
    assert r.info() == [
        "[stage 2] COLMAP side 1: side1",
        "[stage 2] COLMAP side 2: side2",
        f"[stage 2] side side2: no saved intrinsics at {intr} (run side 1 first); "
        "estimating this side's own",
    ]
    # progress split 0-50 / 50-100 between the sides
    assert [p for _, p, _ in r.progress] == [19, 50, 69, 100]
    assert (out / "colmap_side1").is_dir() and (out / "colmap_side2").is_dir()


def test_stage2_env_non_default(s, tmp_path):
    s.update({
        "quality_var": "low", "use_gpu_var": False, "gpu_index_var": "1",
        "img_scale_var": "0.5", "img_stride_var": 4, "sift_features_var": 8000,
        "sift_peak_var": "0.006", "sift_edge_var": "10", "sift_dsp_var": False,
        "sift_affine_var": True, "match_guided_var": False, "match_max_var": 32768,
        "extract_threads_var": 4, "match_threads_var": 0, "mapper_threads_var": 2,
        "fusion_threads_var": 0, "patch_cache_var": 16, "fusion_cache_var": 8,
        "sec_rotate_deg_var": "90", "sec_rotate_axis_var": "y", "sec_extra_x_var": "1",
        "sec_extra_y_var": "2", "sec_extra_z_var": "3", "sec_translate_x_var": "4",
        "sec_translate_y_var": "5", "sec_translate_z_var": "6",
        "sec_align_mode_var": "manual", "camera_model_var": "OPENCV",
        "side2_var": "",
    })
    r = RecordingRunner(s)
    r.execute_stage(1)
    (_, _, env), = r.calls
    assert fipmesh(env) == {
        "FIPMESH_COLMAP_QUALITY": "low",
        "FIPMESH_COLMAP_USE_GPU": "0",
        "FIPMESH_COLMAP_GPU_INDEX": "1",
        "FIPMESH_COLMAP_IMAGE_SCALE": "0.5",
        "FIPMESH_COLMAP_IMAGE_STRIDE": "1",
        "FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES": "8000",
        "FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD": "0.006",
        "FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD": "10",
        "FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING": "0",
        "FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE": "1",
        "FIPMESH_COLMAP_MATCH_GUIDED": "0",
        "FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES": "32768",
        "FIPMESH_SKIP_RECON": "1",
        "FIPMESH_COLMAP_EXTRACT_THREADS": "4",
        "FIPMESH_COLMAP_MAPPER_THREADS": "2",
        "FIPMESH_COLMAP_PATCH_CACHE_SIZE": "16",
        "FIPMESH_COLMAP_FUSION_CACHE_SIZE": "8",
        "FIPMESH_COLMAP_SECONDARY_ROTATE_DEG": "90",
        "FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS": "y",
        "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X": "1",
        "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y": "2",
        "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z": "3",
        "FIPMESH_COLMAP_SECONDARY_TRANSLATE_X": "4",
        "FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y": "5",
        "FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z": "6",
        "FIPMESH_COLMAP_SECONDARY_ALIGN_MODE": "manual",
        "FIPMESH_COLMAP_CAMERA_MODEL": "OPENCV",
        "FIPMESH_COLMAP_INTRINSICS_OUT": str(Path(s["output_var"]) / "camera_intrinsics.json"),
    }
    assert r.info() == ["[stage 2] COLMAP single side: side1"]


def test_stage2_env_keeps_process_env(s, monkeypatch):
    monkeypatch.setenv("SOME_USER_VAR", "kept")
    r = RecordingRunner(s)
    r.execute_stage(1)
    assert r.calls[0][2]["SOME_USER_VAR"] == "kept"


def test_stage2_secondary_reuses_saved_intrinsics(s, tmp_path):
    intr = tmp_path / "out" / "camera_intrinsics.json"
    intr.parent.mkdir(parents=True)
    intr.write_text("{}")
    r = RecordingRunner(s)
    r.execute_stage(1)
    e2 = r.calls[1][2]
    assert e2["FIPMESH_COLMAP_INTRINSICS_IN"] == str(intr)
    assert "FIPMESH_COLMAP_INTRINSICS_OUT" not in e2
    assert r.info()[-1] == ("[stage 2] side side2: reusing side 1's camera intrinsics (fixed) "
                            "from camera_intrinsics.json")


def test_stage2_no_sharing(s, tmp_path):
    intr = tmp_path / "out" / "camera_intrinsics.json"
    intr.parent.mkdir(parents=True)
    intr.write_text("{}")
    s["share_intr_var"] = False
    r = RecordingRunner(s)
    r.execute_stage(1)
    assert "FIPMESH_COLMAP_INTRINSICS_IN" not in r.calls[1][2]
    assert r.info() == ["[stage 2] COLMAP side 1: side1", "[stage 2] COLMAP side 2: side2"]


def test_stage2_user_intrinsics_file_overrides(s, tmp_path):
    s["intr_file_var"] = " /calib/cam.json "
    r = RecordingRunner(s)
    r.execute_stage(1)
    e1, e2 = r.calls[0][2], r.calls[1][2]
    assert e1["FIPMESH_COLMAP_INTRINSICS_IN"] == "/calib/cam.json"
    assert e1["FIPMESH_COLMAP_INTRINSICS_OUT"] == str(tmp_path / "out" / "camera_intrinsics.json")
    assert e2["FIPMESH_COLMAP_INTRINSICS_IN"] == "/calib/cam.json"
    assert "FIPMESH_COLMAP_INTRINSICS_OUT" not in e2
    assert r.info() == [
        "[stage 2] COLMAP side 1: side1",
        "[stage 2] using fixed intrinsics from /calib/cam.json",
        "[stage 2] COLMAP side 2: side2",
        "[stage 2] using fixed intrinsics from /calib/cam.json",
    ]


def test_stage2_side1_failure_skips_side2(s):
    r = RecordingRunner(s, rc=2)
    assert not r.execute_stage(1)
    assert len(r.calls) == 1


def test_stage2_flat(s, tmp_path):
    s["side1_var"] = s["side2_var"] = ""
    out = tmp_path / "out"
    (out / "processed_masks").mkdir(parents=True)
    r = RecordingRunner(s)
    assert r.execute_stage(1)
    (cmd, _, env), = r.calls
    assert cmd == ["bash", RUN_SH, "-i", rel(out / "processed"), "-o", rel(out / "colmap_out"),
                   "-v", "-m", rel(out / "processed_masks")]
    assert fipmesh(env) == GOLDEN_COLMAP_ENV_DEFAULTS  # no intrinsics vars for a flat set
    assert r.info() == ["[stage 2] COLMAP flat image set"]


# ── Stage 3 ───────────────────────────────────────────────────────────────────

@pytest.fixture
def fused(tmp_path):
    out = tmp_path / "out"
    for side in ("side1", "side2"):
        (out / f"colmap_{side}").mkdir(parents=True)
        (out / f"colmap_{side}" / "fused.ply").write_bytes(b"ply")
    return out


def test_stage3_defaults(s, fused):
    r = RecordingRunner(s)
    assert r.execute_stage(2)
    (cmd, cwd, env), = r.calls
    assert cmd == [
        PY, str(SCRIPT_DIR / "alignment" / "run.py"),
        str(fused / "colmap_side1" / "fused.ply"), str(fused / "colmap_side2" / "fused.ply"),
        "-o", str(fused / "aligned_cloud" / "merged_fpfh.ply"),
        "--method", "fpfh",
        "--samples", "60000",
        "--resolve-seam",
        "--seam-mode", "point",
        "--seam-tau", "0.3",
        "--seam-min-conf", "0.35",
        "--seam-window", "10",
        "--seam-min-count", "20",
        "--seam-passes", "3",
    ]
    assert cwd == str(SCRIPT_DIR / "alignment") and env is None
    assert r.info() == ["[stage 3] Aligning fused.ply + fused.ply"]
    assert r.progress[-1] == (2, 100, "Alignment complete")


def test_stage3_refine_voxel_and_fallbacks(s, fused):
    s.update({
        "align_method_var": "all", "align_samples_var": 5000, "align_voxel_var": " 0.02 ",
        "align_seam_var": True, "align_seam_mode_var": "patch",
        "align_seam_tau_var": " ", "align_seam_minconf_var": "", "align_seam_window_var": "",
        "align_seam_minc_var": 5, "align_seam_passes_var": 1,
        "align_refine_var": True, "align_thresholds_var": " 2,1 ", "align_iters_var": 50,
        "align_points_var": 100000, "align_tol_var": "", "align_band_var": "",
        "align_robust_var": " ",
    })
    r = RecordingRunner(s)
    r.execute_stage(2)
    assert r.calls[0][0][6:] == [
        "--method", "all",
        "--samples", "5000",
        "--voxel", "0.02",
        "--resolve-seam",
        "--seam-mode", "patch",
        "--seam-tau", "0.3",
        "--seam-min-conf", "0",
        "--seam-window", "10",
        "--seam-min-count", "5",
        "--seam-passes", "1",
        "--refine",
        "--refine-thresholds", "2,1",
        "--refine-iters", "50",
        "--refine-points", "100000",
        "--refine-tol", "1e-7",
        "--refine-band", "0",
        "--refine-robust", "0",
        "--report", str(fused / "aligned_cloud" / "icp_report.json"),
    ]


def test_stage3_no_seam_voxel_zero(s, fused):
    s.update({"align_seam_var": False, "align_voxel_var": "0"})
    r = RecordingRunner(s)
    r.execute_stage(2)
    assert r.calls[0][0][6:] == ["--method", "fpfh", "--samples", "60000"]


def test_stage3_single_side_skipped(s):
    s["side2_var"] = ""
    r = RecordingRunner(s)
    assert r.execute_stage(2)
    assert r.calls == []
    assert r.info() == ["[stage 3] Single-side scan — alignment skipped"]
    assert r.progress == [(2, 100, "Skipped (single side)")]
    assert r.states[-1] == (2, "done")


def test_stage3_ply_overrides(s, tmp_path):
    a, b = tmp_path / "a.ply", tmp_path / "b.ply"
    a.write_bytes(b"p")
    b.write_bytes(b"p")
    s.update({"side2_var": "", "align_a_var": f" {a} ", "align_b_var": str(b)})
    r = RecordingRunner(s)
    assert r.execute_stage(2)
    assert r.calls[0][0][2:4] == [str(a), str(b)]
    assert r.info() == ["[stage 3] Aligning a.ply + b.ply"]


def test_stage3_missing_ply(s):
    r = RecordingRunner(s)
    assert not r.execute_stage(2)
    assert r.calls == []
    missing = Path(s["output_var"]) / "colmap_side1" / "fused.ply"
    assert r.info() == [f"[error] stage 3: PLY not found: {missing}"]


# ── Stage 4 ───────────────────────────────────────────────────────────────────

def recon_expected(input_ply: Path, recon: Path, prune="white", extra=()):
    return [
        PY, str(SCRIPT_DIR / "src" / "reconstruct_mesh.py"),
        "--input", str(input_ply),
        "--output", str(recon / "recon_mesh_recon.obj"),
        "--output-clean-cloud", str(recon / "clean_cloud.ply"),
        "--output-decimated", str(recon / "recon_mesh_recon_decimated.obj"),
        "--prune-fill-color", prune,
        "--max-input-points", "0",
        "--dbscan-max-points", "0",
        "--outlier-nb-neighbors", "32",
        "--outlier-std-ratio", "0.0",
        "--radius-outlier-nb-points", "0",
        "--radius-outlier-radius-factor", "2.2",
        "--dbscan-min-points", "0",
        "--dbscan-eps-factor", "2.2",
        "--dbscan-keep-largest", "1",
        "--dbscan-min-cluster-ratio", "0.02",
        "--normal-max-nn", "96",
        "--normal-orient-k", "64",
        "--poisson-depth", "10",
        "--density-trim-quantile", "0.02",
        "--poisson-crop-scale", "1.05",
        "--fill-holes", "0",
        "--fill-holes-max-size-ratio", "0.3",
        "--fill-holes-passes", "4",
        "--fill-holes-smooth", "0",
        "--simplified-target-vertices", "60000",
        "--component-min-ratio", "0.01",
        "--component-min-triangles", "1000",
        "--component-max-count", "1",
        "--smooth-iters", "1",
        "--decimate-target-triangles", "300000",
        "--normalize-pose", "0",
        "--poisson-linear-fit",
        *extra,
    ]


@pytest.fixture
def merged(tmp_path):
    out = tmp_path / "out"
    (out / "aligned_cloud").mkdir(parents=True)
    (out / "aligned_cloud" / "merged_fpfh.ply").write_bytes(b"ply")
    return out


def test_stage4_defaults(s, merged):
    r = RecordingRunner(s)
    assert r.execute_stage(3)
    (cmd, cwd, env), = r.calls
    assert cmd == recon_expected(merged / "aligned_cloud" / "merged_fpfh.ply", merged / "recon")
    assert cwd == str(SCRIPT_DIR) and env is None
    assert r.info() == [
        "[stage 4] Reconstructing mesh from merged_fpfh.ply…",
        f"[stage 4] Warning: GLTF not found at {merged / 'recon' / 'recon_mesh_recon.gltf'}",
        "[stage 4] No artifact name set — skipping info.txt",
    ]
    assert r.progress[-1] == (3, 100, "Reconstruction complete")


def test_stage4_single_side_camera_centers_and_options(s, tmp_path):
    out = tmp_path / "out"
    col = out / "colmap_side1"
    col.mkdir(parents=True)
    (col / "fused.ply").write_bytes(b"ply")
    (col / "camera_centers.json").write_text("{}")
    s.update({
        "side2_var": "", "bg_var": "transparent", "r_poisson_linear": False,
        "r_fill_holes": True, "r_fill_smooth": True, "r_normalize_pose": True,
        "r_density_trim": 0.003, "r_outlier_std": 2.5, "r_max_input_pts": 1000000,
    })
    r = RecordingRunner(s)
    assert r.execute_stage(3)
    cmd = r.calls[0][0]
    exp = recon_expected(col / "fused.ply", out / "recon", prune="none",
                         extra=["--side1-camera-centers", str(col / "camera_centers.json")])
    exp.remove("--poisson-linear-fit")
    for flag, val in [("--fill-holes", "1"), ("--fill-holes-smooth", "1"),
                      ("--normalize-pose", "1"), ("--density-trim-quantile", "0.003"),
                      ("--outlier-std-ratio", "2.5"), ("--max-input-points", "1000000")]:
        exp[exp.index(flag) + 1] = val
    assert cmd == exp


def test_stage4_black_background_prunes_black(s, merged):
    s["bg_var"] = "black"
    r = RecordingRunner(s)
    r.execute_stage(3)
    cmd = r.calls[0][0]
    assert cmd[cmd.index("--prune-fill-color") + 1] == "black"


def test_stage4_copies_and_info_txt(s, merged):
    recon = merged / "recon"
    recon.mkdir()
    (recon / "recon_mesh_recon.gltf").write_text("gltf")
    (recon / "recon_mesh_recon_simplified.glb").write_bytes(b"glb")
    s.update({"meta_name_var": " Tablet 7 ", "meta_type_var": " ", "meta_desc_text": " Clay.\n",
              "meta_link_var": "https://x", "meta_link_label_var": "Src"})
    r = RecordingRunner(s)
    assert r.execute_stage(3)
    assert (merged / "model.gltf").read_text() == "gltf"
    assert (merged / "model_simplified.glb").read_bytes() == b"glb"
    info = (merged / "info.txt").read_text()
    assert "Name: Tablet 7" in info and "Type: tablet" in info and "Description: Clay." in info
    assert "Link: https://x" in info and "Link Label: Src" in info
    assert r.info()[1:] == [
        f"[stage 4] GLTF → {merged / 'model.gltf'}",
        f"[stage 4] Simplified GLB → {merged / 'model_simplified.glb'}",
        f"[stage 4] info.txt → {merged / 'info.txt'}",
    ]


def test_stage4_missing_input(s):
    r = RecordingRunner(s)
    assert not r.execute_stage(3)
    assert r.calls == []
    assert r.info()[0].startswith("[error] stage 4: input PLY not found: ")


def test_stage4_failed_process_skips_copies(s, merged):
    recon = merged / "recon"
    recon.mkdir()
    (recon / "recon_mesh_recon.gltf").write_text("gltf")
    r = RecordingRunner(s, rc=1)
    assert not r.execute_stage(3)
    assert not (merged / "model.gltf").exists()


# ── run all ───────────────────────────────────────────────────────────────────

def test_execute_all_stops_after_failure(s):
    r = RecordingRunner(s)   # stage 3 fails: no fused.ply for the recorded COLMAP run
    assert not r.execute_all()
    assert r.states == [(0, "running"), (0, "done"), (1, "running"), (1, "done"),
                        (2, "running"), (2, "failed")]
    assert r.info()[-1] == "[pipeline] stopped after stage 3 failed"


def test_stage_exception_marks_failed(s, monkeypatch):
    r = RecordingRunner(s)

    def boom(on_progress):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(r, "_run_stage_1_bg_removal", boom)
    assert not r.execute_stage(0)
    assert r.info() == ["[error] stage 1: kaboom"]
    assert r.states[-1] == (0, "failed")
