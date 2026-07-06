#!/usr/bin/env python3
"""
Run a COLMAP SfM + dense MVS pipeline and export a fused dense point cloud.

Outputs a dense point cloud (typically `fused.ply`) that can be consumed by the
Open3D meshing pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
import re

def make_snapshot_dir(output_path: Path) -> Path:
    snap_dir = output_path.parent / (output_path.stem + "_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    return snap_dir


def save_pcd_snapshot(path: Path, name: str, pcd: o3d.geometry.PointCloud) -> None:
    if pcd is None or len(pcd.points) == 0:
        return
    out = path / f"{name}.ply"
    o3d.io.write_point_cloud(str(out), pcd, write_ascii=False)


def save_mesh_snapshot(path: Path, name: str, mesh: o3d.geometry.TriangleMesh) -> None:
    if mesh is None or len(mesh.vertices) == 0:
        return
    out = path / f"{name}.ply"
    o3d.io.write_triangle_mesh(str(out), mesh, write_ascii=False)

def get_colmap_version(colmap_bin="colmap"):
    try:
        out = subprocess.check_output(
            [colmap_bin, "help"],
            stderr=subprocess.STDOUT
        ).decode()

        # Look for: "COLMAP 3.8" or similar
        m = re.search(r"COLMAP\s+(\d+)\.(\d+)", out)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except Exception:
        pass

    return (0, 0)

def supports_option(colmap_bin, matcher_cmd, option):
    probe_commands = (
        [colmap_bin, matcher_cmd, "-h"],
        [colmap_bin, matcher_cmd, "--help"],
        [colmap_bin, "help", matcher_cmd],
    )
    for probe_cmd in probe_commands:
        try:
            result = subprocess.run(
                probe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        except Exception:
            continue
        if option in (result.stdout or ""):
            return True
    return False


def first_supported_option(colmap_bin, matcher_cmd, *options):
    for option in options:
        if supports_option(colmap_bin, matcher_cmd, option):
            return option
    return None

def _cpu_threads_default() -> int:
    return max(1, int(os.cpu_count() or 1))


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return int(value)
    except ValueError:
        return fallback


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        return float(value)
    except ValueError:
        return fallback


def _read_exif_focal_length_px(image_dir: Path) -> float | None:
    """Return focal length in pixels from the first image's EXIF, or None if unavailable."""
    try:
        from PIL import Image as PilImage
    except ImportError:
        return None
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    first = next(
        (p for p in sorted(image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in image_exts),
        None,
    )
    if first is None:
        return None
    try:
        with PilImage.open(first) as img:
            width = img.width
            exif = img.getexif()
            if not exif:
                return None
            focal_mm = exif.get(37386)
            fp_xres = exif.get(41486)
            fp_unit = exif.get(41488)
            if focal_mm and fp_xres and fp_unit:
                f = float(focal_mm)
                r = float(fp_xres)
                if fp_unit == 2 and r > 0:
                    return f * r / 25.4
                if fp_unit == 3 and r > 0:
                    return f * r / 10.0
            focal_35mm = exif.get(41989)
            if focal_35mm and width > 0:
                return float(focal_35mm) * width / 36.0
    except Exception:
        pass
    return None


def _get_first_image_size(image_dir: Path) -> tuple[int, int]:
    """Return (width, height) of the first image in the directory."""
    try:
        from PIL import Image as PilImage
    except ImportError:
        return (0, 0)
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    first = next(
        (p for p in sorted(image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in image_exts),
        None,
    )
    if first is None:
        return (0, 0)
    try:
        with PilImage.open(first) as img:
            return (img.width, img.height)
    except Exception:
        return (0, 0)


def _camera_params_str(camera_model: str, focal_px: float, img_w: int, img_h: int) -> str | None:
    """Build COLMAP --ImageReader.camera_params string for common models."""
    cx = img_w / 2.0
    cy = img_h / 2.0
    single_focal = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_OPENCV_FISHEYE"}
    double_focal = {"PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}
    if camera_model in single_focal:
        return f"{focal_px:.4f},{cx:.4f},{cy:.4f}"
    if camera_model in double_focal:
        return f"{focal_px:.4f},{focal_px:.4f},{cx:.4f},{cy:.4f}"
    return None


def _quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float):
    """Convert unit quaternion to 3×3 rotation matrix (numpy array)."""
    import numpy as np
    n = qw * qw + qx * qx + qy * qy + qz * qz
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw)],
        [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw)],
        [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _read_camera_centers_from_txt(images_txt: Path) -> list[list[float]]:
    """Parse COLMAP images.txt and return world-space camera centers."""
    import numpy as np
    centers: list[list[float]] = []
    try:
        with images_txt.open("r", encoding="utf-8") as f:
            lines = [ln.rstrip() for ln in f if ln.strip() and not ln.startswith("#")]
        for i in range(0, len(lines) - 1, 2):
            parts = lines[i].split()
            if len(parts) < 8:
                continue
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            R = _quat_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=np.float64)
            center = -(R.T @ t)
            centers.append([float(center[0]), float(center[1]), float(center[2])])
    except Exception as exc:
        print(f"warning: could not parse {images_txt}: {exc}", file=sys.stderr)
    return centers


def parse_args() -> argparse.Namespace:
    cpu_threads_default = _cpu_threads_default()
    p = argparse.ArgumentParser(description="Run COLMAP dense MVS and export fused cloud.")
    p.add_argument("--images", required=True, help="Input image directory.")
    p.add_argument("--workspace", required=True, help="COLMAP workspace directory.")
    p.add_argument("--dense-cloud-out", required=True, help="Output fused dense cloud path (PLY).")
    p.add_argument(
        "--colmap-bin",
        default=os.environ.get("FIPMESH_COLMAP_BIN", "colmap"),
        help="COLMAP executable name/path (default: env FIPMESH_COLMAP_BIN or 'colmap').",
    )
    p.add_argument(
        "--matcher",
        choices=("exhaustive", "sequential"),
        default=os.environ.get("FIPMESH_COLMAP_MATCHER", "exhaustive"),
        help="Feature matcher type (default: exhaustive).",
    )
    p.add_argument(
        "--single-camera",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_SINGLE_CAMERA", "1")),
        help="Pass ImageReader.single_camera (0/1, default: 1).",
    )
    p.add_argument(
        "--single-camera-per-folder",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_SINGLE_CAMERA_PER_FOLDER", "0")),
        help=(
            "Treat each immediate subfolder of an image set as photos from a distinct "
            "physical camera, so COLMAP estimates separate intrinsics per camera "
            "(ImageReader.single_camera_per_folder). Use this for mixed-camera captures "
            "where two (or more) different cameras each ring-photograph the same side, "
            "e.g. <images>/camA/*.jpg and <images>/camB/*.jpg. "
            "Overrides --single-camera and disables the global --focal-length/EXIF "
            "override (each camera's focal length is auto-read from its own EXIF instead). "
            "(0/1, default: 0). Env: FIPMESH_COLMAP_SINGLE_CAMERA_PER_FOLDER"
        ),
    )
    p.add_argument(
        "--use-gpu",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_USE_GPU", "0")),
        help="Use GPU for SIFT/matching when available (0/1, default: 0).",
    )
    p.add_argument(
        "--gpu-index",
        default=os.environ.get("FIPMESH_COLMAP_GPU_INDEX", "-1"),
        help="GPU index/list for COLMAP (e.g. 0 or 0,1; -1 means all visible GPUs).",
    )
    p.add_argument(
        "--extract-threads",
        type=int,
        default=_env_int("FIPMESH_COLMAP_EXTRACT_THREADS", cpu_threads_default),
        help="COLMAP SIFT extraction thread cap (default: all available CPU threads).",
    )
    p.add_argument(
        "--match-threads",
        type=int,
        default=_env_int("FIPMESH_COLMAP_MATCH_THREADS", cpu_threads_default),
        help="COLMAP matching thread cap (default: all available CPU threads).",
    )
    p.add_argument(
        "--mapper-threads",
        type=int,
        default=_env_int("FIPMESH_COLMAP_MAPPER_THREADS", cpu_threads_default),
        help="COLMAP mapper thread cap (default: all available CPU threads).",
    )
    p.add_argument(
        "--fusion-threads",
        type=int,
        default=_env_int("FIPMESH_COLMAP_FUSION_THREADS", cpu_threads_default),
        help="COLMAP stereo_fusion thread cap (default: all available CPU threads).",
    )
    p.add_argument(
        "--sift-max-image-size",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_SIFT_MAX_IMAGE_SIZE", "1600")),
        help=(
            "Resize images before SIFT extraction to this max dimension. "
            "Lower uses less RAM (default: 1600)."
        ),
    )
    p.add_argument(
        "--sift-max-num-features",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES", "8192")),
        help="SiftExtraction.max_num_features (default: 8192).",
    )
    p.add_argument(
        "--sift-peak-threshold",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD", "0.00667")),
        help="SiftExtraction.peak_threshold (default: 0.00667). Lower detects more features.",
    )
    p.add_argument(
        "--sift-edge-threshold",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD", "10.0")),
        help="SiftExtraction.edge_threshold (default: 10). Higher keeps more edge-like features.",
    )
    p.add_argument(
        "--sift-domain-size-pooling",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING", "0")),
        help="Enable SiftExtraction.domain_size_pooling (0/1, default: 0).",
    )
    p.add_argument(
        "--sift-estimate-affine-shape",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE", "0")),
        help="Enable SiftExtraction.estimate_affine_shape (0/1, default: 0).",
    )
    p.add_argument(
        "--patch-match-max-image-size",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_PATCH_MAX_IMAGE_SIZE", "1600")),
        help="PatchMatchStereo max image size (default: 1600).",
    )
    p.add_argument(
        "--fusion-max-image-size",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_FUSION_MAX_IMAGE_SIZE", "1600")),
        help="StereoFusion max image size (default: 1600).",
    )
    p.add_argument(
        "--input-image-scale",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_IMAGE_SCALE", "1.0")),
        help=(
            "Resize input images to this scale before COLMAP runs "
            "(0 < scale <= 1, default: 1.0). Example: 0.25 keeps 25%% size."
        ),
    )
    p.add_argument(
        "--input-image-stride",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_IMAGE_STRIDE", "1")),
        help=(
            "Keep every Nth input image before COLMAP runs "
            "(N >= 1, default: 1). Example: 3 keeps every 3rd photo."
        ),
    )
    p.add_argument(
        "--quality",
        choices=("low", "medium", "high", "extreme"),
        default=os.environ.get("FIPMESH_COLMAP_QUALITY", "high"),
        help="Quality preset for dense reconstruction options (default: high).",
    )
    p.add_argument(
        "--patch-match-window-radius",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_PATCH_WINDOW_RADIUS", "5")),
        help="PatchMatchStereo window radius (default: 5).",
    )
    p.add_argument(
        "--patch-match-num-samples",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_PATCH_NUM_SAMPLES", "15")),
        help="PatchMatchStereo num_samples (default: 15).",
    )
    p.add_argument(
        "--patch-match-num-iterations",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_PATCH_NUM_ITERATIONS", "5")),
        help="PatchMatchStereo num_iterations (default: 5).",
    )
    p.add_argument(
        "--patch-match-filter-min-ncc",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_PATCH_FILTER_MIN_NCC", "0.1")),
        help="PatchMatchStereo filter_min_ncc (default: 0.1).",
    )
    p.add_argument(
        "--patch-match-filter-min-consistent",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_PATCH_FILTER_MIN_CONSISTENT", "2")),
        help="PatchMatchStereo filter_min_num_consistent (default: 2).",
    )
    p.add_argument(
        "--patch-match-cache-size",
        type=int,
        default=_env_int("FIPMESH_COLMAP_PATCH_CACHE_SIZE", 32),
        help="PatchMatchStereo cache_size in GB (default: 32).",
    )
    p.add_argument(
        "--fusion-min-num-pixels",
        type=int,
        default=int(os.environ.get("FIPMESH_COLMAP_FUSION_MIN_NUM_PIXELS", "5")),
        help="StereoFusion min_num_pixels (default: 5).",
    )
    p.add_argument(
        "--fusion-max-reproj-error",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_FUSION_MAX_REPROJ_ERROR", "2.0")),
        help="StereoFusion max_reproj_error (default: 2.0).",
    )
    p.add_argument(
        "--fusion-max-depth-error",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_FUSION_MAX_DEPTH_ERROR", "0.01")),
        help="StereoFusion max_depth_error (default: 0.01).",
    )
    p.add_argument(
        "--fusion-max-normal-error",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_FUSION_MAX_NORMAL_ERROR", "10.0")),
        help="StereoFusion max_normal_error (default: 10.0).",
    )
    p.add_argument(
        "--fusion-cache-size",
        type=int,
        default=_env_int("FIPMESH_COLMAP_FUSION_CACHE_SIZE", 32),
        help="StereoFusion cache_size in GB (default: 32).",
    )
    p.add_argument(
        "--fusion-use-cache",
        type=int,
        default=_env_int("FIPMESH_COLMAP_FUSION_USE_CACHE", 0),
        help="Enable StereoFusion cache (0/1, default: 0).",
    )
    p.add_argument(
        "--match-guided",
        type=int,
        default=_env_int("FIPMESH_COLMAP_MATCH_GUIDED", 0),
        help="Enable guided matching for feature matching (0/1, default: 0).",
    )
    p.add_argument(
        "--match-max-num-matches",
        type=int,
        default=_env_int("FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES", 32768),
        help="FeatureMatching.max_num_matches (default: 32768).",
    )
    p.add_argument(
        "--clean-workspace",
        action="store_true",
        help="Delete existing COLMAP workspace contents before running.",
    )
    p.add_argument(
        "--images-secondary",
        default=os.environ.get("FIPMESH_COLMAP_IMAGES_SECONDARY", ""),
        help=(
            "Optional second image directory. If set, this script runs two independent "
            "reconstructions and merges them before output."
        ),
    )
    p.add_argument(
        "--secondary-pre-rotate-deg",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_ROTATE_DEG", 180.0),
        help=(
            "Pre-rotate the secondary cloud by this angle before registration/merge "
            "(default: 180)."
        ),
    )
    p.add_argument(
        "--secondary-pre-rotate-axis",
        default=os.environ.get("FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS", "primary_frame_x"),
        help=(
            "Rotation axis for secondary pre-rotation: x, y, z, principal/principal_major, "
            "principal_minor1, principal_minor2, primary_principal/principal_major, "
            "primary_principal_minor1, primary_principal_minor2, primary_frame_x, "
            "primary_frame_y, primary_frame_z, vector, auto_halfturn, or "
            "pca_halfturn_auto (default: primary_frame_x)."
        ),
    )
    p.add_argument(
        "--secondary-pre-rotate-axis-vector",
        type=float,
        nargs=3,
        metavar=("AX", "AY", "AZ"),
        default=None,
        help=(
            "Arbitrary world-space axis vector for secondary pre-rotation when "
            "--secondary-pre-rotate-axis=vector."
        ),
    )
    p.add_argument(
        "--secondary-auto-halfturn-axes",
        type=int,
        default=_env_int("FIPMESH_COLMAP_SECONDARY_AUTO_HALFTURN_AXES", 48),
        help=(
            "When secondary pre-rotation axis is auto_halfturn, search this many arbitrary "
            "axis candidates for the best 180-degree flip (default: 48)."
        ),
    )
    p.add_argument(
        "--secondary-extra-rotate-x",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X", 0.0),
        help="Additional secondary rotation around x after the main flip (default: 0).",
    )
    p.add_argument(
        "--secondary-extra-rotate-y",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y", 0.0),
        help="Additional secondary rotation around y after the main flip (default: 0).",
    )
    p.add_argument(
        "--secondary-extra-rotate-z",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z", 0.0),
        help="Additional secondary rotation around z after the main flip (default: 0).",
    )
    p.add_argument(
        "--secondary-translate-x",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_TRANSLATE_X", 0.0),
        help="Translate the secondary cloud by this amount along x before merge (default: 0).",
    )
    p.add_argument(
        "--secondary-translate-y",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y", 0.0),
        help="Translate the secondary cloud by this amount along y before merge (default: 0).",
    )
    p.add_argument(
        "--secondary-translate-z",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z", 0.0),
        help="Translate the secondary cloud by this amount along z before merge (default: 0).",
    )
    p.add_argument(
        "--secondary-align-mode",
        choices=(
            "auto",
            "off",
            "centroid",
            "centroid_icp",
            "centroid_icp_overlap",
            "centroid_icp_overlap_capped",
            "centroid_overlap_translate",
        ),
        default=(
            os.environ.get("FIPMESH_COLMAP_SECONDARY_ALIGN_MODE", "centroid_icp_overlap").strip().lower()
            or "centroid_icp_overlap"
        ),
        help=(
            "How to merge the secondary cloud after the pre-transform: "
            "'auto' runs feature registration, 'centroid' aligns cloud centroids, "
            "'centroid_icp' runs centroid initialization followed by ICP, "
            "'centroid_icp_overlap' refines that ICP on the overlap band, "
            "'centroid_icp_overlap_capped' does the same but caps how much extra rotation ICP can add, "
            "'centroid_overlap_translate' keeps the pre-rotation fixed and refines translation only, "
            "'off' appends it as-is (default: centroid_icp_overlap)."
        ),
    )
    p.add_argument(
        "--secondary-align-max-rotation-deg",
        type=float,
        default=_env_float("FIPMESH_COLMAP_SECONDARY_ALIGN_MAX_ROTATION_DEG", 20.0),
        help=(
            "Maximum extra rotation allowed during capped secondary alignment modes "
            "(default: 20 degrees)."
        ),
    )
    p.add_argument(
        "--camera-model",
        default=os.environ.get("FIPMESH_COLMAP_CAMERA_MODEL", ""),
        help=(
            "COLMAP ImageReader camera model (e.g. SIMPLE_PINHOLE, PINHOLE, RADIAL, OPENCV). "
            "Empty means let COLMAP decide (default: empty). env: FIPMESH_COLMAP_CAMERA_MODEL."
        ),
    )
    p.add_argument(
        "--focal-length",
        type=float,
        default=float(os.environ.get("FIPMESH_COLMAP_FOCAL_LENGTH", "0")),
        help=(
            "Initial focal length in pixels passed to COLMAP. "
            "0 = auto-detect from image EXIF (default: 0). env: FIPMESH_COLMAP_FOCAL_LENGTH."
        ),
    )
    return p.parse_args()


def apply_quality_preset(args: argparse.Namespace) -> None:
    defaults = {
        "sift_max_image_size": int(os.environ.get("FIPMESH_COLMAP_SIFT_MAX_IMAGE_SIZE", "1600")),
        "sift_max_num_features": int(os.environ.get("FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES", "8192")),
        "sift_peak_threshold": float(os.environ.get("FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD", "0.00667")),
        "sift_edge_threshold": float(os.environ.get("FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD", "10.0")),
        "sift_domain_size_pooling": int(
            os.environ.get("FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING", "0")
        ),
        "sift_estimate_affine_shape": int(
            os.environ.get("FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE", "0")
        ),
        "patch_match_max_image_size": int(
            os.environ.get("FIPMESH_COLMAP_PATCH_MAX_IMAGE_SIZE", "1600")
        ),
        "fusion_max_image_size": int(os.environ.get("FIPMESH_COLMAP_FUSION_MAX_IMAGE_SIZE", "1600")),
        "patch_match_window_radius": int(
            os.environ.get("FIPMESH_COLMAP_PATCH_WINDOW_RADIUS", "5")
        ),
        "patch_match_num_samples": int(os.environ.get("FIPMESH_COLMAP_PATCH_NUM_SAMPLES", "15")),
        "patch_match_num_iterations": int(
            os.environ.get("FIPMESH_COLMAP_PATCH_NUM_ITERATIONS", "5")
        ),
        "patch_match_filter_min_ncc": float(
            os.environ.get("FIPMESH_COLMAP_PATCH_FILTER_MIN_NCC", "0.1")
        ),
        "patch_match_filter_min_consistent": int(
            os.environ.get("FIPMESH_COLMAP_PATCH_FILTER_MIN_CONSISTENT", "2")
        ),
        "fusion_min_num_pixels": int(os.environ.get("FIPMESH_COLMAP_FUSION_MIN_NUM_PIXELS", "5")),
        "fusion_max_reproj_error": float(
            os.environ.get("FIPMESH_COLMAP_FUSION_MAX_REPROJ_ERROR", "2.0")
        ),
        "fusion_max_depth_error": float(
            os.environ.get("FIPMESH_COLMAP_FUSION_MAX_DEPTH_ERROR", "0.01")
        ),
        "fusion_max_normal_error": float(
            os.environ.get("FIPMESH_COLMAP_FUSION_MAX_NORMAL_ERROR", "10.0")
        ),
        "match_guided": int(os.environ.get("FIPMESH_COLMAP_MATCH_GUIDED", "0")),
        "match_max_num_matches": int(
            os.environ.get("FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES", "32768")
        ),
    }

    profiles: dict[str, dict[str, int | float]] = {
        "low": {
            "sift_max_image_size": 1200,
            "sift_max_num_features": 8192,
            "sift_peak_threshold": 0.00667,
            "sift_edge_threshold": 10.0,
            "sift_domain_size_pooling": 0,
            "sift_estimate_affine_shape": 0,
            "patch_match_max_image_size": 1200,
            "fusion_max_image_size": 1200,
            "patch_match_window_radius": 5,
            "patch_match_num_samples": 12,
            "patch_match_num_iterations": 4,
            "patch_match_filter_min_ncc": 0.08,
            "patch_match_filter_min_consistent": 2,
            "fusion_min_num_pixels": 4,
            "fusion_max_reproj_error": 2.5,
            "fusion_max_depth_error": 0.02,
            "fusion_max_normal_error": 12.0,
            "match_guided": 0,
            "match_max_num_matches": 32768,
        },
        "medium": {
            "sift_max_image_size": 1600,
            "sift_max_num_features": 10000,
            "sift_peak_threshold": 0.0055,
            "sift_edge_threshold": 10.0,
            "sift_domain_size_pooling": 0,
            "sift_estimate_affine_shape": 0,
            "patch_match_max_image_size": 1600,
            "fusion_max_image_size": 1600,
            "patch_match_window_radius": 5,
            "patch_match_num_samples": 15,
            "patch_match_num_iterations": 5,
            "patch_match_filter_min_ncc": 0.10,
            "patch_match_filter_min_consistent": 2,
            "fusion_min_num_pixels": 5,
            "fusion_max_reproj_error": 2.0,
            "fusion_max_depth_error": 0.01,
            "fusion_max_normal_error": 10.0,
            "match_guided": 0,
            "match_max_num_matches": 32768,
        },
        "high": {
            "sift_max_image_size": 2400,
            "sift_max_num_features": 16000,
            "sift_peak_threshold": 0.0045,
            "sift_edge_threshold": 12.0,
            "sift_domain_size_pooling": 1,
            "sift_estimate_affine_shape": 0,
            "patch_match_max_image_size": 2400,
            "fusion_max_image_size": 2400,
            "patch_match_window_radius": 6,
            "patch_match_num_samples": 20,
            "patch_match_num_iterations": 7,
            "patch_match_filter_min_ncc": 0.12,
            "patch_match_filter_min_consistent": 3,
            "fusion_min_num_pixels": 8,
            "fusion_max_reproj_error": 1.5,
            "fusion_max_depth_error": 0.006,
            "fusion_max_normal_error": 8.0,
            "match_guided": 1,
            "match_max_num_matches": 65536,
        },
        "extreme": {
            "sift_max_image_size": 3000,
            "sift_max_num_features": 24000,
            "sift_peak_threshold": 0.0035,
            "sift_edge_threshold": 14.0,
            "sift_domain_size_pooling": 1,
            "sift_estimate_affine_shape": 0,
            "patch_match_max_image_size": 3000,
            "fusion_max_image_size": 3000,
            "patch_match_window_radius": 7,
            "patch_match_num_samples": 24,
            "patch_match_num_iterations": 9,
            "patch_match_filter_min_ncc": 0.14,
            "patch_match_filter_min_consistent": 4,
            "fusion_min_num_pixels": 10,
            "fusion_max_reproj_error": 1.2,
            "fusion_max_depth_error": 0.004,
            "fusion_max_normal_error": 7.0,
            "match_guided": 1,
            "match_max_num_matches": 131072,
        },
    }

    profile = profiles.get(str(args.quality), {})
    for key, value in profile.items():
        if getattr(args, key) == defaults[key]:
            setattr(args, key, value)


def run(cmd: list[str]) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _select_image_reencoder():
    try:
        from PIL import Image, ImageOps
    except Exception as pil_exc:
        Image = None
        ImageOps = None
    else:
        def _encode_with_pillow(src: Path, dst: Path) -> None:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img)
                ext = src.suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    rgb = img.convert("RGB")
                    rgb.save(dst, quality=95, optimize=True, exif=b"")
                elif ext == ".png":
                    img.save(dst, optimize=True)
                else:
                    img.save(dst)

        return _encode_with_pillow, "pillow"

    try:
        import cv2
    except Exception as cv2_exc:
        raise RuntimeError(
            "Cannot sanitize images for image_undistorter retry: neither Pillow nor "
            f"OpenCV python is available (Pillow error: {pil_exc!r}, OpenCV error: {cv2_exc!r})."
        ) from cv2_exc

    def _encode_with_cv2(src: Path, dst: Path) -> None:
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode image: {src}")
        ext = src.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        elif ext == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        if not cv2.imwrite(str(dst), image, params):
            raise RuntimeError(f"OpenCV could not encode image: {dst}")

    return _encode_with_cv2, "opencv"


def _rewrite_images_without_metadata(src_root: Path, dst_root: Path) -> Path:
    if not src_root.is_dir():
        raise FileNotFoundError(f"image root not found: {src_root}")
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    encode_image, encoder_name = _select_image_reencoder()
    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".ppm",
        ".pgm",
        ".pnm",
    }

    rewritten = 0
    copied = 0
    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix.lower() in image_exts:
            encode_image(src, dst)
            rewritten += 1
        else:
            shutil.copy2(src, dst)
            copied += 1

    print(
        "image metadata sanitize copy complete: "
        f"rewritten={rewritten} copied_non_images={copied} encoder={encoder_name}"
    )
    return dst_root


def _select_image_resizer():
    try:
        from PIL import Image, ImageOps
    except Exception as pil_exc:
        Image = None
        ImageOps = None
    else:
        def _resize_with_pillow(src: Path, dst: Path, scale: float) -> None:
            with Image.open(src) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                new_width = max(1, int(round(width * scale)))
                new_height = max(1, int(round(height * scale)))
                if new_width != width or new_height != height:
                    lanczos = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                    img = img.resize((new_width, new_height), lanczos)
                ext = src.suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    rgb = img.convert("RGB")
                    rgb.save(dst, quality=95, optimize=True, exif=b"")
                elif ext == ".png":
                    img.save(dst, optimize=True)
                else:
                    img.save(dst)

        return _resize_with_pillow, "pillow"

    try:
        import cv2
    except Exception as cv2_exc:
        raise RuntimeError(
            "Cannot resize images: neither Pillow nor OpenCV python is available "
            f"(Pillow error: {pil_exc!r}, OpenCV error: {cv2_exc!r})."
        ) from cv2_exc

    def _resize_with_cv2(src: Path, dst: Path, scale: float) -> None:
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not decode image: {src}")
        height, width = image.shape[:2]
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if new_width != width or new_height != height:
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        ext = src.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        elif ext == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        if not cv2.imwrite(str(dst), image, params):
            raise RuntimeError(f"OpenCV could not encode image: {dst}")

    return _resize_with_cv2, "opencv"


def _prepare_input_images(
    src_root: Path,
    dst_root: Path,
    *,
    scale: float,
    image_stride: int,
) -> Path:
    if not src_root.is_dir():
        raise FileNotFoundError(f"image root not found: {src_root}")
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    resize_image = None
    encoder_name = "copy"
    if scale < 1.0:
        resize_image, encoder_name = _select_image_resizer()
    image_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".ppm",
        ".pgm",
        ".pnm",
    }

    all_files = sorted(f for f in src_root.rglob("*") if f.is_file())
    image_files = [(i, f) for i, f in enumerate(f for f in all_files if f.suffix.lower() in image_exts)]
    non_image_files = [f for f in all_files if f.suffix.lower() not in image_exts]

    counters = {"kept": 0, "skipped": 0, "copied": 0}
    lock = threading.Lock()

    def _process_image(image_index: int, src: Path) -> None:
        keep_image = (image_index % image_stride) == 0
        if not keep_image:
            with lock:
                counters["skipped"] += 1
            return
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if resize_image is not None:
            resize_image(src, dst, scale)
        else:
            shutil.copy2(src, dst)
        with lock:
            counters["kept"] += 1

    def _copy_non_image(src: Path) -> None:
        dst = dst_root / src.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        with lock:
            counters["copied"] += 1

    with concurrent.futures.ThreadPoolExecutor() as pool:
        img_futs = [pool.submit(_process_image, idx, src) for idx, src in image_files]
        non_futs = [pool.submit(_copy_non_image, src) for src in non_image_files]
        for fut in img_futs + non_futs:
            fut.result()

    print(
        "prepared image copy complete: "
        f"scale={scale:g} stride={image_stride} kept_images={counters['kept']} "
        f"skipped_images={counters['skipped']} copied_non_images={counters['copied']} encoder={encoder_name}"
    )
    return dst_root


def _build_image_undistorter_cmd(
    colmap_bin: str,
    image_path: Path,
    sparse_model_dir: Path,
    dense_workspace_dir: Path,
    patch_max_image_size: int,
) -> list[str]:
    return [
        colmap_bin,
        "image_undistorter",
        "--image_path",
        str(image_path),
        "--input_path",
        str(sparse_model_dir),
        "--output_path",
        str(dense_workspace_dir),
        "--output_type",
        "COLMAP",
        "--max_image_size",
        str(patch_max_image_size),
    ]


def detect_colmap_option_style(colmap_bin: str) -> str:
    """
    COLMAP CLI option names changed in recent versions:
      old:  SiftExtraction.use_gpu / SiftMatching.use_gpu
      new:  FeatureExtraction.use_gpu / FeatureMatching.use_gpu
    Detect by inspecting feature_extractor help output.
    """
    try:
        proc = subprocess.run(
            [colmap_bin, "feature_extractor", "-h"],
            check=False,
            capture_output=True,
            text=True,
        )
        help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception:
        return "legacy"

    if "FeatureExtraction.use_gpu" in help_text:
        return "modern"
    if "SiftExtraction.use_gpu" in help_text:
        return "legacy"
    # Default to modern for current COLMAP master builds.
    return "modern"


def has_model_files(model_dir: Path) -> bool:
    names = [
        ("cameras.bin", "images.bin", "points3D.bin"),
        ("cameras.txt", "images.txt", "points3D.txt"),
    ]
    for triplet in names:
        if all((model_dir / n).exists() for n in triplet):
            return True
    return False


def points_count(model_dir: Path) -> int:
    for fname in ("points3D.bin", "points3D.txt"):
        fp = model_dir / fname
        if fp.exists():
            try:
                return fp.stat().st_size
            except OSError:
                return 0
    return 0


def list_sparse_models(sparse_root: Path) -> list[Path]:
    candidates = [p for p in sparse_root.iterdir() if p.is_dir() and has_model_files(p)]
    if candidates:
        candidates.sort(key=points_count, reverse=True)
        return candidates
    if has_model_files(sparse_root):
        return [sparse_root]
    raise FileNotFoundError(f"no COLMAP sparse model found under {sparse_root}")


def run_dense_for_model(
    colmap_bin: str,
    images: Path,
    sparse_model_dir: Path,
    dense_workspace_dir: Path,
    cloud_out: Path,
    patch_max_image_size: int,
    fusion_max_image_size: int,
    patch_window_radius: int,
    patch_num_samples: int,
    patch_num_iterations: int,
    patch_filter_min_ncc: float,
    patch_filter_min_consistent: int,
    patch_cache_size: int,
    patch_gpu_index: str,
    fusion_min_num_pixels: int,
    fusion_max_reproj_error: float,
    fusion_max_depth_error: float,
    fusion_max_normal_error: float,
    fusion_threads: int,
    fusion_cache_size: int,
    fusion_use_cache: bool,
    model_label: str,
) -> None:
    """Run COLMAP dense reconstruction (undistort, stereo, fusion)."""
    dense_workspace_dir.mkdir(parents=True, exist_ok=True)

    # 1. Undistort
    print(f"[{model_label}] colmap image_undistorter")
    subprocess.run([
        colmap_bin, "image_undistorter",
        "--image_path", str(images),
        "--input_path", str(sparse_model_dir),
        "--output_path", str(dense_workspace_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(patch_max_image_size),
    ], check=True)

    # 2. Patch Match Stereo
    print(f"[{model_label}] colmap patch_match_stereo")
    subprocess.run([
        colmap_bin, "patch_match_stereo",
        "--workspace_path", str(dense_workspace_dir),
        "--PatchMatchStereo.window_radius", str(patch_window_radius),
        "--PatchMatchStereo.num_samples", str(patch_num_samples),
        "--PatchMatchStereo.num_iterations", str(patch_num_iterations),
        "--PatchMatchStereo.filter_min_ncc", str(patch_filter_min_ncc),
        "--PatchMatchStereo.filter_min_num_consistent", str(patch_filter_min_consistent),
        "--PatchMatchStereo.cache_size", str(patch_cache_size),
        "--PatchMatchStereo.gpu_index", str(patch_gpu_index),
    ], check=True)

    # 3. Stereo Fusion
    print(f"[{model_label}] colmap stereo_fusion")
    subprocess.run([
        colmap_bin, "stereo_fusion",
        "--workspace_path", str(dense_workspace_dir),
        "--output_path", str(cloud_out),
        "--StereoFusion.max_image_size", str(fusion_max_image_size),
        "--StereoFusion.min_num_pixels", str(fusion_min_num_pixels),
        "--StereoFusion.max_reproj_error", str(fusion_max_reproj_error),
        "--StereoFusion.max_depth_error", str(fusion_max_depth_error),
        "--StereoFusion.max_normal_error", str(fusion_max_normal_error),
        "--StereoFusion.cache_size", str(fusion_cache_size),
        "--StereoFusion.use_cache", "1" if fusion_use_cache else "0",
        "--StereoFusion.num_threads", str(fusion_threads),
    ], check=True)


def try_align_cloud_to_base(source, target):
    try:
        import numpy as np
        import open3d as o3d
    except Exception:
        return None, "Open3D unavailable"

    if len(source.points) < 200 or len(target.points) < 200:
        return None, "too few points for robust registration"

    src_extent = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    voxel = max(diag * 0.01, 1e-4)

    src_down = source.voxel_down_sample(voxel)
    tgt_down = target.voxel_down_sample(voxel)
    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return None, "downsampled clouds too small"

    radius_normal = voxel * 2.0
    radius_feature = voxel * 5.0
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        tgt_down,
        src_fpfh,
        tgt_fpfh,
        True,
        voxel * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    if ransac.fitness < 0.02:
        return None, f"RANSAC fitness too low ({ransac.fitness:.4f})"

    icp = o3d.pipelines.registration.registration_icp(
        src_down,
        tgt_down,
        voxel,
        ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    transform = icp.transformation if icp.fitness >= ransac.fitness else ransac.transformation

    aligned = copy.deepcopy(source)
    aligned.transform(transform)
    fitness = max(float(ransac.fitness), float(icp.fitness))
    return aligned, f"aligned (fitness={fitness:.4f}, voxel={voxel:.6f})"


def align_cloud_centroids(source, target):
    try:
        import numpy as np
    except Exception:
        return None, "NumPy unavailable"

    if len(source.points) == 0 or len(target.points) == 0:
        return None, "empty cloud"

    src_pts = np.asarray(source.points, dtype=np.float64)
    tgt_pts = np.asarray(target.points, dtype=np.float64)
    src_center = src_pts.mean(axis=0)
    tgt_center = tgt_pts.mean(axis=0)
    delta = tgt_center - src_center

    aligned = copy.deepcopy(source)
    aligned.translate(delta)
    return aligned, f"centroid translate ({delta[0]:.6f}, {delta[1]:.6f}, {delta[2]:.6f})"


def align_cloud_centroid_icp(source, target):
    try:
        import numpy as np
        import open3d as o3d
    except Exception:
        return None, "Open3D/NumPy unavailable"

    if len(source.points) < 200 or len(target.points) < 200:
        return None, "too few points for robust ICP"

    src_extent = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    voxel = max(diag * 0.01, 1e-4)

    src_down = source.voxel_down_sample(voxel)
    tgt_down = target.voxel_down_sample(voxel)
    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return None, "downsampled clouds too small"

    src_center = np.asarray(src_down.points, dtype=np.float64).mean(axis=0)
    tgt_center = np.asarray(tgt_down.points, dtype=np.float64).mean(axis=0)
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = tgt_center - src_center

    radius_normal = voxel * 2.0
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    result = None
    current = init
    for factor in (8.0, 4.0, 2.0, 1.0):
        max_corr = voxel * factor
        result = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            max_corr,
            current,
            estimation,
        )
        current = result.transformation

    if result is None or result.fitness <= 0.0:
        return None, "ICP failed to converge"

    aligned = copy.deepcopy(source)
    aligned.transform(current)
    translation = current[:3, 3]
    return aligned, (
        "centroid+icp "
        f"(fitness={float(result.fitness):.4f}, rmse={float(result.inlier_rmse):.6f}, "
        f"voxel={voxel:.6f}, t=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}))"
    )


def align_cloud_centroid_icp_overlap(source, target):
    try:
        import numpy as np
        import open3d as o3d
    except Exception:
        return None, "Open3D/NumPy unavailable"

    aligned_base, detail = align_cloud_centroid_icp(source, target)
    if aligned_base is None:
        return None, detail

    src_extent = np.asarray(aligned_base.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    voxel = max(diag * 0.01, 1e-4)

    src_down = aligned_base.voxel_down_sample(voxel)
    tgt_down = target.voxel_down_sample(voxel)
    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return aligned_base, f"{detail}, overlap refine skipped (downsampled clouds too small)"

    d_src = np.asarray(src_down.compute_point_cloud_distance(tgt_down), dtype=np.float64)
    d_tgt = np.asarray(tgt_down.compute_point_cloud_distance(src_down), dtype=np.float64)
    if len(d_src) == 0 or len(d_tgt) == 0:
        return aligned_base, f"{detail}, overlap refine skipped (empty distance arrays)"

    src_thresh = float(np.quantile(d_src, 0.55))
    tgt_thresh = float(np.quantile(d_tgt, 0.55))
    src_idx = np.where(np.isfinite(d_src) & (d_src <= src_thresh))[0]
    tgt_idx = np.where(np.isfinite(d_tgt) & (d_tgt <= tgt_thresh))[0]
    if len(src_idx) < 50 or len(tgt_idx) < 50:
        return aligned_base, f"{detail}, overlap refine skipped (insufficient overlap band)"

    src_band = src_down.select_by_index(src_idx.tolist())
    tgt_band = tgt_down.select_by_index(tgt_idx.tolist())
    radius_normal = voxel * 2.0
    src_band.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_band.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    refine = o3d.pipelines.registration.registration_icp(
        src_band,
        tgt_band,
        voxel * 2.0,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    if refine.fitness <= 0.0:
        return aligned_base, f"{detail}, overlap refine failed"

    aligned = copy.deepcopy(aligned_base)
    aligned.transform(refine.transformation)
    translation = refine.transformation[:3, 3]
    return aligned, (
        "centroid+icp+overlap "
        f"(base={detail}; refine_fitness={float(refine.fitness):.4f}, "
        f"refine_rmse={float(refine.inlier_rmse):.6f}, "
        f"refine_t=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}))"
    )


def _orthonormalize_rotation(rotation):
    import numpy as np

    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    ortho = u @ vt
    if float(np.linalg.det(ortho)) < 0.0:
        u[:, -1] *= -1.0
        ortho = u @ vt
    return ortho


def _axis_angle_from_rotation(rotation):
    import math

    import numpy as np

    r = _orthonormalize_rotation(rotation)
    cos_theta = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_theta)
    if angle <= 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0

    if abs(math.pi - angle) <= 1e-5:
        eigvals, eigvecs = np.linalg.eig(r)
        best_idx = 0
        best_err = float("inf")
        for idx, eig in enumerate(eigvals):
            err = abs(float(np.real(eig)) - 1.0) + abs(float(np.imag(eig)))
            if err < best_err:
                best_err = err
                best_idx = idx
        axis = np.real(eigvecs[:, best_idx]).astype(np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            axis = axis / norm
        return axis, angle

    axis = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=np.float64,
    )
    denom = 2.0 * math.sin(angle)
    if abs(denom) <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), angle
    axis = axis / denom
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        axis = axis / norm
    return axis, angle


def _clamp_transform_rotation(transform, max_rotation_deg):
    import math

    import numpy as np

    max_rotation_deg = max(0.0, float(max_rotation_deg))
    result = np.array(transform, dtype=np.float64, copy=True)
    raw_rotation = _orthonormalize_rotation(result[:3, :3])
    axis, raw_angle = _axis_angle_from_rotation(raw_rotation)
    raw_deg = math.degrees(raw_angle)
    if raw_deg <= max_rotation_deg + 1e-9:
        result[:3, :3] = raw_rotation
        return result, raw_deg, raw_deg

    clamped_rotation = _rotation_matrix_from_vector(axis, max_rotation_deg)[:3, :3]
    result[:3, :3] = clamped_rotation
    return result, raw_deg, max_rotation_deg


def align_cloud_centroid_icp_overlap_capped(source, target, *, max_rotation_deg):
    try:
        import numpy as np
        import open3d as o3d
    except Exception:
        return None, "Open3D/NumPy unavailable"

    if len(source.points) < 200 or len(target.points) < 200:
        return None, "too few points for robust ICP"

    src_extent = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    voxel = max(diag * 0.01, 1e-4)

    src_down = source.voxel_down_sample(voxel)
    tgt_down = target.voxel_down_sample(voxel)
    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        return None, "downsampled clouds too small"

    src_center = np.asarray(src_down.points, dtype=np.float64).mean(axis=0)
    tgt_center = np.asarray(tgt_down.points, dtype=np.float64).mean(axis=0)
    init = np.eye(4, dtype=np.float64)
    init[:3, 3] = tgt_center - src_center

    radius_normal = voxel * 2.0
    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    result = None
    current = init
    max_raw_deg = 0.0
    max_clamped_deg = 0.0
    for factor in (8.0, 4.0, 2.0, 1.0):
        max_corr = voxel * factor
        result = o3d.pipelines.registration.registration_icp(
            src_down,
            tgt_down,
            max_corr,
            current,
            estimation,
        )
        current, raw_deg, clamped_deg = _clamp_transform_rotation(
            result.transformation,
            max_rotation_deg,
        )
        max_raw_deg = max(max_raw_deg, float(raw_deg))
        max_clamped_deg = max(max_clamped_deg, float(clamped_deg))

    if result is None or result.fitness <= 0.0:
        return None, "ICP failed to converge"

    aligned_base = copy.deepcopy(source)
    aligned_base.transform(current)

    src_down = aligned_base.voxel_down_sample(voxel)
    tgt_down = target.voxel_down_sample(voxel)
    if len(src_down.points) < 100 or len(tgt_down.points) < 100:
        translation = current[:3, 3]
        return aligned_base, (
            "centroid+icp+overlap-capped "
            f"(base_fitness={float(result.fitness):.4f}, base_rmse={float(result.inlier_rmse):.6f}, "
            f"voxel={voxel:.6f}, rot<= {max_rotation_deg:g} deg "
            f"(raw_max={max_raw_deg:.3f}, used_max={max_clamped_deg:.3f}), "
            f"t=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}), "
            "overlap refine skipped)"
        )

    d_src = np.asarray(src_down.compute_point_cloud_distance(tgt_down), dtype=np.float64)
    d_tgt = np.asarray(tgt_down.compute_point_cloud_distance(src_down), dtype=np.float64)
    if len(d_src) == 0 or len(d_tgt) == 0:
        translation = current[:3, 3]
        return aligned_base, (
            "centroid+icp+overlap-capped "
            f"(base_fitness={float(result.fitness):.4f}, base_rmse={float(result.inlier_rmse):.6f}, "
            f"voxel={voxel:.6f}, rot<= {max_rotation_deg:g} deg "
            f"(raw_max={max_raw_deg:.3f}, used_max={max_clamped_deg:.3f}), "
            f"t=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}), "
            "overlap refine skipped: empty distance arrays)"
        )

    src_thresh = float(np.quantile(d_src, 0.55))
    tgt_thresh = float(np.quantile(d_tgt, 0.55))
    src_idx = np.where(np.isfinite(d_src) & (d_src <= src_thresh))[0]
    tgt_idx = np.where(np.isfinite(d_tgt) & (d_tgt <= tgt_thresh))[0]
    if len(src_idx) < 50 or len(tgt_idx) < 50:
        translation = current[:3, 3]
        return aligned_base, (
            "centroid+icp+overlap-capped "
            f"(base_fitness={float(result.fitness):.4f}, base_rmse={float(result.inlier_rmse):.6f}, "
            f"voxel={voxel:.6f}, rot<= {max_rotation_deg:g} deg "
            f"(raw_max={max_raw_deg:.3f}, used_max={max_clamped_deg:.3f}), "
            f"t=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}), "
            "overlap refine skipped: insufficient overlap band)"
        )

    src_band = src_down.select_by_index(src_idx.tolist())
    tgt_band = tgt_down.select_by_index(tgt_idx.tolist())
    src_band.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt_band.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    refine = o3d.pipelines.registration.registration_icp(
        src_band,
        tgt_band,
        voxel * 2.0,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    total = refine.transformation @ current if refine.fitness > 0.0 else current
    total, total_raw_deg, total_clamped_deg = _clamp_transform_rotation(total, max_rotation_deg)

    aligned = copy.deepcopy(source)
    aligned.transform(total)
    translation = total[:3, 3]
    return aligned, (
        "centroid+icp+overlap-capped "
        f"(base_fitness={float(result.fitness):.4f}, base_rmse={float(result.inlier_rmse):.6f}, "
        f"refine_fitness={float(refine.fitness):.4f}, refine_rmse={float(refine.inlier_rmse):.6f}, "
        f"voxel={voxel:.6f}, rot<= {max_rotation_deg:g} deg "
        f"(base_raw_max={max_raw_deg:.3f}, base_used_max={max_clamped_deg:.3f}, "
        f"total_raw={total_raw_deg:.3f}, total_used={total_clamped_deg:.3f}), "
        f"t=({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}))"
    )


def _median_nn_translation(source, target, *, max_correspondence_distance):
    import numpy as np
    import open3d as o3d

    if len(source.points) == 0 or len(target.points) == 0:
        return None, "empty cloud"

    src_points = np.asarray(source.points, dtype=np.float64)
    tgt_points = np.asarray(target.points, dtype=np.float64)
    tree = o3d.geometry.KDTreeFlann(target)
    max_sq = float(max_correspondence_distance) * float(max_correspondence_distance)
    deltas = []
    for point in src_points:
        found, idxs, dists = tree.search_knn_vector_3d(point, 1)
        if found <= 0 or not dists:
            continue
        if float(dists[0]) > max_sq:
            continue
        deltas.append(tgt_points[int(idxs[0])] - point)

    if len(deltas) < 25:
        return None, f"too few correspondences ({len(deltas)})"

    delta = np.median(np.asarray(deltas, dtype=np.float64), axis=0)
    return delta, f"nn={len(deltas)}"


def align_cloud_centroid_overlap_translate(source, target):
    try:
        import numpy as np
    except Exception:
        return None, "NumPy unavailable"

    if len(source.points) < 50 or len(target.points) < 50:
        return None, "too few points for overlap translation"

    src_pts = np.asarray(source.points, dtype=np.float64)
    tgt_pts = np.asarray(target.points, dtype=np.float64)
    init_delta = tgt_pts.mean(axis=0) - src_pts.mean(axis=0)

    aligned = copy.deepcopy(source)
    aligned.translate(init_delta)
    refine_total = np.zeros(3, dtype=np.float64)

    src_extent = np.asarray(aligned.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    voxel = max(diag * 0.01, 1e-4)

    notes = []
    for factor in (4.0, 2.0, 1.0):
        src_down = aligned.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) < 50 or len(tgt_down.points) < 50:
            break

        d_src = np.asarray(src_down.compute_point_cloud_distance(tgt_down), dtype=np.float64)
        d_tgt = np.asarray(tgt_down.compute_point_cloud_distance(src_down), dtype=np.float64)
        if len(d_src) == 0 or len(d_tgt) == 0:
            break

        src_thresh = float(np.quantile(d_src, 0.55))
        tgt_thresh = float(np.quantile(d_tgt, 0.55))
        src_idx = np.where(np.isfinite(d_src) & (d_src <= src_thresh))[0]
        tgt_idx = np.where(np.isfinite(d_tgt) & (d_tgt <= tgt_thresh))[0]
        if len(src_idx) < 25 or len(tgt_idx) < 25:
            continue

        src_band = src_down.select_by_index(src_idx.tolist())
        tgt_band = tgt_down.select_by_index(tgt_idx.tolist())
        delta, detail = _median_nn_translation(
            src_band,
            tgt_band,
            max_correspondence_distance=max(voxel * factor, 1e-4),
        )
        if delta is None:
            notes.append(f"{factor:g}x:{detail}")
            continue

        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= 1e-9:
            notes.append(f"{factor:g}x:{detail},dt=(0,0,0)")
            continue

        aligned.translate(delta)
        refine_total += delta
        notes.append(
            f"{factor:g}x:{detail},dt=({delta[0]:.6f},{delta[1]:.6f},{delta[2]:.6f})"
        )

    total = init_delta + refine_total
    notes_text = "; ".join(notes) if notes else "no overlap refinement"
    return aligned, (
        "centroid+overlap-translate "
        f"(init_t=({init_delta[0]:.6f}, {init_delta[1]:.6f}, {init_delta[2]:.6f}), "
        f"refine_t=({refine_total[0]:.6f}, {refine_total[1]:.6f}, {refine_total[2]:.6f}), "
        f"total_t=({total[0]:.6f}, {total[1]:.6f}, {total[2]:.6f}), "
        f"voxel={voxel:.6f}; {notes_text})"
    )


def _principal_axes(points):
    import numpy as np

    if len(points) < 3:
        raise RuntimeError("too few points to compute principal axes")
    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = []
    for idx in order:
        axis_vec = np.asarray(eigvecs[:, int(idx)], dtype=np.float64)
        norm = float(np.linalg.norm(axis_vec))
        if norm <= 1e-12:
            continue
        axes.append(axis_vec / norm)
    if len(axes) < 3:
        raise RuntimeError("principal axes are degenerate")
    return axes[:3]


def _rotation_matrix_from_vector(axis_vec, rotation_degrees):
    import math

    import numpy as np

    axis_vec = np.asarray(axis_vec, dtype=np.float64)
    norm = float(np.linalg.norm(axis_vec))
    if norm <= 1e-12:
        raise RuntimeError("rotation axis is degenerate")
    axis_vec = axis_vec / norm

    theta = math.radians(rotation_degrees)
    c = math.cos(theta)
    s = math.sin(theta)
    x, y, z = axis_vec.tolist()
    t = 1.0 - c
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    return transform


def _axis_vector_for_name(rotation_axis, points, *, reference_points=None, axis_vector=None):
    import numpy as np

    if rotation_axis == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if rotation_axis == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if rotation_axis == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if rotation_axis == "vector":
        if axis_vector is None:
            raise ValueError("vector axis requires an explicit axis_vector")
        vec = np.asarray(axis_vector, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            raise ValueError("vector axis is degenerate")
        return vec / norm
    p_axes = _principal_axes(points)
    if rotation_axis in {"principal", "principal_major"}:
        return p_axes[0]
    if rotation_axis == "principal_minor1":
        return p_axes[1]
    if rotation_axis == "principal_minor2":
        return p_axes[2]
    if rotation_axis in {"primary_principal", "primary_principal_major", "primary_principal_minor1", "primary_principal_minor2"}:
        if reference_points is None:
            raise ValueError(f"{rotation_axis} requires reference_points")
        ref_axes = _principal_axes(np.asarray(reference_points, dtype=np.float64))
        if rotation_axis in {"primary_principal", "primary_principal_major"}:
            return ref_axes[0]
        if rotation_axis == "primary_principal_minor1":
            return ref_axes[1]
        return ref_axes[2]
    raise ValueError(f"unsupported axis name: {rotation_axis}")


def _apply_pre_transform_about_axis_vector(
    cloud,
    *,
    axis_vec,
    degrees,
    extra_rotate_x=0.0,
    extra_rotate_y=0.0,
    extra_rotate_z=0.0,
    translate_x=0.0,
    translate_y=0.0,
    translate_z=0.0,
    rotate_about_center=False,
    rotation_center=None,
):
    import numpy as np

    transformed = copy.deepcopy(cloud)
    points = np.asarray(transformed.points, dtype=np.float64)
    transform = _rotation_matrix_from_vector(axis_vec, degrees)

    if rotate_about_center and abs(degrees) >= 1e-9:
        if rotation_center is None:
            center = np.asarray(transformed.get_center(), dtype=np.float64)
        else:
            center = np.asarray(rotation_center, dtype=np.float64)
        to_origin = np.eye(4, dtype=np.float64)
        from_origin = np.eye(4, dtype=np.float64)
        to_origin[:3, 3] = -center
        from_origin[:3, 3] = center
        transform = from_origin @ transform @ to_origin

    for extra_axis, extra_degrees in (
        ("x", extra_rotate_x),
        ("y", extra_rotate_y),
        ("z", extra_rotate_z),
    ):
        if abs(extra_degrees) >= 1e-9:
            extra_axis_vec = _axis_vector_for_name(extra_axis, points)
            transform = _rotation_matrix_from_vector(extra_axis_vec, extra_degrees) @ transform

    transformed.transform(transform)
    if abs(translate_x) >= 1e-9 or abs(translate_y) >= 1e-9 or abs(translate_z) >= 1e-9:
        transformed.translate(np.array([translate_x, translate_y, translate_z], dtype=np.float64))
    return transformed


def _align_cloud_with_mode(source, target, align_mode, *, align_max_rotation_deg=20.0):
    if align_mode == "off":
        return copy.deepcopy(source), "alignment disabled"
    if align_mode == "centroid":
        return align_cloud_centroids(source, target)
    if align_mode == "centroid_icp":
        return align_cloud_centroid_icp(source, target)
    if align_mode == "centroid_icp_overlap":
        return align_cloud_centroid_icp_overlap(source, target)
    if align_mode == "centroid_icp_overlap_capped":
        return align_cloud_centroid_icp_overlap_capped(
            source,
            target,
            max_rotation_deg=align_max_rotation_deg,
        )
    if align_mode == "centroid_overlap_translate":
        return align_cloud_centroid_overlap_translate(source, target)
    return try_align_cloud_to_base(source, target)


def _canonicalize_axis_vector(axis_vec):
    import numpy as np

    vec = np.asarray(axis_vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise RuntimeError("axis vector is degenerate")
    vec = vec / norm
    for component in vec.tolist():
        if abs(component) > 1e-12:
            if component < 0.0:
                vec = -vec
            break
    return vec


def _generate_halfturn_axis_candidates(count):
    import math

    import numpy as np

    count = max(1, int(count))
    full_count = max(count * 2, 2)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    axes = []
    seen = set()
    for idx in range(full_count):
        z = 1.0 - (2.0 * (idx + 0.5) / full_count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * idx
        vec = np.array(
            [radius * math.cos(theta), radius * math.sin(theta), z],
            dtype=np.float64,
        )
        if vec[2] < 0.0:
            vec = -vec
        vec = _canonicalize_axis_vector(vec)
        key = tuple(np.round(vec, 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        axes.append(vec)
        if len(axes) >= count:
            break
    return axes


def _principal_frame(points):
    import numpy as np

    axes = _principal_axes(points)
    frame = np.column_stack(axes)
    for idx in range(3):
        vec = frame[:, idx]
        for component in vec.tolist():
            if abs(component) > 1e-12:
                if component < 0.0:
                    frame[:, idx] = -vec
                break
    if float(np.linalg.det(frame)) < 0.0:
        frame[:, 2] = -frame[:, 2]
    return frame


def _halfturn_frame_candidates():
    import itertools

    import numpy as np

    candidates = []
    for perm in itertools.permutations((0, 1, 2)):
        perm_matrix = np.zeros((3, 3), dtype=np.float64)
        for row, col in enumerate(perm):
            perm_matrix[row, col] = 1.0
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            sign_matrix = np.diag(np.asarray(signs, dtype=np.float64))
            frame_matrix = sign_matrix @ perm_matrix
            det = round(float(np.linalg.det(frame_matrix)))
            if det != 1:
                continue
            if abs(float(np.trace(frame_matrix)) + 1.0) > 1e-9:
                continue
            label = (
                "perm="
                f"{perm[0]}{perm[1]}{perm[2]} "
                f"signs=({int(signs[0]):+d},{int(signs[1]):+d},{int(signs[2]):+d})"
            )
            candidates.append((frame_matrix, label))
    return candidates


def _alignment_score(source, target):
    import math

    import numpy as np

    if len(source.points) < 20 or len(target.points) < 20:
        return float("-inf"), "too few points for scoring"

    src_extent = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    voxel = max(diag * 0.01, 1e-4)

    src_down = source.voxel_down_sample(voxel)
    tgt_down = target.voxel_down_sample(voxel)
    if len(src_down.points) < 20 or len(tgt_down.points) < 20:
        src_down = source
        tgt_down = target

    d_src = np.asarray(src_down.compute_point_cloud_distance(tgt_down), dtype=np.float64)
    d_tgt = np.asarray(tgt_down.compute_point_cloud_distance(src_down), dtype=np.float64)
    d_all = np.concatenate(
        [
            d_src[np.isfinite(d_src)],
            d_tgt[np.isfinite(d_tgt)],
        ]
    )
    if len(d_all) == 0:
        return float("-inf"), "empty distance arrays"

    threshold = max(voxel * 2.5, 1e-4)
    overlap = 0.5 * (
        float(np.mean(d_src[np.isfinite(d_src)] <= threshold))
        + float(np.mean(d_tgt[np.isfinite(d_tgt)] <= threshold))
    )
    clipped = np.clip(d_all, 0.0, threshold)
    rmse = math.sqrt(float(np.mean(clipped * clipped)))
    median = float(np.median(d_all))
    score = (overlap * 2.0) - (rmse / threshold)
    return score, f"score={score:.4f}, overlap={overlap:.4f}, median={median:.6f}, rmse={rmse:.6f}"


def _apply_rotation_matrix_about_center(cloud, rotation3x3):
    import numpy as np

    transformed = copy.deepcopy(cloud)
    center = np.asarray(transformed.get_center(), dtype=np.float64)
    return _apply_rotation_matrix_about_point(transformed, rotation3x3, center)


def _apply_rotation_matrix_about_point(cloud, rotation3x3, center):
    import numpy as np

    transformed = copy.deepcopy(cloud)
    center = np.asarray(center, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation3x3, dtype=np.float64)
    to_origin = np.eye(4, dtype=np.float64)
    from_origin = np.eye(4, dtype=np.float64)
    to_origin[:3, 3] = -center
    from_origin[:3, 3] = center
    transformed.transform(from_origin @ transform @ to_origin)
    return transformed


def _apply_primary_frame_flip_and_align(
    source,
    target,
    *,
    frame_axis,
    degrees,
    extra_rotate_x=0.0,
    extra_rotate_y=0.0,
    extra_rotate_z=0.0,
    translate_x=0.0,
    translate_y=0.0,
    translate_z=0.0,
    align_mode="auto",
    align_max_rotation_deg=20.0,
):
    import numpy as np

    target_points = np.asarray(target.points, dtype=np.float64)
    if len(target_points) < 3:
        raise RuntimeError("too few target points for primary-frame transform")

    frame_center = target_points.mean(axis=0)
    frame = _principal_frame(target_points)
    to_frame = frame.T
    from_frame = frame
    axis_vec_map = {
        "primary_frame_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "primary_frame_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "primary_frame_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if frame_axis not in axis_vec_map:
        raise ValueError(f"unsupported primary-frame axis: {frame_axis}")

    target_frame = _apply_rotation_matrix_about_point(target, to_frame, frame_center)
    source_frame = _apply_rotation_matrix_about_point(source, to_frame, frame_center)
    source_frame = _apply_pre_transform_about_axis_vector(
        source_frame,
        axis_vec=axis_vec_map[frame_axis],
        degrees=degrees,
        extra_rotate_x=extra_rotate_x,
        extra_rotate_y=extra_rotate_y,
        extra_rotate_z=extra_rotate_z,
        translate_x=translate_x,
        translate_y=translate_y,
        translate_z=translate_z,
        rotate_about_center=True,
        rotation_center=frame_center,
    )
    pre_world = _apply_rotation_matrix_about_point(source_frame, from_frame, frame_center)

    if align_mode == "off":
        return pre_world, pre_world, (
            f"primary-frame flip using {frame_axis} about target PCA frame center "
            f"({frame_center[0]:.6f}, {frame_center[1]:.6f}, {frame_center[2]:.6f}); "
            "alignment disabled"
        )

    aligned_frame, align_detail = _align_cloud_with_mode(
        source_frame,
        target_frame,
        align_mode,
        align_max_rotation_deg=align_max_rotation_deg,
    )
    if aligned_frame is None:
        return pre_world, None, (
            f"primary-frame flip using {frame_axis} about target PCA frame center "
            f"({frame_center[0]:.6f}, {frame_center[1]:.6f}, {frame_center[2]:.6f}); "
            f"alignment failed: {align_detail}"
        )

    aligned_world = _apply_rotation_matrix_about_point(aligned_frame, from_frame, frame_center)
    return pre_world, aligned_world, (
        f"primary-frame flip using {frame_axis} about target PCA frame center "
        f"({frame_center[0]:.6f}, {frame_center[1]:.6f}, {frame_center[2]:.6f}); "
        f"{align_detail}"
    )


def _search_best_halfturn_axis(
    source,
    target,
    *,
    degrees,
    extra_rotate_x=0.0,
    extra_rotate_y=0.0,
    extra_rotate_z=0.0,
    translate_x=0.0,
    translate_y=0.0,
    translate_z=0.0,
    align_mode="auto",
    axis_count=48,
    align_max_rotation_deg=20.0,
):
    import numpy as np

    if len(source.points) < 50 or len(target.points) < 50:
        raise RuntimeError("too few points for auto half-turn search")

    src_extent = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    search_voxel = max(diag * 0.02, 1e-4)
    source_search = source.voxel_down_sample(search_voxel)
    target_search = target.voxel_down_sample(search_voxel)
    if len(source_search.points) < 50:
        source_search = source
    if len(target_search.points) < 50:
        target_search = target

    source_points = np.asarray(source_search.points, dtype=np.float64)
    candidates = []
    seen = set()

    def add_candidate(axis_vec, label):
        vec = _canonicalize_axis_vector(axis_vec)
        key = tuple(np.round(vec, 6).tolist())
        if key in seen:
            return
        seen.add(key)
        candidates.append((vec, label))

    add_candidate(np.array([1.0, 0.0, 0.0], dtype=np.float64), "world x")
    add_candidate(np.array([0.0, 1.0, 0.0], dtype=np.float64), "world y")
    add_candidate(np.array([0.0, 0.0, 1.0], dtype=np.float64), "world z")
    try:
        p_axes = _principal_axes(source_points)
        add_candidate(p_axes[0], "principal major")
        add_candidate(p_axes[1], "principal minor1")
        add_candidate(p_axes[2], "principal minor2")
    except Exception:
        pass
    for idx, axis_vec in enumerate(_generate_halfturn_axis_candidates(axis_count), start=1):
        add_candidate(axis_vec, f"search axis {idx}")

    best = None
    for axis_vec, label in candidates:
        candidate = _apply_pre_transform_about_axis_vector(
            source_search,
            axis_vec=axis_vec,
            degrees=degrees,
            extra_rotate_x=extra_rotate_x,
            extra_rotate_y=extra_rotate_y,
            extra_rotate_z=extra_rotate_z,
            translate_x=translate_x,
            translate_y=translate_y,
            translate_z=translate_z,
            rotate_about_center=True,
        )
        aligned, align_detail = _align_cloud_with_mode(
            candidate,
            target_search,
            align_mode,
            align_max_rotation_deg=align_max_rotation_deg,
        )
        if aligned is None:
            continue
        score, score_detail = _alignment_score(aligned, target_search)
        if best is None or score > best["score"]:
            best = {
                "axis_vec": axis_vec,
                "label": label,
                "score": score,
                "align_detail": align_detail,
                "score_detail": score_detail,
            }

    if best is None:
        raise RuntimeError("auto half-turn search failed to find a valid candidate")

    transformed = _apply_pre_transform_about_axis_vector(
        source,
        axis_vec=best["axis_vec"],
        degrees=degrees,
        extra_rotate_x=extra_rotate_x,
        extra_rotate_y=extra_rotate_y,
        extra_rotate_z=extra_rotate_z,
        translate_x=translate_x,
        translate_y=translate_y,
        translate_z=translate_z,
        rotate_about_center=True,
    )
    axis_fmt = ", ".join(f"{float(v):.6f}" for v in best["axis_vec"])
    detail = (
        f"auto half-turn picked {best['label']} axis=({axis_fmt}) "
        f"using {align_mode} search; {best['align_detail']}; {best['score_detail']}"
    )
    return transformed, detail


def _search_best_pca_halfturn_transform(
    source,
    target,
    *,
    extra_rotate_x=0.0,
    extra_rotate_y=0.0,
    extra_rotate_z=0.0,
    translate_x=0.0,
    translate_y=0.0,
    translate_z=0.0,
    align_mode="auto",
    align_max_rotation_deg=20.0,
):
    import numpy as np

    if len(source.points) < 50 or len(target.points) < 50:
        raise RuntimeError("too few points for PCA half-turn search")

    src_extent = np.asarray(source.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    tgt_extent = np.asarray(target.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
    diag = float(max(np.linalg.norm(src_extent), np.linalg.norm(tgt_extent), 1e-6))
    search_voxel = max(diag * 0.02, 1e-4)
    source_search = source.voxel_down_sample(search_voxel)
    target_search = target.voxel_down_sample(search_voxel)
    if len(source_search.points) < 50:
        source_search = source
    if len(target_search.points) < 50:
        target_search = target

    source_basis = _principal_frame(np.asarray(source_search.points, dtype=np.float64))
    target_basis = _principal_frame(np.asarray(target_search.points, dtype=np.float64))

    best = None
    for frame_matrix, label in _halfturn_frame_candidates():
        rotation3x3 = target_basis @ frame_matrix @ source_basis.T
        candidate = _apply_rotation_matrix_about_center(source_search, rotation3x3)
        candidate = _apply_pre_transform_about_axis_vector(
            candidate,
            axis_vec=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            degrees=0.0,
            extra_rotate_x=extra_rotate_x,
            extra_rotate_y=extra_rotate_y,
            extra_rotate_z=extra_rotate_z,
            translate_x=translate_x,
            translate_y=translate_y,
            translate_z=translate_z,
            rotate_about_center=False,
        )
        aligned, align_detail = _align_cloud_with_mode(
            candidate,
            target_search,
            align_mode,
            align_max_rotation_deg=align_max_rotation_deg,
        )
        if aligned is None:
            continue
        score, score_detail = _alignment_score(aligned, target_search)
        if best is None or score > best["score"]:
            best = {
                "rotation3x3": rotation3x3,
                "label": label,
                "score": score,
                "align_detail": align_detail,
                "score_detail": score_detail,
            }

    if best is None:
        raise RuntimeError("PCA half-turn search failed to find a valid candidate")

    transformed = _apply_rotation_matrix_about_center(source, best["rotation3x3"])
    transformed = _apply_pre_transform_about_axis_vector(
        transformed,
        axis_vec=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        degrees=0.0,
        extra_rotate_x=extra_rotate_x,
        extra_rotate_y=extra_rotate_y,
        extra_rotate_z=extra_rotate_z,
        translate_x=translate_x,
        translate_y=translate_y,
        translate_z=translate_z,
        rotate_about_center=False,
    )
    matrix_fmt = "; ".join(
        ", ".join(f"{float(v):.6f}" for v in row.tolist())
        for row in np.asarray(best["rotation3x3"], dtype=np.float64)
    )
    detail = (
        f"PCA half-turn picked {best['label']} "
        f"R=[{matrix_fmt}] using {align_mode} search; "
        f"{best['align_detail']}; {best['score_detail']}"
    )
    return transformed, detail


def _normalize_pre_transforms(
    pre_transforms: list[dict[str, float | str] | None] | None,
    count: int,
) -> list[dict[str, float | str] | None]:
    normalized = list(pre_transforms or [])
    if len(normalized) < count:
        normalized.extend([None] * (count - len(normalized)))
    elif len(normalized) > count:
        normalized = normalized[:count]

    result: list[dict[str, float | str] | None] = []
    for spec in normalized:
        if spec is None:
            result.append(None)
            continue

        axis = str(spec.get("axis", "y")).strip().lower()
        if axis not in {
            "x",
            "y",
            "z",
            "principal",
            "principal_major",
            "principal_minor1",
            "principal_minor2",
            "primary_principal",
            "primary_principal_major",
            "primary_principal_minor1",
            "primary_principal_minor2",
            "primary_frame_x",
            "primary_frame_y",
            "primary_frame_z",
            "vector",
            "auto_halfturn",
            "pca_halfturn_auto",
        }:
            raise ValueError(f"invalid merge pre-rotation axis: {axis}")
        degrees = float(spec.get("degrees", 0.0))
        extra_rotate_x = float(spec.get("extra_rotate_x", 0.0))
        extra_rotate_y = float(spec.get("extra_rotate_y", 0.0))
        extra_rotate_z = float(spec.get("extra_rotate_z", 0.0))
        translate_x = float(spec.get("translate_x", 0.0))
        translate_y = float(spec.get("translate_y", 0.0))
        translate_z = float(spec.get("translate_z", 0.0))
        axis_vector_x = float(spec.get("axis_vector_x", 0.0))
        axis_vector_y = float(spec.get("axis_vector_y", 0.0))
        axis_vector_z = float(spec.get("axis_vector_z", 0.0))
        auto_halfturn_axes = max(1, int(spec.get("auto_halfturn_axes", 48)))
        align_max_rotation_deg = max(0.0, float(spec.get("align_max_rotation_deg", 20.0)))
        align_mode = str(spec.get("align_mode", "auto")).strip().lower() or "auto"
        if align_mode not in {
            "auto",
            "off",
            "centroid",
            "centroid_icp",
            "centroid_icp_overlap",
            "centroid_icp_overlap_capped",
            "centroid_overlap_translate",
        }:
            raise ValueError(f"invalid merge alignment mode: {align_mode}")
        if axis == "vector":
            axis_norm = (axis_vector_x * axis_vector_x) + (axis_vector_y * axis_vector_y) + (axis_vector_z * axis_vector_z)
            if axis_norm <= 1e-18:
                raise ValueError("vector pre-rotation axis requires a non-zero axis_vector")

        if (
            abs(degrees) < 1e-9
            and abs(extra_rotate_x) < 1e-9
            and abs(extra_rotate_y) < 1e-9
            and abs(extra_rotate_z) < 1e-9
            and abs(translate_x) < 1e-9
            and abs(translate_y) < 1e-9
            and abs(translate_z) < 1e-9
            and align_mode == "auto"
        ):
            result.append(None)
        else:
            result.append(
                {
                    "axis": axis,
                    "degrees": degrees,
                    "extra_rotate_x": extra_rotate_x,
                    "extra_rotate_y": extra_rotate_y,
                    "extra_rotate_z": extra_rotate_z,
                    "translate_x": translate_x,
                    "translate_y": translate_y,
                    "translate_z": translate_z,
                    "axis_vector_x": axis_vector_x,
                    "axis_vector_y": axis_vector_y,
                    "axis_vector_z": axis_vector_z,
                    "auto_halfturn_axes": auto_halfturn_axes,
                    "align_max_rotation_deg": align_max_rotation_deg,
                    "align_mode": align_mode,
                }
            )
    return result


def _apply_pre_transform(
    cloud,
    *,
    axis: str,
    degrees: float,
    extra_rotate_x: float = 0.0,
    extra_rotate_y: float = 0.0,
    extra_rotate_z: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
    translate_z: float = 0.0,
    reference_points=None,
    axis_vector=None,
):
    import numpy as np
    points = np.asarray(cloud.points, dtype=np.float64)
    axis_vec = _axis_vector_for_name(
        axis,
        points,
        reference_points=reference_points,
        axis_vector=axis_vector,
    )
    return _apply_pre_transform_about_axis_vector(
        cloud,
        axis_vec=axis_vec,
        degrees=degrees,
        extra_rotate_x=extra_rotate_x,
        extra_rotate_y=extra_rotate_y,
        extra_rotate_z=extra_rotate_z,
        translate_x=translate_x,
        translate_y=translate_y,
        translate_z=translate_z,
        rotate_about_center=axis in {
            "principal",
            "principal_major",
            "principal_minor1",
            "principal_minor2",
            "primary_principal",
            "primary_principal_major",
            "primary_principal_minor1",
            "primary_principal_minor2",
            "vector",
        },
    )


def _describe_pre_transform(spec: dict[str, float | str] | None) -> str:
    if spec is None:
        return "identity"

    parts: list[str] = []
    degrees = float(spec.get("degrees", 0.0))
    axis = str(spec.get("axis", "y"))
    if abs(degrees) >= 1e-9:
        if axis in {"principal", "principal_major"}:
            parts.append(f"rotate {degrees:g} deg around principal major axis about cloud center")
        elif axis == "principal_minor1":
            parts.append(f"rotate {degrees:g} deg around principal minor1 axis about cloud center")
        elif axis == "principal_minor2":
            parts.append(f"rotate {degrees:g} deg around principal minor2 axis about cloud center")
        elif axis in {"primary_principal", "primary_principal_major"}:
            parts.append(f"rotate {degrees:g} deg around primary principal major axis about cloud center")
        elif axis == "primary_principal_minor1":
            parts.append(f"rotate {degrees:g} deg around primary principal minor1 axis about cloud center")
        elif axis == "primary_principal_minor2":
            parts.append(f"rotate {degrees:g} deg around primary principal minor2 axis about cloud center")
        elif axis in {"primary_frame_x", "primary_frame_y", "primary_frame_z"}:
            parts.append(f"rotate {degrees:g} deg around {axis} in primary PCA frame")
        elif axis == "vector":
            axis_vector_x = float(spec.get("axis_vector_x", 0.0))
            axis_vector_y = float(spec.get("axis_vector_y", 0.0))
            axis_vector_z = float(spec.get("axis_vector_z", 0.0))
            parts.append(
                f"rotate {degrees:g} deg around vector ({axis_vector_x:g}, {axis_vector_y:g}, {axis_vector_z:g}) about cloud center"
            )
        elif axis == "auto_halfturn":
            parts.append(f"auto-search {degrees:g} deg half-turn axis about cloud center")
        elif axis == "pca_halfturn_auto":
            parts.append("auto-search PCA-frame half-turn about cloud center")
        else:
            parts.append(f"rotate {degrees:g} deg around {axis}-axis")

    for extra_axis in ("x", "y", "z"):
        extra_degrees = float(spec.get(f"extra_rotate_{extra_axis}", 0.0))
        if abs(extra_degrees) >= 1e-9:
            parts.append(f"extra rotate {extra_degrees:g} deg around {extra_axis}-axis")

    translate_x = float(spec.get("translate_x", 0.0))
    translate_y = float(spec.get("translate_y", 0.0))
    translate_z = float(spec.get("translate_z", 0.0))
    if abs(translate_x) >= 1e-9 or abs(translate_y) >= 1e-9 or abs(translate_z) >= 1e-9:
        parts.append(f"translate ({translate_x:g}, {translate_y:g}, {translate_z:g})")

    align_mode = str(spec.get("align_mode", "auto")).strip().lower() or "auto"
    if align_mode != "auto":
        parts.append(f"align={align_mode}")

    return ", ".join(parts) if parts else "identity"


def merge_component_clouds(
    part_clouds: list[Path],
    merged_out: Path,
    pre_transforms: list[dict[str, float | str] | None] | None = None,
) -> None:
    if not part_clouds:
        raise ValueError("no clouds provided for merge")

    transforms = _normalize_pre_transforms(pre_transforms, len(part_clouds))
    if len(part_clouds) == 1 and transforms[0] is None:
        if part_clouds[0] != merged_out:
            shutil.copyfile(part_clouds[0], merged_out)
        return

    try:
        import open3d as o3d
    except Exception as exc:
        print(
            "warning: Open3D is unavailable for merge alignment "
            f"({exc}); using first cloud only",
            file=sys.stderr,
        )
        shutil.copyfile(part_clouds[0], merged_out)
        return

    print("[step] merge dense components")
    merged = o3d.io.read_point_cloud(str(part_clouds[0]))
    if len(merged.points) == 0:
        raise RuntimeError(f"base component cloud is empty: {part_clouds[0]}")

    base_transform = transforms[0]
    if base_transform is not None:
        merged = _apply_pre_transform(
            merged,
            axis=str(base_transform["axis"]),
            degrees=float(base_transform["degrees"]),
            extra_rotate_x=float(base_transform.get("extra_rotate_x", 0.0)),
            extra_rotate_y=float(base_transform.get("extra_rotate_y", 0.0)),
            extra_rotate_z=float(base_transform.get("extra_rotate_z", 0.0)),
            translate_x=float(base_transform.get("translate_x", 0.0)),
            translate_y=float(base_transform.get("translate_y", 0.0)),
            translate_z=float(base_transform.get("translate_z", 0.0)),
        )
        print(f"merge component 0: {_describe_pre_transform(base_transform)}")

    for idx, cloud_path in enumerate(part_clouds[1:], start=1):
        pcd = o3d.io.read_point_cloud(str(cloud_path))
        if len(pcd.points) == 0:
            print(f"warning: component {idx} cloud is empty: {cloud_path}", file=sys.stderr)
            continue

        pre = transforms[idx]
        align_mode = "auto"
        if pre is not None:
            align_mode = str(pre.get("align_mode", "auto")).strip().lower() or "auto"
            axis = str(pre["axis"])
            if axis == "auto_halfturn":
                pcd, search_detail = _search_best_halfturn_axis(
                    pcd,
                    merged,
                    degrees=float(pre["degrees"]),
                    extra_rotate_x=float(pre.get("extra_rotate_x", 0.0)),
                    extra_rotate_y=float(pre.get("extra_rotate_y", 0.0)),
                    extra_rotate_z=float(pre.get("extra_rotate_z", 0.0)),
                    translate_x=float(pre.get("translate_x", 0.0)),
                    translate_y=float(pre.get("translate_y", 0.0)),
                    translate_z=float(pre.get("translate_z", 0.0)),
                    align_mode=align_mode,
                    axis_count=int(pre.get("auto_halfturn_axes", 48)),
                    align_max_rotation_deg=float(pre.get("align_max_rotation_deg", 20.0)),
                )
                print(f"merge component {idx}: {_describe_pre_transform(pre)}")
                print(f"merge component {idx}: {search_detail}")
            elif axis == "pca_halfturn_auto":
                pcd, search_detail = _search_best_pca_halfturn_transform(
                    pcd,
                    merged,
                    extra_rotate_x=float(pre.get("extra_rotate_x", 0.0)),
                    extra_rotate_y=float(pre.get("extra_rotate_y", 0.0)),
                    extra_rotate_z=float(pre.get("extra_rotate_z", 0.0)),
                    translate_x=float(pre.get("translate_x", 0.0)),
                    translate_y=float(pre.get("translate_y", 0.0)),
                    translate_z=float(pre.get("translate_z", 0.0)),
                    align_mode=align_mode,
                    align_max_rotation_deg=float(pre.get("align_max_rotation_deg", 20.0)),
                )
                print(f"merge component {idx}: {_describe_pre_transform(pre)}")
                print(f"merge component {idx}: {search_detail}")
            elif axis in {"primary_frame_x", "primary_frame_y", "primary_frame_z"}:
                pre_world, aligned_world, frame_detail = _apply_primary_frame_flip_and_align(
                    pcd,
                    merged,
                    frame_axis=axis,
                    degrees=float(pre["degrees"]),
                    extra_rotate_x=float(pre.get("extra_rotate_x", 0.0)),
                    extra_rotate_y=float(pre.get("extra_rotate_y", 0.0)),
                    extra_rotate_z=float(pre.get("extra_rotate_z", 0.0)),
                    translate_x=float(pre.get("translate_x", 0.0)),
                    translate_y=float(pre.get("translate_y", 0.0)),
                    translate_z=float(pre.get("translate_z", 0.0)),
                    align_mode=align_mode,
                    align_max_rotation_deg=float(pre.get("align_max_rotation_deg", 20.0)),
                )
                print(f"merge component {idx}: {_describe_pre_transform(pre)}")
                print(f"merge component {idx}: {frame_detail}")
                if aligned_world is not None:
                    merged += aligned_world
                else:
                    print(
                        "warning: primary-frame alignment failed for component "
                        f"{idx}; appending pre-transformed cloud",
                        file=sys.stderr,
                    )
                    merged += pre_world
                continue
            else:
                reference_points = None
                if axis in {
                    "primary_principal",
                    "primary_principal_major",
                    "primary_principal_minor1",
                    "primary_principal_minor2",
                }:
                    import numpy as np

                    reference_points = np.asarray(merged.points, dtype=np.float64)
                pcd = _apply_pre_transform(
                    pcd,
                    axis=axis,
                    degrees=float(pre["degrees"]),
                    extra_rotate_x=float(pre.get("extra_rotate_x", 0.0)),
                    extra_rotate_y=float(pre.get("extra_rotate_y", 0.0)),
                    extra_rotate_z=float(pre.get("extra_rotate_z", 0.0)),
                    translate_x=float(pre.get("translate_x", 0.0)),
                    translate_y=float(pre.get("translate_y", 0.0)),
                    translate_z=float(pre.get("translate_z", 0.0)),
                    reference_points=reference_points,
                    axis_vector=(
                        float(pre.get("axis_vector_x", 0.0)),
                        float(pre.get("axis_vector_y", 0.0)),
                        float(pre.get("axis_vector_z", 0.0)),
                    ),
                )
                print(f"merge component {idx}: {_describe_pre_transform(pre)}")

        if align_mode == "off":
            print(f"merge component {idx}: alignment disabled, appending pre-transformed cloud")
            merged += pcd
            continue

        if align_mode == "centroid":
            aligned, detail = align_cloud_centroids(pcd, merged)
            if aligned is not None:
                print(f"merge component {idx}: {detail}")
                merged += aligned
            else:
                print(
                    "warning: centroid alignment failed for component "
                    f"{idx} ({detail}); appending unaligned points",
                    file=sys.stderr,
                )
                merged += pcd
            continue

        if align_mode == "centroid_icp":
            aligned, detail = align_cloud_centroid_icp(pcd, merged)
            if aligned is not None:
                print(f"merge component {idx}: {detail}")
                merged += aligned
            else:
                print(
                    "warning: centroid+icp alignment failed for component "
                    f"{idx} ({detail}); appending unaligned points",
                    file=sys.stderr,
                )
                merged += pcd
            continue

        if align_mode == "centroid_icp_overlap":
            aligned, detail = align_cloud_centroid_icp_overlap(pcd, merged)
            if aligned is not None:
                print(f"merge component {idx}: {detail}")
                merged += aligned
            else:
                print(
                    "warning: centroid+icp+overlap alignment failed for component "
                    f"{idx} ({detail}); appending unaligned points",
                    file=sys.stderr,
                )
                merged += pcd
            continue

        if align_mode == "centroid_icp_overlap_capped":
            aligned, detail = align_cloud_centroid_icp_overlap_capped(
                pcd,
                merged,
                max_rotation_deg=float(pre.get("align_max_rotation_deg", 20.0)) if pre is not None else 20.0,
            )
            if aligned is not None:
                print(f"merge component {idx}: {detail}")
                merged += aligned
            else:
                print(
                    "warning: centroid+icp+overlap-capped alignment failed for component "
                    f"{idx} ({detail}); appending unaligned points",
                    file=sys.stderr,
                )
                merged += pcd
            continue

        if align_mode == "centroid_overlap_translate":
            aligned, detail = align_cloud_centroid_overlap_translate(pcd, merged)
            if aligned is not None:
                print(f"merge component {idx}: {detail}")
                merged += aligned
            else:
                print(
                    "warning: centroid+overlap-translate alignment failed for component "
                    f"{idx} ({detail}); appending unaligned points",
                    file=sys.stderr,
                )
                merged += pcd
            continue

        aligned, detail = try_align_cloud_to_base(pcd, merged)
        if aligned is not None:
            print(f"merge component {idx}: {detail}")
            merged += aligned
        else:
            print(
                "warning: merge alignment failed for component "
                f"{idx} ({detail}); appending unaligned points",
                file=sys.stderr,
            )
            merged += pcd

    merge_voxel = float(os.environ.get("FIPMESH_COLMAP_MERGE_VOXEL", "0"))
    if merge_voxel > 0:
        merged = merged.voxel_down_sample(merge_voxel)
        print(f"merge voxel downsample: {merge_voxel:g}")

    ok = o3d.io.write_point_cloud(str(merged_out), merged, write_ascii=False)
    if not ok:
        raise RuntimeError(f"failed to write merged cloud: {merged_out}")

    merged_snap = make_snapshot_dir(merged_out) / f"{merged_out.stem}_stitched_combination.ply"
    o3d.io.write_point_cloud(str(merged_snap), merged, write_ascii=False)
    print(f"snapshot saved: {merged_snap}")

def run_pipeline_for_image_set(
    *,
    args: argparse.Namespace,
    images: Path,
    workspace: Path,
    dense_cloud_out: Path,
    clean_workspace: bool,
    set_label: str,
    use_gpu: bool,
    gpu_index: str,
    single_camera: bool,
    extract_threads: int,
    match_threads: int,
    mapper_threads: int,
    fusion_threads: int,
    input_image_scale: float,
    input_image_stride: int,
    sift_max_image_size: int,
    sift_max_num_features: int,
    sift_peak_threshold: float,
    sift_edge_threshold: float,
    sift_domain_size_pooling: bool,
    sift_estimate_affine_shape: bool,
    patch_max_image_size: int,
    fusion_max_image_size: int,
    patch_window_radius: int,
    patch_num_samples: int,
    patch_num_iterations: int,
    patch_filter_min_ncc: float,
    patch_filter_min_consistent: int,
    patch_cache_size: int,
    fusion_min_num_pixels: int,
    fusion_max_reproj_error: float,
    fusion_max_depth_error: float,
    fusion_max_normal_error: float,
    fusion_cache_size: int,
    fusion_use_cache: bool,
    match_guided: bool,
    match_max_num_matches: int,
    option_style: str,
    camera_model: str = "",
    focal_length_px: float = 0.0,
    single_camera_per_folder: bool = False,
) -> int:
    if clean_workspace and workspace.exists():
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)
    dense_cloud_out.parent.mkdir(parents=True, exist_ok=True)

    db_path = workspace / "database.db"
    sparse_root = workspace / "sparse"
    dense_root = workspace / "dense"
    images_for_colmap = images
    if input_image_scale < 1.0 or input_image_stride > 1:
        images_for_colmap = _prepare_input_images(
            images,
            workspace / "_input_images",
            scale=input_image_scale,
            image_stride=input_image_stride,
        )
    sparse_root.mkdir(parents=True, exist_ok=True)
    dense_root.mkdir(parents=True, exist_ok=True)

    if option_style == "modern":
        fx_use_gpu = "--FeatureExtraction.use_gpu"
        fx_threads = "--FeatureExtraction.num_threads"
        fx_gpu_index = "--FeatureExtraction.gpu_index"
        fx_max_img = "--FeatureExtraction.max_image_size"
        fm_use_gpu = "--FeatureMatching.use_gpu"
        fm_threads = "--FeatureMatching.num_threads"
        fm_gpu_index = "--FeatureMatching.gpu_index"
    else:
        fx_use_gpu = "--SiftExtraction.use_gpu"
        fx_threads = "--SiftExtraction.num_threads"
        fx_gpu_index = "--SiftExtraction.gpu_index"
        fx_max_img = "--SiftExtraction.max_image_size"
        fm_use_gpu = "--SiftMatching.use_gpu"
        fm_threads = "--SiftMatching.num_threads"
        fm_gpu_index = "--SiftMatching.gpu_index"

    print(f"[set] {set_label} images={images}")
    if images_for_colmap != images:
        print(
            f"[set] {set_label} prepared_images={images_for_colmap} "
            f"scale={input_image_scale:g} stride={input_image_stride}"
        )
    print(f"[set] {set_label} workspace={workspace}")
    print("[step] colmap feature_extractor")
    print(
        "settings: "
        f"quality={args.quality} "
        f"input_image_scale={input_image_scale:g} "
        f"input_image_stride={input_image_stride} "
        f"use_gpu={int(use_gpu)} "
        f"gpu_index={gpu_index} "
        f"extract_threads={extract_threads} "
        f"match_threads={match_threads} "
        f"mapper_threads={mapper_threads} "
        f"fusion_threads={fusion_threads} "
        f"sift_max_image_size={sift_max_image_size} "
        f"sift_max_num_features={sift_max_num_features} "
        f"sift_peak_threshold={sift_peak_threshold:g} "
        f"sift_edge_threshold={sift_edge_threshold:g} "
        f"sift_domain_size_pooling={int(sift_domain_size_pooling)} "
        f"sift_estimate_affine_shape={int(sift_estimate_affine_shape)} "
        f"patch_max_image_size={patch_max_image_size} "
        f"fusion_max_image_size={fusion_max_image_size} "
        f"patch_cache_size={patch_cache_size} "
        f"fusion_cache_size={fusion_cache_size} "
        f"fusion_use_cache={int(fusion_use_cache)} "
        f"match_guided={int(match_guided)} "
        f"match_max_num_matches={match_max_num_matches} "
        f"option_style={option_style} "
        f"camera_model={camera_model!r} focal_length_px={focal_length_px:g} "
        f"single_camera_per_folder={int(single_camera_per_folder)}"
    )

    base_fx_args = [
        fx_use_gpu,
        "1" if use_gpu else "0",
        fx_gpu_index,
        gpu_index,
        fx_threads,
        str(extract_threads),
        fx_max_img,
        str(sift_max_image_size),
        "--SiftExtraction.max_num_features",
        str(sift_max_num_features),
        "--SiftExtraction.peak_threshold",
        f"{sift_peak_threshold:g}",
        "--SiftExtraction.edge_threshold",
        f"{sift_edge_threshold:g}",
        "--SiftExtraction.domain_size_pooling",
        "1" if sift_domain_size_pooling else "0",
        "--SiftExtraction.estimate_affine_shape",
        "1" if sift_estimate_affine_shape else "0",
    ]

    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".ppm", ".pgm", ".pnm"}

    def _camera_params_args(folder: Path) -> list[str]:
        if not camera_model:
            return []
        extra = ["--ImageReader.camera_model", camera_model]
        effective_focal = focal_length_px
        if effective_focal <= 0:
            exif_focal = _read_exif_focal_length_px(folder)
            if exif_focal and exif_focal > 0:
                print(f"[set] {set_label} EXIF focal length ({folder}): {exif_focal:.2f}px")
                effective_focal = exif_focal
        if effective_focal > 0:
            img_w, img_h = _get_first_image_size(folder)
            if img_w > 0 and img_h > 0:
                params = _camera_params_str(camera_model, effective_focal, img_w, img_h)
                if params:
                    extra += ["--ImageReader.camera_params", params]
                    print(f"[set] {set_label} camera_params ({folder}): {params}")
        return extra

    def _run_feature_extractor(target: Path, image_list: list[str] | None, label: str) -> None:
        fx_cmd = [
            args.colmap_bin,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_for_colmap),
            "--ImageReader.single_camera",
            "1" if (single_camera_per_folder or single_camera) else "0",
        ] + base_fx_args + _camera_params_args(target)
        if image_list is not None:
            list_path = workspace / f"_image_list_{label}.txt"
            list_path.write_text("\n".join(image_list) + "\n")
            fx_cmd += ["--image_list_path", str(list_path)]
        print(f"[step] colmap feature_extractor ({label})")
        run(fx_cmd)

    if single_camera_per_folder:
        # Run one feature_extractor pass per immediate subfolder, each scoped via
        # --image_list_path with single_camera=1. --ImageReader.single_camera_per_folder
        # does not reliably scope the "established camera" per folder on all COLMAP
        # builds: images from a second camera with different native dimensions get
        # rejected with CAMERA_SINGLE_DIM_ERROR and silently skipped (no features
        # extracted) instead of getting their own camera. Splitting the passes
        # guarantees each physical camera gets a fresh camera built only from its
        # own images.
        subfolders = sorted(p for p in images_for_colmap.iterdir() if p.is_dir())
        root_images = sorted(
            f.relative_to(images_for_colmap).as_posix()
            for f in images_for_colmap.iterdir()
            if f.is_file() and f.suffix.lower() in image_exts
        )
        groups: list[tuple[Path, list[str], str]] = []
        for folder in subfolders:
            rel_paths = sorted(
                f.relative_to(images_for_colmap).as_posix()
                for f in folder.rglob("*")
                if f.is_file() and f.suffix.lower() in image_exts
            )
            if rel_paths:
                label = folder.relative_to(images_for_colmap).as_posix().replace("/", "_")
                groups.append((folder, rel_paths, label))
        if root_images:
            groups.append((images_for_colmap, root_images, "root"))

        if not groups:
            raise RuntimeError(
                f"single_camera_per_folder enabled but no images found under {images_for_colmap}"
            )

        print(
            f"[set] {set_label} multi-camera mode: {len(groups)} camera "
            f"group(s) under {images_for_colmap}, each gets its own "
            f"single_camera=1 feature_extractor pass"
        )
        for folder, rel_paths, label in groups:
            print(f"[set] {set_label}   - {label}: {len(rel_paths)} images")
            _run_feature_extractor(folder, rel_paths, label)
    else:
        _run_feature_extractor(images_for_colmap, None, "all")
    matcher_cmd = f"{args.matcher}_matcher"
    print("[debug] probing COLMAP guided matching option...", file=sys.stderr)

    guided_option = first_supported_option(
        args.colmap_bin,
        matcher_cmd,
        "--GuidedMatching.enable",
        "--FeatureMatching.guided_matching",
        "--SiftMatching.guided_matching",
    )
    if guided_option is None:
        guided_option = (
            "--FeatureMatching.guided_matching"
            if option_style == "modern"
            else "--SiftMatching.guided_matching"
        )
        print(f"[debug] defaulting: {guided_option}", file=sys.stderr)
    else:
        print(f"[debug] supported: {guided_option}", file=sys.stderr)
    match_max_num_matches_option = first_supported_option(
        args.colmap_bin,
        matcher_cmd,
        "--FeatureMatching.max_num_matches",
        "--SiftMatching.max_num_matches",
    )
    if match_max_num_matches_option is None:
        match_max_num_matches_option = (
            "--FeatureMatching.max_num_matches"
            if option_style == "modern"
            else "--SiftMatching.max_num_matches"
        )
        print(
            f"[debug] defaulting max_num_matches option: {match_max_num_matches_option}",
            file=sys.stderr,
        )
    else:
        print(
            f"[debug] supported max_num_matches option: {match_max_num_matches_option}",
            file=sys.stderr,
        )
    print(f"[step] colmap {matcher_cmd}")
    matcher_args = [
        args.colmap_bin,
        matcher_cmd,
        "--database_path",
        str(db_path),
        fm_use_gpu,
        "1" if use_gpu else "0",
        fm_gpu_index,
        gpu_index,
        fm_threads,
        str(match_threads),
        guided_option,
        "1" if match_guided else "0",
        match_max_num_matches_option,
        str(match_max_num_matches),
    ]
    if args.matcher == "sequential":
        matcher_args.extend(["--SequentialMatching.loop_detection", "1"])
    run(matcher_args)

    print("[step] colmap mapper")
    run(
        [
            args.colmap_bin,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(images_for_colmap),
            "--output_path",
            str(sparse_root),
            "--Mapper.num_threads",
            str(mapper_threads),
        ]
    )

    model_dirs = list_sparse_models(sparse_root)
    print(f"[set] {set_label} sparse models found: {len(model_dirs)}")
    for idx, model_dir in enumerate(model_dirs):
        print(f"  [{idx}] {model_dir} (points3D bytes={points_count(model_dir)})")

    # Export best sparse model to TXT and extract camera centers for normal orientation.
    cam_centers_path = dense_cloud_out.parent / "camera_centers.json"
    try:
        txt_dir = sparse_root / "_txt_export"
        txt_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                args.colmap_bin, "model_converter",
                "--input_path", str(model_dirs[0]),
                "--output_path", str(txt_dir),
                "--output_type", "TXT",
            ],
            check=True,
            capture_output=True,
        )
        centers = _read_camera_centers_from_txt(txt_dir / "images.txt")
        if centers:
            cam_centers_path.parent.mkdir(parents=True, exist_ok=True)
            with cam_centers_path.open("w") as _f:
                json.dump(centers, _f)
            print(f"[set] {set_label} camera centers saved: {cam_centers_path} ({len(centers)} cameras)")
    except Exception as _cam_exc:
        print(f"[set] {set_label} warning: could not extract camera centers: {_cam_exc}", file=sys.stderr)

    dense_parts: list[Path] = []
    single_model = len(model_dirs) == 1
    for idx, model_dir in enumerate(model_dirs):
        if single_model:
            dense_workspace_dir = dense_root
            cloud_out = dense_cloud_out
        else:
            dense_workspace_dir = dense_root / f"component_{idx:02d}"
            cloud_out = dense_root / f"fused_component_{idx:02d}.ply"

        # --- SNAPSHOT 1: Keypoint detection and sparse cloud creation ---
        print(f"[step] colmap model_converter (sparse snapshot {idx})")
        snapshot_dir = make_snapshot_dir(dense_cloud_out)
        sparse_ply_out = snapshot_dir / f"{set_label}_sparse_cloud_{idx:02d}.ply"
        try:
            subprocess.run([
                args.colmap_bin,
                "model_converter",
                "--input_path", str(model_dir),
                "--output_path", str(sparse_ply_out),
                "--output_type", "PLY"
            ], check=True)
            print(f"snapshot saved: {sparse_ply_out}")
        except Exception as e:
            print(f"warning: could not save sparse snapshot: {e}")

        # --- THE ORIGINAL WORKING CALL (Image undistortion and dense cloud creation) ---
        run_dense_for_model(
            colmap_bin=args.colmap_bin,
            images=images_for_colmap, # Matches your working script
            sparse_model_dir=model_dir,
            dense_workspace_dir=dense_workspace_dir,
            cloud_out=cloud_out,
            patch_max_image_size=patch_max_image_size,
            fusion_max_image_size=fusion_max_image_size,
            patch_window_radius=patch_window_radius,
            patch_num_samples=patch_num_samples,
            patch_num_iterations=patch_num_iterations,
            patch_filter_min_ncc=patch_filter_min_ncc,
            patch_filter_min_consistent=patch_filter_min_consistent,
            patch_cache_size=patch_cache_size,
            patch_gpu_index=gpu_index, # Matches your working script
            fusion_min_num_pixels=fusion_min_num_pixels,
            fusion_max_reproj_error=fusion_max_reproj_error,
            fusion_max_depth_error=fusion_max_depth_error,
            fusion_max_normal_error=fusion_max_normal_error,
            fusion_threads=fusion_threads, # Matches your working script
            fusion_cache_size=fusion_cache_size,
            fusion_use_cache=fusion_use_cache,
            model_label=f"{set_label} component {idx}",
        )

        if not cloud_out.exists():
            print(f"error: dense cloud component not created: {cloud_out}", file=sys.stderr)
            return 4
        dense_parts.append(cloud_out)
        
        # --- SNAPSHOT 2: Normal/Depth maps and fused .ply cloud ---
        dense_ply_snap = snapshot_dir / f"{set_label}_dense_fused_{idx:02d}.ply"
        shutil.copy2(cloud_out, dense_ply_snap)
        print(f"snapshot saved: {dense_ply_snap}")

    merge_component_clouds(dense_parts, dense_cloud_out)

    if not dense_cloud_out.exists():
        print(f"error: dense cloud was not created: {dense_cloud_out}", file=sys.stderr)
        return 4

    print(f"[set] {set_label} dense cloud output: {dense_cloud_out}")
    return 0


def main() -> int:
    args = parse_args()
    apply_quality_preset(args)
    images = Path(args.images)
    images_secondary_raw = str(args.images_secondary).strip()
    images_secondary = Path(images_secondary_raw) if images_secondary_raw else None
    workspace = Path(args.workspace)
    dense_cloud_out = Path(args.dense_cloud_out)

    if not images.is_dir():
        print(f"error: image dir not found: {images}", file=sys.stderr)
        return 2
    if images_secondary is not None and not images_secondary.is_dir():
        print(f"error: secondary image dir not found: {images_secondary}", file=sys.stderr)
        return 2

    secondary_pre_rotate_axis = str(args.secondary_pre_rotate_axis).strip().lower() or "y"
    if secondary_pre_rotate_axis not in {
        "x",
        "y",
        "z",
        "principal",
        "principal_major",
        "principal_minor1",
        "principal_minor2",
        "primary_principal",
        "primary_principal_major",
        "primary_principal_minor1",
        "primary_principal_minor2",
        "primary_frame_x",
        "primary_frame_y",
        "primary_frame_z",
        "vector",
        "auto_halfturn",
        "pca_halfturn_auto",
    }:
        print(
            "error: --secondary-pre-rotate-axis must be one of: "
            "x, y, z, principal, principal_major, principal_minor1, principal_minor2, "
            "primary_principal, primary_principal_major, primary_principal_minor1, "
            "primary_principal_minor2, primary_frame_x, primary_frame_y, primary_frame_z, "
            "vector, auto_halfturn, pca_halfturn_auto",
            file=sys.stderr,
        )
        return 2
    secondary_pre_rotate_deg = float(args.secondary_pre_rotate_deg)
    secondary_pre_rotate_axis_vector = (
        tuple(float(v) for v in args.secondary_pre_rotate_axis_vector)
        if args.secondary_pre_rotate_axis_vector is not None
        else None
    )
    if secondary_pre_rotate_axis == "vector" and secondary_pre_rotate_axis_vector is None:
        print(
            "error: --secondary-pre-rotate-axis=vector requires --secondary-pre-rotate-axis-vector AX AY AZ",
            file=sys.stderr,
        )
        return 2
    secondary_auto_halfturn_axes = max(1, int(args.secondary_auto_halfturn_axes))
    secondary_extra_rotate_x = float(args.secondary_extra_rotate_x)
    secondary_extra_rotate_y = float(args.secondary_extra_rotate_y)
    secondary_extra_rotate_z = float(args.secondary_extra_rotate_z)
    secondary_translate_x = float(args.secondary_translate_x)
    secondary_translate_y = float(args.secondary_translate_y)
    secondary_translate_z = float(args.secondary_translate_z)
    secondary_align_max_rotation_deg = max(0.0, float(args.secondary_align_max_rotation_deg))
    secondary_align_mode = str(args.secondary_align_mode).strip().lower() or "auto"
    if secondary_align_mode not in {
        "auto",
        "off",
        "centroid",
        "centroid_icp",
        "centroid_icp_overlap",
        "centroid_icp_overlap_capped",
        "centroid_overlap_translate",
    }:
        print(
            "error: --secondary-align-mode must be one of: auto, off, centroid, centroid_icp, "
            "centroid_icp_overlap, centroid_icp_overlap_capped, centroid_overlap_translate",
            file=sys.stderr,
        )
        return 2

    if shutil.which(args.colmap_bin) is None:
        print(
            f"error: COLMAP binary not found: {args.colmap_bin}. "
            "Install COLMAP or set FIPMESH_COLMAP_BIN.",
            file=sys.stderr,
        )
        return 3

    use_gpu = int(args.use_gpu) != 0
    gpu_index = str(args.gpu_index).strip() or "-1"
    single_camera = int(args.single_camera) != 0
    single_camera_per_folder = int(args.single_camera_per_folder) != 0
    extract_threads = max(1, int(args.extract_threads))
    match_threads = max(1, int(args.match_threads))
    mapper_threads = max(1, int(args.mapper_threads))
    fusion_threads = max(1, int(args.fusion_threads))
    input_image_scale = min(max(float(args.input_image_scale), 1e-3), 1.0)
    input_image_stride = max(1, int(args.input_image_stride))
    sift_max_image_size = max(256, int(args.sift_max_image_size))
    sift_max_num_features = max(1024, int(args.sift_max_num_features))
    sift_peak_threshold = max(1e-6, float(args.sift_peak_threshold))
    sift_edge_threshold = max(1.0, float(args.sift_edge_threshold))
    sift_domain_size_pooling = int(args.sift_domain_size_pooling) != 0
    sift_estimate_affine_shape = int(args.sift_estimate_affine_shape) != 0
    patch_max_image_size = max(256, int(args.patch_match_max_image_size))
    fusion_max_image_size = max(256, int(args.fusion_max_image_size))
    patch_window_radius = max(1, int(args.patch_match_window_radius))
    patch_num_samples = max(5, int(args.patch_match_num_samples))
    patch_num_iterations = max(1, int(args.patch_match_num_iterations))
    patch_filter_min_ncc = min(max(float(args.patch_match_filter_min_ncc), 0.0), 1.0)
    patch_filter_min_consistent = max(1, int(args.patch_match_filter_min_consistent))
    patch_cache_size = max(1, int(args.patch_match_cache_size))
    fusion_min_num_pixels = max(1, int(args.fusion_min_num_pixels))
    fusion_max_reproj_error = max(0.1, float(args.fusion_max_reproj_error))
    fusion_max_depth_error = max(1e-5, float(args.fusion_max_depth_error))
    fusion_max_normal_error = max(0.1, float(args.fusion_max_normal_error))
    fusion_cache_size = max(1, int(args.fusion_cache_size))
    fusion_use_cache = int(args.fusion_use_cache) != 0
    match_guided = int(args.match_guided) != 0
    match_max_num_matches = max(1024, int(args.match_max_num_matches))
    option_style = detect_colmap_option_style(args.colmap_bin)
    camera_model = str(args.camera_model).strip()
    focal_length_px = max(0.0, float(args.focal_length))

    image_sets: list[tuple[str, Path, dict[str, float | str] | None]] = [("primary", images, None)]
    if images_secondary is not None:
        image_sets.append(
            (
                "secondary",
                images_secondary,
                {
                    "axis": secondary_pre_rotate_axis,
                    "degrees": secondary_pre_rotate_deg,
                    "extra_rotate_x": secondary_extra_rotate_x,
                    "extra_rotate_y": secondary_extra_rotate_y,
                    "extra_rotate_z": secondary_extra_rotate_z,
                    "translate_x": secondary_translate_x,
                    "translate_y": secondary_translate_y,
                    "translate_z": secondary_translate_z,
                    "axis_vector_x": secondary_pre_rotate_axis_vector[0] if secondary_pre_rotate_axis_vector is not None else 0.0,
                    "axis_vector_y": secondary_pre_rotate_axis_vector[1] if secondary_pre_rotate_axis_vector is not None else 0.0,
                    "axis_vector_z": secondary_pre_rotate_axis_vector[2] if secondary_pre_rotate_axis_vector is not None else 0.0,
                    "auto_halfturn_axes": secondary_auto_halfturn_axes,
                    "align_max_rotation_deg": secondary_align_max_rotation_deg,
                    "align_mode": secondary_align_mode,
                },
            )
        )
        print(
            "dual image-set mode: "
            f"primary={images} secondary={images_secondary} "
            f"secondary_pre_rotate={secondary_pre_rotate_deg:g}deg@{secondary_pre_rotate_axis} "
            f"secondary_pre_rotate_axis_vector={secondary_pre_rotate_axis_vector} "
            f"secondary_extra_rotate=({secondary_extra_rotate_x:g},{secondary_extra_rotate_y:g},{secondary_extra_rotate_z:g}) "
            f"secondary_translate=({secondary_translate_x:g},{secondary_translate_y:g},{secondary_translate_z:g}) "
            f"secondary_align_mode={secondary_align_mode} "
            f"secondary_align_max_rotation_deg={secondary_align_max_rotation_deg:g}"
        )
        if args.clean_workspace and workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

    set_clouds: list[Path] = []
    set_pre_transforms: list[dict[str, float | str] | None] = []
    multi_set = len(image_sets) > 1

    def _build_set_paths(idx: int, set_label: str) -> tuple[Path, Path]:
        if multi_set:
            set_workspace = workspace / f"set_{idx:02d}_{set_label}"
            set_dense_cloud = set_workspace / "dense" / "fused.ply"
        else:
            set_workspace = workspace
            set_dense_cloud = dense_cloud_out
        return set_workspace, set_dense_cloud

    def _run_one_set(idx: int, set_label: str, set_images: Path) -> tuple[int, Path]:
        set_workspace, set_dense_cloud = _build_set_paths(idx, set_label)
        rc = run_pipeline_for_image_set(
            args=args,
            images=set_images,
            workspace=set_workspace,
            dense_cloud_out=set_dense_cloud,
            clean_workspace=args.clean_workspace,
            set_label=set_label,
            use_gpu=use_gpu,
            gpu_index=gpu_index,
            single_camera=single_camera,
            extract_threads=extract_threads,
            match_threads=match_threads,
            mapper_threads=mapper_threads,
            fusion_threads=fusion_threads,
            input_image_scale=input_image_scale,
            input_image_stride=input_image_stride,
            sift_max_image_size=sift_max_image_size,
            sift_max_num_features=sift_max_num_features,
            sift_peak_threshold=sift_peak_threshold,
            sift_edge_threshold=sift_edge_threshold,
            sift_domain_size_pooling=sift_domain_size_pooling,
            sift_estimate_affine_shape=sift_estimate_affine_shape,
            patch_max_image_size=patch_max_image_size,
            fusion_max_image_size=fusion_max_image_size,
            patch_window_radius=patch_window_radius,
            patch_num_samples=patch_num_samples,
            patch_num_iterations=patch_num_iterations,
            patch_filter_min_ncc=patch_filter_min_ncc,
            patch_filter_min_consistent=patch_filter_min_consistent,
            patch_cache_size=patch_cache_size,
            fusion_min_num_pixels=fusion_min_num_pixels,
            fusion_max_reproj_error=fusion_max_reproj_error,
            fusion_max_depth_error=fusion_max_depth_error,
            fusion_max_normal_error=fusion_max_normal_error,
            fusion_cache_size=fusion_cache_size,
            fusion_use_cache=fusion_use_cache,
            match_guided=match_guided,
            match_max_num_matches=match_max_num_matches,
            option_style=option_style,
            camera_model=camera_model,
            focal_length_px=focal_length_px,
            single_camera_per_folder=single_camera_per_folder,
        )
        return rc, set_dense_cloud

    if multi_set:
        print(f"[parallel] running {len(image_sets)} image sets simultaneously")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(image_sets)) as pool:
            futures = [
                pool.submit(_run_one_set, idx, set_label, set_images)
                for idx, (set_label, set_images, _) in enumerate(image_sets)
            ]
            results = [f.result() for f in futures]
        for (rc, set_dense_cloud), (_, _, pre_transform) in zip(results, image_sets):
            if rc != 0:
                return rc
            set_clouds.append(set_dense_cloud)
            set_pre_transforms.append(pre_transform)
    else:
        idx, (set_label, set_images, pre_transform) = 0, image_sets[0]
        rc, set_dense_cloud = _run_one_set(idx, set_label, set_images)
        if rc != 0:
            return rc
        set_clouds.append(set_dense_cloud)
        set_pre_transforms.append(pre_transform)

    if multi_set:
        dense_cloud_out.parent.mkdir(parents=True, exist_ok=True)
        print("[step] merge image-set clouds")
        merge_component_clouds(
            set_clouds,
            dense_cloud_out,
            pre_transforms=set_pre_transforms,
        )

    if not dense_cloud_out.exists():
        print(f"error: dense cloud was not created: {dense_cloud_out}", file=sys.stderr)
        return 4

    print(f"dense cloud output: {dense_cloud_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())