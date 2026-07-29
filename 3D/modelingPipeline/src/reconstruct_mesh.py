#!/usr/bin/env python3
"""
Reconstruct a mesh from a point cloud file using Open3D.

Pipeline:
1. Read point cloud from OBJ `v` lines or a point-cloud file (PLY/PCD/XYZ/XYZN/XYZRGB).
2. Clean/filter cloud (dedupe, statistical outliers, radius outliers, cluster trim, crop).
3. Estimate and orient normals.
4. Surface reconstruction (Poisson).
5. Mesh cleanup (density trim, topology cleanup, component filtering, smoothing).
6. Optional decimation (quadric error).
7. Optional texture hook (external command).

This is the "dense cloud -> usable mesh" stage. It also works on sparse clouds, but
reconstruction quality depends heavily on point density and coverage.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover - runtime dependency
    print(
        "error: Open3D is required. Install with `pip install open3d` "
        "or your distro package manager.",
        file=sys.stderr,
    )
    print(f"detail: {exc}", file=sys.stderr)
    raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cloud cleanup + Poisson reconstruction + mesh cleanup via Open3D."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input point cloud (OBJ with `v` lines, or PLY/PCD/XYZ/...)",
    )
    parser.add_argument("--output", required=True, help="Output reconstructed mesh (OBJ/PLY/etc).")
    parser.add_argument(
        "--output-clean-cloud",
        default="",
        help="Optional cleaned point cloud output (PLY recommended).",
    )
    parser.add_argument(
        "--output-decimated",
        default="",
        help="Optional decimated mesh output path.",
    )
    parser.add_argument(
        "--export-all-variants",
        action="store_true",
        default=os.environ.get("FIPMESH_EXPORT_ALL_VARIANTS", "0") == "1",
        help=(
            "Also write legacy decimated/textured variant files. "
            "Default is off, which exports only the single highest-quality mesh per format."
        ),
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional downsample voxel size (0 disables; default: 0).",
    )
    parser.add_argument(
        "--max-input-points",
        type=int,
        default=int(os.environ.get("FIPMESH_RECON_MAX_INPUT_POINTS", "400000")),
        help=(
            "Randomly downsample to at most this many points before reconstruction. "
            "<=0 disables (default: 400000)."
        ),
    )
    parser.add_argument(
        "--outlier-nb-neighbors",
        type=int,
        default=24,
        help="Neighbors for statistical outlier removal (default: 24).",
    )
    parser.add_argument(
        "--outlier-std-ratio",
        type=float,
        default=2.0,
        help="Std dev threshold for statistical outlier removal (<=0 disables; default: 2.0).",
    )
    parser.add_argument(
        "--radius-outlier-nb-points",
        type=int,
        default=8,
        help="Min neighbors for radius outlier removal (<=0 disables; default: 8).",
    )
    parser.add_argument(
        "--radius-outlier-radius",
        type=float,
        default=0.0,
        help="Radius for radius outlier removal (0 = auto from NN distances; default: 0).",
    )
    parser.add_argument(
        "--radius-outlier-radius-factor",
        type=float,
        default=2.5,
        help="Auto radius = median NN distance * factor (default: 2.5).",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.0,
        help="DBSCAN eps for cluster cleanup (0 = auto from NN distances; default: 0).",
    )
    parser.add_argument(
        "--dbscan-eps-factor",
        type=float,
        default=3.5,
        help="Auto DBSCAN eps = median NN distance * factor (default: 3.5).",
    )
    parser.add_argument(
        "--dbscan-min-points",
        type=int,
        default=12,
        help="DBSCAN min_points (<=0 disables cluster cleanup; default: 12).",
    )
    parser.add_argument(
        "--dbscan-max-points",
        type=int,
        default=int(os.environ.get("FIPMESH_RECON_DBSCAN_MAX_POINTS", "150000")),
        help=(
            "Skip DBSCAN cluster cleanup when point count exceeds this value. "
            "<=0 disables the guard (default: 150000)."
        ),
    )
    parser.add_argument(
        "--dbscan-keep-largest",
        type=int,
        default=4,
        help="Keep up to N largest DBSCAN clusters (0 disables cap; default: 4).",
    )
    parser.add_argument(
        "--dbscan-min-cluster-ratio",
        type=float,
        default=0.01,
        help="Keep DBSCAN clusters at least this size ratio of largest (default: 0.01).",
    )
    parser.add_argument(
        "--crop-quantile",
        type=float,
        default=0.0,
        help=(
            "Optional per-axis quantile crop. Example: 0.01 trims 1%% from each side of "
            "each axis. 0 disables (default: 0)."
        ),
    )

    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.0,
        help="Normal estimation radius (0 = auto from cloud scale; default: 0).",
    )
    parser.add_argument(
        "--normal-max-nn",
        type=int,
        default=48,
        help="Max neighbors for normal estimation (default: 48).",
    )
    parser.add_argument(
        "--normal-orient-k",
        type=int,
        default=32,
        help="k for consistent normal orientation (default: 32).",
    )

    parser.add_argument(
        "--poisson-depth",
        type=int,
        default=9,
        help="Poisson octree depth (default: 9).",
    )
    parser.add_argument(
        "--poisson-scale",
        type=float,
        default=1.1,
        help="Poisson scale parameter (default: 1.1).",
    )
    parser.add_argument(
        "--poisson-crop-scale",
        type=float,
        default=1.05,
        help="Crop Poisson mesh to point-cloud AABB scaled by this factor (default: 1.05).",
    )
    parser.add_argument(
        "--poisson-linear-fit",
        action="store_true",
        help="Enable linear fit in Poisson reconstruction.",
    )
    parser.add_argument(
        "--density-trim-quantile",
        type=float,
        default=0.02,
        help="Remove vertices below this Poisson density quantile (default: 0.02).",
    )

    parser.add_argument(
        "--component-min-ratio",
        type=float,
        default=0.01,
        help=(
            "Keep mesh connected components with at least this triangle-count ratio of "
            "the largest component (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--component-min-triangles",
        type=int,
        default=200,
        help="Absolute minimum triangle count to keep a component (default: 200).",
    )
    parser.add_argument(
        "--component-max-count",
        type=int,
        default=8,
        help="Maximum number of largest components to keep (default: 8).",
    )
    parser.add_argument(
        "--smooth-iters",
        type=int,
        default=2,
        help="Taubin smoothing iterations after cleanup (default: 2).",
    )
    parser.add_argument(
        "--decimate-target-triangles",
        type=int,
        default=300000,
        help="Quadric decimation target triangle count (<=0 disables; default: 300000).",
    )
    parser.add_argument(
        "--simplified-target-vertices",
        type=int,
        default=int(os.environ.get("FIPMESH_SIMPLIFIED_TARGET_VERTICES", "60000")),
        help=(
            "Also export a second, decimated '<name>_simplified.glb' mesh with "
            "approximately this many vertices, as a single small binary glTF file "
            "(vertex-colored, no separate texture/MTL) meant for easy loading into a "
            "web viewer. <=0 disables this export (default: 60000)."
        ),
    )

    parser.add_argument(
        "--fill-holes",
        type=int,
        default=int(os.environ.get("FIPMESH_RECON_FILL_HOLES", "0")),
        help=(
            "Watertight repair pass: triangulate remaining mesh boundary loops to close "
            "holes left by density trimming / component filtering, bridging sparsely "
            "sampled regions (e.g. near a turntable rotation axis, or low-texture/smooth "
            "surfaces) instead of leaving a gap (0/1, default: 0)."
        ),
    )
    parser.add_argument(
        "--fill-holes-max-size",
        type=float,
        default=0.0,
        help=(
            "Maximum hole radius to fill, in point-cloud units. <=0 = auto, computed as "
            "--fill-holes-max-size-ratio times the cleaned point cloud's bounding-box "
            "diagonal (default: 0, i.e. auto)."
        ),
    )
    parser.add_argument(
        "--fill-holes-max-size-ratio",
        type=float,
        default=0.3,
        help=(
            "Auto hole-fill radius = this ratio * cleaned point cloud bounding-box "
            "diagonal. Only used when --fill-holes-max-size <= 0 (default: 0.3). Raise "
            "it to close larger gaps; lower it to avoid bridging genuinely unscanned "
            "regions."
        ),
    )
    parser.add_argument(
        "--fill-holes-passes",
        type=int,
        default=int(os.environ.get("FIPMESH_RECON_FILL_HOLES_PASSES", "4")),
        help=(
            "Open3D's hole filler does not always converge in one call: closing one "
            "loop can leave a neighboring loop fillable only on a later pass, so a "
            "single call can leave small holes behind well under the size threshold. "
            "This repeats the fill pass (on its own output) up to this many times, "
            "stopping early once a pass adds no new triangles (default: 4)."
        ),
    )

    parser.add_argument(
        "--normalize-pose",
        type=int,
        default=int(os.environ.get("FIPMESH_RECON_NORMALIZE_POSE", "0")),
        help=(
            "Recenter the mesh (and cleaned cloud) at the origin and rotate it via PCA "
            "so its largest cross-section lies in the XY plane, thickness along Z -- "
            "i.e. orient the tablet flat, like a coin lying on a table. Applied once, "
            "before decimation/texturing, so it carries through to every exported "
            "variant (0/1, default: 0)."
        ),
    )
    parser.add_argument(
        "--mirror-backfill",
        choices=("none", "x", "y", "z", "auto"),
        default="none",
        help=(
            "Mirror points across a median plane before Poisson. Useful if cloud is "
            "strongly one-sided (default: none)."
        ),
    )
    parser.add_argument(
        "--mirror-auto-threshold",
        type=float,
        default=0.35,
        help=(
            "For mirror-backfill=auto: trigger when one-sidedness ratio is below this "
            "value (default: 0.35)."
        ),
    )

    parser.add_argument(
        "--texture-command",
        default="",
        help=(
            "Optional external texturing command template. Supported placeholders: "
            "{mesh}, {decimated}, {input}. If empty, texture stage is skipped."
        ),
    )
    parser.add_argument(
        "--obj-texture-max-size",
        type=int,
        default=int(os.environ.get("FIPMESH_OBJ_TEXTURE_MAX_SIZE", "4096")),
        help=(
            "Maximum width/height of generated OBJ texture atlas in pixels "
            "(default: 4096)."
        ),
    )
    parser.add_argument(
        "--obj-texture-min-tile",
        type=int,
        default=int(os.environ.get("FIPMESH_OBJ_TEXTURE_MIN_TILE", "4")),
        help=(
            "Minimum per-triangle tile size in texture atlas pixels "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--obj-texture-max-tile",
        type=int,
        default=int(os.environ.get("FIPMESH_OBJ_TEXTURE_MAX_TILE", "16")),
        help=(
            "Maximum per-triangle tile size in texture atlas pixels "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--camera-centers",
        default=os.environ.get("FIPMESH_CAMERA_CENTERS", ""),
        help=(
            "Path to camera_centers.json produced by run_colmap_mvs.py. "
            "When provided, normals are oriented toward camera positions instead of "
            "using the centroid heuristic. Env: FIPMESH_CAMERA_CENTERS."
        ),
    )
    parser.add_argument(
        "--side1-camera-centers",
        default=os.environ.get("FIPMESH_SIDE1_CAMERA_CENTERS", ""),
        help=(
            "Path to side 1's camera_centers.json (produced by run_colmap_mvs.py for "
            "the side-1 COLMAP run). Only used when --normalize-pose is enabled: after "
            "flattening, if side 1's cameras ended up on the -Z side, the mesh is "
            "rotated 180 degrees so side 1 faces +Z (up) -- without mirroring the "
            "geometry. Env: FIPMESH_SIDE1_CAMERA_CENTERS."
        ),
    )
    return parser.parse_args()

def save_snapshot(base_path: Path, step_name: str, pcd_or_mesh) -> None:
    """Saves an intermediate point cloud or mesh to a snapshots directory."""
    snap_dir = base_path.parent / (base_path.stem + "_snapshots")
    snap_dir.mkdir(parents=True, exist_ok=True)
    out_path = snap_dir / f"{base_path.stem}_{step_name}.ply"
    if isinstance(pcd_or_mesh, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(str(out_path), pcd_or_mesh, write_ascii=False)
    elif isinstance(pcd_or_mesh, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(str(out_path), pcd_or_mesh, write_ascii=False)
    print(f"snapshot saved: {out_path}")

def ensure_parent(path_str: str) -> None:
    if not path_str:
        return
    path = Path(path_str)
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)


def read_obj_vertices_and_colors(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    verts = []
    cols = []
    saw_any_color = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                if len(parts) >= 7:
                    r = float(parts[4])
                    g = float(parts[5])
                    b = float(parts[6])
                    # Accept either 0..1 or 0..255 vertex colors.
                    if max(abs(r), abs(g), abs(b)) > 1.0:
                        r /= 255.0
                        g /= 255.0
                        b /= 255.0
                    cols.append((r, g, b))
                    saw_any_color = True
                else:
                    cols.append((np.nan, np.nan, np.nan))
            except ValueError:
                continue
    if not verts:
        return np.empty((0, 3), dtype=np.float64), None
    points = np.asarray(verts, dtype=np.float64)
    if saw_any_color:
        colors = np.asarray(cols, dtype=np.float64)
        valid = np.isfinite(colors).all(axis=1)
        if np.any(valid):
            if not np.all(valid):
                colors = colors.copy()
                colors[~valid] = np.clip(np.nanmean(colors[valid], axis=0), 0.0, 1.0)
            colors = np.clip(colors, 0.0, 1.0)
            return points, colors
    return points, None


def read_input_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        points, colors = read_obj_vertices_and_colors(path)
        pcd = pcd_from_points(points)
        if colors is not None and len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    # Prefer direct point-cloud readers for dense MVS outputs like COLMAP fused.ply.
    try:
        pcd = o3d.io.read_point_cloud(str(path))
    except Exception:
        pcd = None
    if pcd is not None and len(pcd.points) > 0:
        return pcd

    # Fallback: if a mesh file is passed, use its vertices as a point set.
    try:
        mesh = o3d.io.read_triangle_mesh(str(path))
    except Exception:
        mesh = None
    if mesh is not None and len(mesh.vertices) > 0:
        pcd = pcd_from_points(np.asarray(mesh.vertices, dtype=np.float64))
        if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
            pcd.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors, dtype=np.float64))
        return pcd

    return pcd_from_points(np.empty((0, 3), dtype=np.float64))


def pcd_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd


def pcd_has_colors(pcd: o3d.geometry.PointCloud) -> bool:
    return pcd.has_colors() and len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0


def sanitize_points(points: np.ndarray, quant_eps: float = 1e-7) -> np.ndarray:
    if len(points) == 0:
        return points
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return points
    quantized = np.round(points / quant_eps).astype(np.int64)
    _, unique_idx = np.unique(quantized, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx]


def sanitize_point_cloud(
    pcd: o3d.geometry.PointCloud,
    quant_eps: float = 1e-7,
) -> tuple[o3d.geometry.PointCloud, int]:
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) == 0:
        return pcd, 0

    valid = np.isfinite(points).all(axis=1)
    points_valid = points[valid]
    if len(points_valid) == 0:
        return pcd_from_points(np.empty((0, 3), dtype=np.float64)), int(len(points))

    quantized = np.round(points_valid / quant_eps).astype(np.int64)
    _, unique_local_idx = np.unique(quantized, axis=0, return_index=True)
    unique_local_idx.sort()
    kept_count = len(unique_local_idx)

    pcd2 = pcd_from_points(points_valid[unique_local_idx])

    if pcd_has_colors(pcd):
        colors = np.asarray(pcd.colors, dtype=np.float64)
        colors_valid = colors[valid]
        if len(colors_valid) == len(points_valid):
            pcd2.colors = o3d.utility.Vector3dVector(
                np.clip(colors_valid[unique_local_idx], 0.0, 1.0)
            )

    removed = int(len(points) - kept_count)
    return pcd2, removed


def median_nn_distance(pcd: o3d.geometry.PointCloud) -> float:
    if len(pcd.points) < 2:
        return 0.0
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(d) == 0:
        return 0.0
    return float(np.median(d))


def auto_normal_radius(points: np.ndarray) -> float:
    if len(points) < 4:
        return 1.0
    med = median_nn_distance(pcd_from_points(points))
    if med <= 0.0:
        return 1.0
    return float(med * 3.0)


def quantile_crop_points(points: np.ndarray, q: float) -> np.ndarray:
    q = float(np.clip(q, 0.0, 0.49))
    if q <= 0.0 or len(points) < 10:
        return points
    lo = np.quantile(points, q, axis=0)
    hi = np.quantile(points, 1.0 - q, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    out = points[keep]
    return out if len(out) > 0 else points


def axis_one_sided_ratio(values: np.ndarray) -> float:
    q10, q50, q90 = np.percentile(values, [10, 50, 90])
    lo = float(q50 - q10)
    hi = float(q90 - q50)
    den = max(lo, hi, 1e-9)
    return min(lo, hi) / den


def pick_auto_mirror_axis(points: np.ndarray) -> tuple[str, float]:
    ratios = {
        "x": axis_one_sided_ratio(points[:, 0]),
        "y": axis_one_sided_ratio(points[:, 1]),
        "z": axis_one_sided_ratio(points[:, 2]),
    }
    axis = min(ratios, key=ratios.get)
    return axis, float(ratios[axis])


def apply_mirror_backfill(points: np.ndarray, axis: str) -> tuple[np.ndarray, float]:
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    idx = axis_to_idx[axis]
    pivot = float(np.median(points[:, idx]))
    mirrored = points.copy()
    mirrored[:, idx] = 2.0 * pivot - mirrored[:, idx]
    merged = np.vstack([points, mirrored])
    quantized = np.round(merged / 1e-7).astype(np.int64)
    _, unique_idx = np.unique(quantized, axis=0, return_index=True)
    unique_idx.sort()
    return merged[unique_idx], pivot


def load_camera_centers(path: str) -> list[list[float]]:
    """Load a camera_centers.json (list of [x, y, z]) written by run_colmap_mvs.py.
    Returns [] if `path` is blank or unreadable."""
    path = str(path).strip()
    if not path:
        return []
    try:
        with open(path) as f:
            centers = json.load(f)
        print(f"camera centers loaded: {len(centers)} from {path}")
        return centers
    except Exception as exc:
        print(f"warning: could not load camera centers from {path}: {exc}", file=sys.stderr)
        return []


def orient_up_towards_side1(mesh, pcd, pose_transform: np.ndarray,
                             side1_camera_centers: list[list[float]]) -> bool:
    """After `pose_transform` has flattened the cloud onto the XY plane, flip it
    180 degrees (about X) if side 1's cameras ended up on the -Z side, so side 1
    consistently ends up facing +Z. A 180-degree rotation (rather than negating a
    single axis) keeps the transform a proper rotation, so the mesh is reoriented,
    never mirrored. Returns True if a flip was applied."""
    if not side1_camera_centers:
        return False
    centers = np.asarray(side1_camera_centers, dtype=np.float64)
    avg_center = np.append(centers.mean(axis=0), 1.0)
    local_z = float((pose_transform @ avg_center)[2])
    if local_z >= 0:
        return False
    flip_180_x = np.diag([1.0, -1.0, -1.0, 1.0])
    mesh.transform(flip_180_x)
    pcd.transform(flip_180_x)
    return True


def compute_pca_pose(points: np.ndarray) -> np.ndarray:
    """4x4 rigid transform that recenters `points` at the origin and rotates them
    so the two largest-variance axes span X/Y and the smallest-variance
    (thickness) axis lies along Z -- i.e. the largest cross-section ends up in
    the XY plane, like a coin lying flat on a table."""
    centroid = points.mean(axis=0)
    cov = np.cov((points - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    order = np.argsort(eigvals)[::-1]       # descending variance: X, Y, Z(=thinnest)
    rotation = eigvecs[:, order]
    if np.linalg.det(rotation) < 0:         # keep it a proper (non-mirroring) rotation
        rotation[:, -1] *= -1.0
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = -rotation.T @ centroid
    return transform


def orient_normals_outward_from_centroid(pcd: o3d.geometry.PointCloud) -> None:
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    if len(points) == 0 or len(normals) == 0:
        return
    centroid = points.mean(axis=0)
    rays = points - centroid
    flip = np.einsum("ij,ij->i", normals, rays) < 0.0
    normals[flip] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(normals)


def orient_normals_towards_cameras(
    pcd: o3d.geometry.PointCloud,
    camera_centers: list[list[float]],
) -> None:
    """Orient normals toward COLMAP-recovered camera positions, then run a consistency pass."""
    if len(camera_centers) == 0 or len(pcd.points) == 0 or not pcd.has_normals():
        return
    for center in camera_centers:
        pcd.orient_normals_towards_camera_location(np.array(center, dtype=np.float64))
    pcd.orient_normals_consistent_tangent_plane(30)


def cluster_cleanup_point_cloud(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
    keep_largest: int,
    min_cluster_ratio: float,
) -> tuple[o3d.geometry.PointCloud, dict[str, int | float]]:
    info: dict[str, int | float] = {
        "clusters_total": 0,
        "clusters_kept": 0,
        "largest_cluster": 0,
        "eps": float(eps),
    }
    if len(pcd.points) < max(10, min_points) or min_points <= 0:
        return pcd, info

    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0:
        return pcd, info

    valid = labels >= 0
    if not np.any(valid):
        return pcd, info

    counts = np.bincount(labels[valid])
    if counts.size == 0:
        return pcd, info

    info["clusters_total"] = int(counts.size)
    largest = int(np.max(counts))
    info["largest_cluster"] = largest
    min_keep = max(1, int(np.floor(largest * max(0.0, min_cluster_ratio))))

    order = np.argsort(counts)[::-1]
    keep_ids: list[int] = []
    for cid in order:
        if int(counts[cid]) < min_keep:
            continue
        keep_ids.append(int(cid))
        if keep_largest > 0 and len(keep_ids) >= keep_largest:
            break
    if not keep_ids:
        keep_ids = [int(order[0])]

    keep_mask = np.isin(labels, np.asarray(keep_ids, dtype=np.int64))
    keep_idx = np.where(keep_mask)[0]
    info["clusters_kept"] = len(keep_ids)
    return pcd.select_by_index(keep_idx), info


def cleanup_mesh_topology(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0:
        return mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def orient_mesh_winding_outward(
    mesh: o3d.geometry.TriangleMesh,
) -> tuple[o3d.geometry.TriangleMesh, bool]:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh, False

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(tris) == 0:
        return mesh, False

    tri_pts = verts[tris]
    e1 = tri_pts[:, 1] - tri_pts[:, 0]
    e2 = tri_pts[:, 2] - tri_pts[:, 0]
    cross = np.cross(e1, e2)  # magnitude = 2 * area, direction = winding normal
    tri_centers = tri_pts.mean(axis=1)
    mesh_center = verts.mean(axis=0)
    rays = tri_centers - mesh_center

    # Positive score means normals generally point away from the mesh centroid.
    score = float(np.sum(np.einsum("ij,ij->i", cross, rays)))
    if not np.isfinite(score) or abs(score) < 1e-12:
        return mesh, False
    if score > 0.0:
        return mesh, False

    tris_flipped = tris[:, [0, 2, 1]].copy()
    mesh.triangles = o3d.utility.Vector3iVector(tris_flipped.astype(np.int32))
    return mesh, True


def keep_significant_mesh_components(
    mesh: o3d.geometry.TriangleMesh,
    min_ratio: float,
    min_triangles: int,
    max_count: int,
) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) == 0:
        return mesh
    tri_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    cluster_n_tris = np.asarray(cluster_n_tris)
    if len(cluster_n_tris) == 0:
        return mesh

    largest_count = int(np.max(cluster_n_tris))
    min_keep = max(int(min_triangles), int(np.floor(largest_count * max(0.0, min_ratio))))

    keep_cluster_ids = np.where(cluster_n_tris >= min_keep)[0]
    if max_count > 0 and len(keep_cluster_ids) > max_count:
        ranked = keep_cluster_ids[np.argsort(cluster_n_tris[keep_cluster_ids])[::-1]]
        keep_cluster_ids = ranked[:max_count]

    if len(keep_cluster_ids) == 0:
        keep_cluster_ids = np.array([int(np.argmax(cluster_n_tris))], dtype=np.int64)

    remove_mask = ~np.isin(tri_clusters, keep_cluster_ids)
    mesh.remove_triangles_by_mask(remove_mask)
    mesh.remove_unreferenced_vertices()
    return mesh


def fill_mesh_holes(
    mesh: o3d.geometry.TriangleMesh,
    max_hole_size: float,
    max_passes: int = 4,
) -> tuple[o3d.geometry.TriangleMesh, int]:
    """Triangulate remaining boundary loops to close holes up to `max_hole_size` radius.

    Open3D's hole filler does not always converge in a single call: closing one loop
    can leave an adjacent loop fillable only once the mesh has already been updated.
    Repeating the pass on its own output picks up those stragglers; each pass is
    cheap once nearly converged, so this stops as soon as a pass adds no triangles.
    """
    if len(mesh.triangles) == 0 or max_hole_size <= 0 or max_passes <= 0:
        return mesh, 0
    current = mesh
    passes_run = 0
    for _ in range(max_passes):
        try:
            t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(current)
            filled = t_mesh.fill_holes(hole_size=float(max_hole_size))
            legacy = filled.to_legacy()
        except Exception as exc:
            print(f"warning: hole filling failed, skipping ({exc})", file=sys.stderr)
            break
        if len(legacy.vertices) == 0 or len(legacy.triangles) == 0:
            break
        added = len(legacy.triangles) - len(current.triangles)
        current = legacy
        passes_run += 1
        if added <= 0:
            break
    return current, passes_run


def run_texture_hook(
    cmd_template: str,
    mesh_path: Path,
    decimated_path: Path | None,
    input_path: Path,
) -> int:
    if not cmd_template:
        return 0
    try:
        cmd = cmd_template.format(
            mesh=str(mesh_path),
            decimated=str(decimated_path) if decimated_path else "",
            input=str(input_path),
        )
    except KeyError as exc:
        print(f"warning: invalid texture-command placeholder: {exc}", file=sys.stderr)
        return 1

    print(f"texture hook command: {cmd}")
    proc = subprocess.run(cmd, shell=True, check=False)
    return int(proc.returncode)


def default_textured_output_path(mesh_path: Path) -> Path:
    stem = mesh_path.stem
    if stem.endswith("_textured"):
        return mesh_path.with_suffix(".ply")
    return mesh_path.with_name(f"{stem}_textured.ply")


def default_ply_output_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".ply")


def default_textured_obj_output_path(mesh_path: Path) -> Path:
    stem = mesh_path.stem
    if stem.endswith("_textured"):
        return mesh_path.with_suffix(".obj")
    return mesh_path.with_name(f"{stem}_textured.obj")


def default_obj_texture_map_path(obj_path: Path) -> Path:
    return obj_path.with_suffix(".png")


def default_gltf_output_path(mesh_path: Path) -> Path:
    return mesh_path.with_suffix(".gltf")


def default_simplified_output_path(mesh_path: Path) -> Path:
    stem = mesh_path.stem
    if stem.endswith("_simplified"):
        return mesh_path.with_suffix(".glb")
    return mesh_path.with_name(f"{stem}_simplified.glb")


def choose_texture_atlas_layout(
    triangle_count: int,
    max_size: int,
    min_tile: int,
    max_tile: int,
) -> tuple[int, int, int, int, int]:
    triangle_count = max(1, int(triangle_count))
    max_size = max(64, int(max_size))
    min_tile = max(1, int(min_tile))
    max_tile = max(1, int(max_tile))
    if max_tile < min_tile:
        max_tile = min_tile

    cols = int(np.ceil(np.sqrt(triangle_count)))
    rows = int(np.ceil(triangle_count / cols))
    grid_dim = max(cols, rows)

    tile = max_size // max(1, grid_dim)
    tile = max(1, min(tile, max_tile))
    tile = max(tile, min_tile)
    if tile * grid_dim > max_size:
        tile = max(1, max_size // max(1, grid_dim))

    tex_w = cols * tile
    tex_h = rows * tile
    return cols, rows, tile, tex_w, tex_h


def build_vertex_color_texture_atlas(
    mesh: o3d.geometry.TriangleMesh,
    max_size: int,
    min_tile: int,
    max_tile: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if (
        not mesh.has_vertex_colors()
        or len(mesh.vertex_colors) != len(mesh.vertices)
        or len(mesh.triangles) == 0
    ):
        return None

    tris = np.asarray(mesh.triangles, dtype=np.int64)
    colors = np.clip(np.asarray(mesh.vertex_colors, dtype=np.float64), 0.0, 1.0)
    if len(tris) == 0 or len(colors) == 0:
        return None

    cols, rows, tile, tex_w, tex_h = choose_texture_atlas_layout(
        triangle_count=len(tris),
        max_size=max_size,
        min_tile=min_tile,
        max_tile=max_tile,
    )
    if tile < 3:
        return None

    pad = 1 if tile >= 6 else 0
    a = np.asarray((pad + 0.5, pad + 0.5), dtype=np.float32)
    b = np.asarray((tile - pad - 0.5, pad + 0.5), dtype=np.float32)
    c = np.asarray((pad + 0.5, tile - pad - 0.5), dtype=np.float32)
    if b[0] <= a[0] or c[1] <= a[1]:
        a = np.asarray((0.5, 0.5), dtype=np.float32)
        b = np.asarray((tile - 0.5, 0.5), dtype=np.float32)
        c = np.asarray((0.5, tile - 0.5), dtype=np.float32)

    ys, xs = np.mgrid[0:tile, 0:tile]
    px = xs.astype(np.float32) + 0.5
    py = ys.astype(np.float32) + 0.5

    denom = ((b[1] - c[1]) * (a[0] - c[0])) + ((c[0] - b[0]) * (a[1] - c[1]))
    if abs(float(denom)) < 1e-12:
        return None

    w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
    w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(mask):
        return None

    bary_weights = np.stack((w0[mask], w1[mask], w2[mask]), axis=1).astype(np.float32)
    uv_local = np.asarray((a, b, c), dtype=np.float64)

    texture = np.zeros((tex_h, tex_w, 3), dtype=np.float32)
    tri_uvs = np.zeros((len(tris), 3, 2), dtype=np.float64)
    tri_colors = colors[tris]

    for tri_idx in range(len(tris)):
        row = tri_idx // cols
        col = tri_idx % cols
        x_off = col * tile
        y_off = row * tile

        block = texture[y_off : y_off + tile, x_off : x_off + tile]
        tri_col = tri_colors[tri_idx].astype(np.float32, copy=False)
        block[mask] = bary_weights @ tri_col

        uv_global = uv_local + np.asarray((x_off, y_off), dtype=np.float64)
        tri_uvs[tri_idx, :, 0] = uv_global[:, 0] / float(tex_w)
        tri_uvs[tri_idx, :, 1] = 1.0 - (uv_global[:, 1] / float(tex_h))

    texture_u8 = np.clip(np.round(texture * 255.0), 0, 255).astype(np.uint8)
    return texture_u8, tri_uvs


def write_texture_image(path: Path, image_rgb_u8: np.ndarray) -> Path | None:
    if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
        return None

    ensure_parent(str(path))
    image_rgb_u8 = np.ascontiguousarray(image_rgb_u8, dtype=np.uint8)

    try:
        ok = o3d.io.write_image(str(path), o3d.geometry.Image(image_rgb_u8))
    except Exception:
        ok = False
    if ok:
        return path

    ppm_path = path.with_suffix(".ppm")
    try:
        ensure_parent(str(ppm_path))
        h, w = image_rgb_u8.shape[:2]
        with ppm_path.open("wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(image_rgb_u8.tobytes(order="C"))
        return ppm_path
    except OSError:
        return None


def write_obj_with_mtl(
    mesh: o3d.geometry.TriangleMesh,
    obj_path: Path,
    obj_texture_max_size: int = 4096,
    obj_texture_min_tile: int = 4,
    obj_texture_max_tile: int = 16,
) -> tuple[bool, Path, Path | None]:
    """Write OBJ + companion MTL.

    If vertex colors are present, this prefers baking them into a generated texture atlas
    (OBJ vt + MTL map_Kd). If that fails, it falls back to extended OBJ vertex colors
    (`v x y z r g b`), which are not part of strict OBJ+MTL texturing.
    """
    ensure_parent(str(obj_path))
    mtl_path = obj_path.with_suffix(".mtl")
    material_name = "material0"

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if len(verts) == 0 or len(tris) == 0:
        return False, mtl_path, None

    normals = None
    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    colors = None
    if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
        colors = np.clip(np.asarray(mesh.vertex_colors, dtype=np.float64), 0.0, 1.0)

    tri_uvs = None
    texture_map_path = None
    if colors is not None:
        atlas = build_vertex_color_texture_atlas(
            mesh,
            max_size=max(256, int(obj_texture_max_size)),
            min_tile=max(1, int(obj_texture_min_tile)),
            max_tile=max(1, int(obj_texture_max_tile)),
        )
        if atlas is not None:
            atlas_img, tri_uvs = atlas
            texture_map_path = write_texture_image(default_obj_texture_map_path(obj_path), atlas_img)
            if texture_map_path is None:
                tri_uvs = None

    mean_color = None
    if colors is not None and len(colors) > 0:
        mean_color = np.clip(np.mean(colors, axis=0), 0.0, 1.0)

    try:
        with obj_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Generated by fipmesh/tools/reconstruct_mesh.py\n")
            f.write(f"mtllib {mtl_path.name}\n")
            f.write("o mesh\n")
            f.write(f"usemtl {material_name}\n")

            if colors is not None and tri_uvs is None:
                for v, c in zip(verts, colors):
                    f.write(
                        f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g} "
                        f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"
                    )
            else:
                for v in verts:
                    f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")

            if normals is not None:
                for n in normals:
                    f.write(f"vn {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n")

            if tri_uvs is not None:
                for uv in tri_uvs.reshape(-1, 2):
                    f.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")

            # Open3D uses 0-based indices; OBJ is 1-based.
            for tri_idx, tri in enumerate(tris):
                a, b, c = (int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1)
                if tri_uvs is not None:
                    ta = tri_idx * 3 + 1
                    tb = tri_idx * 3 + 2
                    tc = tri_idx * 3 + 3
                    if normals is not None:
                        f.write(f"f {a}/{ta}/{a} {b}/{tb}/{b} {c}/{tc}/{c}\n")
                    else:
                        f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
                elif normals is not None:
                    f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
                else:
                    f.write(f"f {a} {b} {c}\n")

        with mtl_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Companion material for OBJ export\n")
            if tri_uvs is not None and texture_map_path is not None:
                f.write("# Color is baked into a generated texture atlas.\n")
            elif colors is not None:
                f.write("# NOTE: Mesh color is stored as vertex colors in OBJ `v` records.\n")
                f.write("# This MTL does not reference a baked texture image.\n")
            else:
                f.write("# NOTE: Mesh has no color data, so only scalar material values are written.\n")
            f.write(f"newmtl {material_name}\n")
            f.write("Ka 0.000000 0.000000 0.000000\n")
            if tri_uvs is not None and texture_map_path is not None:
                f.write("Kd 1.000000 1.000000 1.000000\n")
            elif mean_color is not None:
                f.write(f"Kd {mean_color[0]:.6f} {mean_color[1]:.6f} {mean_color[2]:.6f}\n")
            else:
                f.write("Kd 0.800000 0.800000 0.800000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("d 1.000000\n")
            f.write("illum 1\n")
            if tri_uvs is not None and texture_map_path is not None:
                f.write(f"map_Kd {texture_map_path.name}\n")
    except OSError:
        return False, mtl_path, texture_map_path

    return True, mtl_path, texture_map_path


def mesh_has_vertex_colors(mesh: o3d.geometry.TriangleMesh) -> bool:
    return mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices)


def write_mesh_file(
    mesh: o3d.geometry.TriangleMesh,
    path: Path,
    obj_texture_max_size: int = 4096,
    obj_texture_min_tile: int = 4,
    obj_texture_max_tile: int = 16,
) -> tuple[bool, Path | None, Path | None]:
    if path.suffix.lower() == ".obj":
        return write_obj_with_mtl(
            mesh,
            path,
            obj_texture_max_size=obj_texture_max_size,
            obj_texture_min_tile=obj_texture_min_tile,
            obj_texture_max_tile=obj_texture_max_tile,
        )

    ensure_parent(str(path))
    try:
        ok = o3d.io.write_triangle_mesh(
            str(path),
            mesh,
            write_ascii=True,
            write_vertex_normals=True,
            write_vertex_colors=mesh_has_vertex_colors(mesh),
            write_triangle_uvs=False,
        )
    except Exception:
        ok = False
    return bool(ok), None, None


def build_gltf_document(
    mesh: o3d.geometry.TriangleMesh,
    quantize_colors: bool = False,
) -> tuple[dict, bytes] | None:
    """Build a glTF JSON document (without a `buffers[].uri`) plus its binary buffer.

    Shared by both the JSON `.gltf` (base64-embedded) and binary `.glb` writers below.
    `quantize_colors` packs COLOR_0 as normalized unsigned bytes instead of float32,
    quartering that attribute's size -- worthwhile for the decimated web-viewer export
    where file size matters more than color precision.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.triangles, dtype=np.uint32)
    if len(verts) == 0 or len(tris) == 0:
        return None
    if not np.isfinite(verts).all():
        return None

    positions = np.ascontiguousarray(verts, dtype=np.float32)
    indices = np.ascontiguousarray(tris.reshape(-1), dtype=np.uint32)

    normals = None
    if mesh.has_vertex_normals() and len(mesh.vertex_normals) == len(mesh.vertices):
        normals_np = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if np.isfinite(normals_np).all():
            normals = np.ascontiguousarray(normals_np, dtype=np.float32)

    colors = None
    colors_are_quantized = False
    if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
        colors_np = np.clip(np.asarray(mesh.vertex_colors, dtype=np.float64), 0.0, 1.0)
        if np.isfinite(colors_np).all():
            alpha = np.ones((len(colors_np), 1), dtype=np.float64)
            rgba = np.concatenate((colors_np, alpha), axis=1)
            if quantize_colors:
                colors = np.ascontiguousarray(
                    np.round(rgba * 255.0).astype(np.uint8)
                )
                colors_are_quantized = True
            else:
                colors = np.ascontiguousarray(rgba.astype(np.float32))

    buffer = bytearray()
    buffer_views: list[dict[str, object]] = []
    accessors: list[dict[str, object]] = []

    def add_accessor(
        array: np.ndarray,
        accessor_type: str,
        component_type: int,
        target: int,
        min_vals: list[float] | None = None,
        max_vals: list[float] | None = None,
        normalized: bool = False,
    ) -> int:
        pad = (-len(buffer)) % 4
        if pad:
            # glTF buffers should stay aligned for float32 / uint32 accessors.
            buffer.extend(b"\x00" * pad)

        byte_offset = len(buffer)
        raw = np.ascontiguousarray(array).tobytes(order="C")
        buffer.extend(raw)

        buffer_view_index = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": byte_offset,
                "byteLength": len(raw),
                "target": target,
            }
        )

        count = int(array.shape[0]) if array.ndim > 1 else int(array.size)
        accessor: dict[str, object] = {
            "bufferView": buffer_view_index,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if normalized:
            accessor["normalized"] = True
        if min_vals is not None:
            accessor["min"] = min_vals
        if max_vals is not None:
            accessor["max"] = max_vals

        accessors.append(accessor)
        return len(accessors) - 1

    pos_accessor = add_accessor(
        positions,
        accessor_type="VEC3",
        component_type=5126,
        target=34962,
        min_vals=positions.min(axis=0).astype(np.float64).tolist(),
        max_vals=positions.max(axis=0).astype(np.float64).tolist(),
    )
    idx_accessor = add_accessor(
        indices,
        accessor_type="SCALAR",
        component_type=5125,
        target=34963,
        min_vals=[int(indices.min())],
        max_vals=[int(indices.max())],
    )

    attributes: dict[str, int] = {"POSITION": pos_accessor}
    if normals is not None:
        attributes["NORMAL"] = add_accessor(
            normals,
            accessor_type="VEC3",
            component_type=5126,
            target=34962,
        )
    if colors is not None:
        attributes["COLOR_0"] = add_accessor(
            colors,
            accessor_type="VEC4",
            component_type=5121 if colors_are_quantized else 5126,
            target=34962,
            normalized=colors_are_quantized,
        )

    material: dict[str, object] = {
        "name": "material0",
        "pbrMetallicRoughness": {
            "baseColorFactor": [1.0, 1.0, 1.0, 1.0]
            if colors is not None
            else [0.8, 0.8, 0.8, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 1.0,
        },
        "doubleSided": True,
    }

    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "fipmesh/tools/reconstruct_mesh.py",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "mesh"}],
        "meshes": [
            {
                "name": "mesh",
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": idx_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [material],
        "buffers": [{"byteLength": len(buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    return gltf, bytes(buffer)


def write_gltf(mesh: o3d.geometry.TriangleMesh, gltf_path: Path) -> bool:
    built = build_gltf_document(mesh)
    if built is None:
        return False
    gltf, buffer = built

    gltf["buffers"][0]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(buffer).decode("ascii")
    )

    ensure_parent(str(gltf_path))
    try:
        with gltf_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(gltf, f, indent=2)
            f.write("\n")
    except OSError:
        return False

    return True


def write_glb(
    mesh: o3d.geometry.TriangleMesh,
    glb_path: Path,
    quantize_colors: bool = True,
) -> bool:
    """Write a binary glTF (.glb): one self-contained file, no base64 inflation.

    Meant for the decimated web-viewer export, where small filesize matters more
    than the JSON-embeddable convenience of `.gltf`.
    """
    built = build_gltf_document(mesh, quantize_colors=quantize_colors)
    if built is None:
        return False
    gltf, buffer = built

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    json_bytes += b" " * json_pad

    bin_bytes = buffer
    bin_pad = (-len(bin_bytes)) % 4
    bin_bytes += b"\x00" * bin_pad

    total_length = 12 + (8 + len(json_bytes)) + (8 + len(bin_bytes))

    ensure_parent(str(glb_path))
    try:
        with glb_path.open("wb") as f:
            f.write(b"glTF")
            f.write((2).to_bytes(4, "little"))
            f.write(total_length.to_bytes(4, "little"))
            f.write(len(json_bytes).to_bytes(4, "little"))
            f.write(b"JSON")
            f.write(json_bytes)
            f.write(len(bin_bytes).to_bytes(4, "little"))
            f.write(b"BIN\x00")
            f.write(bin_bytes)
    except OSError:
        return False

    return True


def transfer_vertex_colors_from_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh | None:
    if len(mesh.vertices) == 0 or not pcd_has_colors(pcd):
        return None

    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64)
    if len(points) == 0 or len(colors) != len(points):
        return None

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    vcols = np.zeros((len(verts), 3), dtype=np.float64)

    for i, v in enumerate(verts):
        _, idx, _ = kdtree.search_knn_vector_3d(v, 1)
        if len(idx) > 0:
            vcols[i] = colors[int(idx[0])]

    out = copy.deepcopy(mesh)
    out.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcols, 0.0, 1.0))
    return out


def main() -> int:
    args = parse_args()

    t_start = time.time()
    _t_prev = [t_start]

    def _step(name: str) -> None:
        now = time.time()
        print(f"[step] {name}  (prev stage: {now - _t_prev[0]:.1f}s)")
        _t_prev[0] = now

    input_path = Path(args.input)
    output_path = Path(args.output)
    clean_cloud_path = Path(args.output_clean_cloud) if args.output_clean_cloud else None
    decimated_path = Path(args.output_decimated) if args.output_decimated else None
    export_all_variants = bool(args.export_all_variants)
    ply_output_path = default_ply_output_path(output_path)
    gltf_output_path = default_gltf_output_path(output_path)
    simplified_glb_path = default_simplified_output_path(output_path)
    gltf_decimated_path = (
        default_gltf_output_path(decimated_path) if decimated_path is not None else None
    )
    textured_output_path = default_textured_output_path(output_path)
    textured_output_obj_path = default_textured_obj_output_path(output_path)
    textured_decimated_path = (
        default_textured_output_path(decimated_path) if decimated_path is not None else None
    )
    textured_decimated_obj_path = (
        default_textured_obj_output_path(decimated_path) if decimated_path is not None else None
    )

    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    pcd_input = read_input_point_cloud(input_path)
    points_raw = np.asarray(pcd_input.points, dtype=np.float64)
    input_has_colors = pcd_has_colors(pcd_input)
    if len(points_raw) < 10:
        print(f"error: need at least 10 input points, got {len(points_raw)}", file=sys.stderr)
        return 3

    pcd, removed_sanitize = sanitize_point_cloud(pcd_input)
    points = np.asarray(pcd.points, dtype=np.float64)
    if len(points) < 10:
        print("error: too few points after dedupe/sanitize", file=sys.stderr)
        return 3

    max_input_points = int(args.max_input_points)
    if max_input_points > 0 and len(points) > max_input_points:
        _step("random downsample (point cap)")
        rng = np.random.default_rng(seed=0)
        keep_idx = rng.choice(len(points), size=max_input_points, replace=False)
        keep_idx.sort()
        pcd = pcd.select_by_index(keep_idx.tolist())
        points = np.asarray(pcd.points, dtype=np.float64)
        print(f"point cap target:     {max_input_points}")
        print(f"points after cap:     {len(points)}")

    _step("input")
    print(f"input points raw:    {len(points_raw)}")
    print(f"input points uniq:   {len(points)}")
    print(f"input colors:        {'yes' if input_has_colors else 'no'}")
    if removed_sanitize > 0:
        print(f"sanitize removed:    {removed_sanitize}")

    if args.voxel_size > 0:
        _step("voxel downsample")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"points after voxel:  {len(pcd.points)}")

    if args.outlier_std_ratio > 0 and args.outlier_nb_neighbors > 0:
        _step("statistical outlier removal")
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=max(3, int(args.outlier_nb_neighbors)),
            std_ratio=float(args.outlier_std_ratio),
        )
        print(f"points after SOR:    {len(pcd.points)}")
        if len(pcd.points) < 10:
            print("error: too few points after statistical outlier removal", file=sys.stderr)
            return 4

        save_snapshot(output_path, "01_statistical_outliers_removed", pcd)

    med_nn = median_nn_distance(pcd)
    if args.radius_outlier_nb_points > 0:
        radius = float(args.radius_outlier_radius)
        if radius <= 0 and med_nn > 0:
            radius = med_nn * max(0.5, float(args.radius_outlier_radius_factor))
        if radius > 0:
            _step("radius outlier removal")
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=max(2, int(args.radius_outlier_nb_points)),
                radius=radius,
            )
            print(f"radius outlier r:    {radius:.6f}")
            print(f"points after radius: {len(pcd.points)}")
            if len(pcd.points) < 10:
                print("error: too few points after radius outlier removal", file=sys.stderr)
                return 4
            med_nn = median_nn_distance(pcd)

            save_snapshot(output_path, "02_radius_outliers_removed", pcd)

    if args.crop_quantile > 0:
        _step("bbox quantile crop")
        pcd = pcd_from_points(quantile_crop_points(np.asarray(pcd.points), args.crop_quantile))
        print(f"crop quantile:       {args.crop_quantile:g}")
        print(f"points after crop:   {len(pcd.points)}")
        if len(pcd.points) < 10:
            print("error: too few points after crop", file=sys.stderr)
            return 4
        med_nn = median_nn_distance(pcd)

    dbscan_guard = int(args.dbscan_max_points)
    dbscan_allowed = (dbscan_guard <= 0) or (len(pcd.points) <= dbscan_guard)
    if args.dbscan_min_points > 0 and len(pcd.points) >= max(20, args.dbscan_min_points) and dbscan_allowed:
        db_eps = float(args.dbscan_eps)
        if db_eps <= 0 and med_nn > 0:
            db_eps = med_nn * max(1.0, float(args.dbscan_eps_factor))
        if db_eps > 0:
            _step("cluster cleanup (DBSCAN)")
            pcd, db_info = cluster_cleanup_point_cloud(
                pcd,
                eps=db_eps,
                min_points=max(2, int(args.dbscan_min_points)),
                keep_largest=max(0, int(args.dbscan_keep_largest)),
                min_cluster_ratio=float(np.clip(args.dbscan_min_cluster_ratio, 0.0, 1.0)),
            )
            print(f"dbscan eps:          {float(db_info['eps']):.6f}")
            print(
                "dbscan clusters:     "
                f"{int(db_info['clusters_kept'])}/{int(db_info['clusters_total'])} kept"
            )
            print(f"points after dbscan: {len(pcd.points)}")
            if len(pcd.points) < 10:
                print("error: too few points after DBSCAN cleanup", file=sys.stderr)
                return 4
            med_nn = median_nn_distance(pcd)
    elif args.dbscan_min_points > 0 and not dbscan_allowed:
        _step("cluster cleanup (DBSCAN)")
        print(
            "dbscan skipped:      "
            f"point count {len(pcd.points)} exceeds dbscan-max-points={dbscan_guard}"
        )

    mirror_applied = "none"
    mirror_pivot = np.nan
    auto_ratio = np.nan
    if args.mirror_backfill != "none":
        _step("optional mirror backfill")
        work_points = np.asarray(pcd.points)
        selected_axis = args.mirror_backfill
        if args.mirror_backfill == "auto":
            axis, ratio = pick_auto_mirror_axis(work_points)
            auto_ratio = ratio
            if ratio < float(args.mirror_auto_threshold):
                selected_axis = axis
            else:
                selected_axis = "none"
        if selected_axis in ("x", "y", "z"):
            mirrored_points, pivot = apply_mirror_backfill(work_points, selected_axis)
            pcd = pcd_from_points(mirrored_points)
            mirror_applied = selected_axis
            mirror_pivot = pivot

    normal_radius = float(args.normal_radius)
    if normal_radius <= 0:
        normal_radius = auto_normal_radius(np.asarray(pcd.points))
    normal_radius = max(normal_radius, 1e-6)

    camera_centers = load_camera_centers(args.camera_centers)
    side1_camera_centers = load_camera_centers(args.side1_camera_centers)

    _step("estimate + orient normals")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=max(10, int(args.normal_max_nn)),
        )
    )
    if camera_centers:
        orient_normals_towards_cameras(pcd, camera_centers)
        normal_orient_mode = "camera_positions"
    else:
        try:
            pcd.orient_normals_consistent_tangent_plane(max(5, int(args.normal_orient_k)))
            normal_orient_mode = "consistent_tangent_plane"
        except RuntimeError:
            orient_normals_outward_from_centroid(pcd)
            normal_orient_mode = "centroid_outward_fallback"
    print(f"normal radius:       {normal_radius:.6f}")
    print(f"normal orient mode:  {normal_orient_mode}")

    if clean_cloud_path is not None:
        _step("write cleaned cloud")
        ensure_parent(str(clean_cloud_path))
        ok_cloud = o3d.io.write_point_cloud(str(clean_cloud_path), pcd, write_ascii=True)
        if not ok_cloud:
            print(f"warning: failed to write clean cloud to {clean_cloud_path}", file=sys.stderr)

    _step("poisson reconstruction")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=max(5, int(args.poisson_depth)),
        scale=max(1.0, float(args.poisson_scale)),
        linear_fit=bool(args.poisson_linear_fit),
    )
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("error: poisson reconstruction produced empty mesh", file=sys.stderr)
        return 5

    save_snapshot(output_path, "03_poisson_raw", mesh)

    densities_np = np.asarray(densities)
    density_q = float(np.clip(args.density_trim_quantile, 0.0, 0.99))
    if density_q > 0 and len(densities_np) == len(mesh.vertices) and len(densities_np) > 0:
        _step("density trim")
        density_cut = float(np.quantile(densities_np, density_q))
        mesh.remove_vertices_by_mask(densities_np < density_cut)
        print(f"density q:           {density_q:.4f}")
        print(f"density cut:         {density_cut:.6f}")

    if len(pcd.points) > 0 and args.poisson_crop_scale > 0:
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb = aabb.scale(max(1.0, float(args.poisson_crop_scale)), aabb.get_center())
        mesh = mesh.crop(aabb)

    _step("mesh cleanup")
    mesh = cleanup_mesh_topology(mesh)
    mesh = keep_significant_mesh_components(
        mesh,
        min_ratio=float(np.clip(args.component_min_ratio, 0.0, 1.0)),
        min_triangles=max(0, int(args.component_min_triangles)),
        max_count=max(0, int(args.component_max_count)),
    )
    mesh = cleanup_mesh_topology(mesh)

    if args.fill_holes and len(mesh.triangles) > 0:
        hole_size = float(args.fill_holes_max_size)
        if hole_size <= 0 and len(pcd.points) > 0:
            diag = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))
            hole_size = diag * max(0.0, float(args.fill_holes_max_size_ratio))
        if hole_size > 0:
            _step("fill mesh holes")
            verts_before, tris_before = len(mesh.vertices), len(mesh.triangles)
            mesh, fill_passes_run = fill_mesh_holes(
                mesh, hole_size, max_passes=max(1, int(args.fill_holes_passes))
            )
            mesh = cleanup_mesh_topology(mesh)
            print(f"hole fill max radius: {hole_size:.6f}")
            print(f"hole fill passes:    {fill_passes_run}/{max(1, int(args.fill_holes_passes))}")
            print(f"vertices added:      {len(mesh.vertices) - verts_before}")
            print(f"triangles added:     {len(mesh.triangles) - tris_before}")
            save_snapshot(output_path, "03b_holes_filled", mesh)

    if args.smooth_iters > 0 and len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=max(1, int(args.smooth_iters)))
        mesh = cleanup_mesh_topology(mesh)

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        print("error: mesh became empty after cleanup", file=sys.stderr)
        return 6

    _step("mesh normal orientation check")
    mesh, mesh_winding_flipped = orient_mesh_winding_outward(mesh)
    print(f"mesh winding flipped: {'yes' if mesh_winding_flipped else 'no'}")
    mesh.compute_vertex_normals()

    if args.normalize_pose and len(mesh.vertices) > 0:
        _step("normalize pose (center + flatten cross-section)")
        pose_transform = compute_pca_pose(np.asarray(mesh.vertices))
        mesh.transform(pose_transform)
        pcd.transform(pose_transform)
        print("pose:                centered at origin, largest cross-section in XY plane")
        if side1_camera_centers:
            flipped = orient_up_towards_side1(mesh, pcd, pose_transform, side1_camera_centers)
            print(f"pose side1 up:       {'flipped 180deg' if flipped else 'already up'}")

    decimated_mesh = None
    if (
        export_all_variants
        and args.decimate_target_triangles > 0
        and len(mesh.triangles) > args.decimate_target_triangles
    ):
        _step("optional decimation")
        decimated_mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=max(4, int(args.decimate_target_triangles))
        )
        decimated_mesh = cleanup_mesh_topology(decimated_mesh)
        if len(decimated_mesh.vertices) > 0 and len(decimated_mesh.triangles) > 0:
            decimated_mesh, dec_winding_flipped = orient_mesh_winding_outward(decimated_mesh)
            if dec_winding_flipped:
                print("decimated winding:   flipped")
            decimated_mesh.compute_vertex_normals()
        else:
            print("warning: decimation produced empty mesh, skipping output", file=sys.stderr)
            decimated_mesh = None

    texture_cmd = args.texture_command or os.environ.get("FIPMESH_TEXTURE_CMD", "")
    textured_mesh = None
    textured_decimated_mesh = None
    primary_output_path_written = None
    primary_output_mtl_path = None
    primary_output_texture_path = None
    ply_output_path_written = None
    textured_obj_path = None
    textured_obj_mtl_path = None
    textured_obj_texture_path = None
    decimated_output_path_written = None
    textured_decimated_obj_path_written = None
    textured_decimated_mtl_path = None
    textured_decimated_texture_path = None
    gltf_path_written = None
    gltf_decimated_path_written = None
    if pcd_has_colors(pcd):
        _step("textured output (vertex colors)")
        textured_mesh = transfer_vertex_colors_from_point_cloud(mesh, pcd)
        if textured_mesh is None:
            print(
                "warning: failed to transfer vertex colors onto reconstructed mesh",
                file=sys.stderr,
            )
        if export_all_variants and decimated_mesh is not None and textured_decimated_path is not None:
            textured_decimated_mesh = transfer_vertex_colors_from_point_cloud(decimated_mesh, pcd)
    else:
        _step("textured output (vertex colors)")
        print("textured output:     skipped (input/clean cloud has no colors)")

    _step("write mesh exports")
    primary_mesh = textured_mesh if textured_mesh is not None else mesh
    ok_primary, obj_mtl, obj_tex = write_mesh_file(
        primary_mesh,
        output_path,
        obj_texture_max_size=int(args.obj_texture_max_size),
        obj_texture_min_tile=int(args.obj_texture_min_tile),
        obj_texture_max_tile=int(args.obj_texture_max_tile),
    )
    if not ok_primary:
        print(f"error: failed to write mesh to {output_path}", file=sys.stderr)
        return 7
    primary_output_path_written = output_path
    primary_output_mtl_path = obj_mtl
    primary_output_texture_path = obj_tex

    if ply_output_path == output_path:
        ply_output_path_written = output_path
    else:
        ok_ply, _, _ = write_mesh_file(primary_mesh, ply_output_path)
        if ok_ply:
            ply_output_path_written = ply_output_path
        else:
            print(f"warning: failed to write PLY mesh to {ply_output_path}", file=sys.stderr)

    if gltf_output_path == output_path:
        gltf_path_written = output_path
    elif write_gltf(primary_mesh, gltf_output_path):
        gltf_path_written = gltf_output_path
    else:
        print(f"warning: failed to write glTF mesh to {gltf_output_path}", file=sys.stderr)

    simplified_output_path_written = None
    simplified_vertex_count = 0
    simplified_triangle_count = 0
    simplified_target_vertices = int(args.simplified_target_vertices)
    if simplified_target_vertices > 0 and len(primary_mesh.triangles) > 0:
        _step("simplified export (for web viewer)")
        # A closed 2-manifold triangle mesh has roughly twice as many faces as
        # vertices (Euler's formula), so this converts the requested vertex budget
        # into the triangle-count target quadric decimation actually takes.
        target_triangles = max(4, int(round(simplified_target_vertices * 2)))
        if len(primary_mesh.triangles) > target_triangles:
            simplified_mesh = primary_mesh.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles
            )
        else:
            simplified_mesh = copy.deepcopy(primary_mesh)
        simplified_mesh = cleanup_mesh_topology(simplified_mesh)
        if len(simplified_mesh.vertices) > 0 and len(simplified_mesh.triangles) > 0:
            simplified_mesh, simplified_winding_flipped = orient_mesh_winding_outward(simplified_mesh)
            simplified_mesh.compute_vertex_normals()
            if write_glb(simplified_mesh, simplified_glb_path):
                simplified_output_path_written = simplified_glb_path
                simplified_vertex_count = len(simplified_mesh.vertices)
                simplified_triangle_count = len(simplified_mesh.triangles)
                if simplified_winding_flipped:
                    print("simplified winding:  flipped")
            else:
                print(
                    f"warning: failed to write simplified glb to {simplified_glb_path}",
                    file=sys.stderr,
                )
        else:
            print("warning: simplified decimation produced empty mesh, skipping", file=sys.stderr)

    if export_all_variants and decimated_mesh is not None and decimated_path is not None:
        ensure_parent(str(decimated_path))
        ok_dec = o3d.io.write_triangle_mesh(
            str(decimated_path),
            decimated_mesh,
            write_ascii=True,
            write_vertex_normals=True,
            write_vertex_colors=False,
            write_triangle_uvs=False,
        )
        if ok_dec:
            decimated_output_path_written = decimated_path
        else:
            print(f"warning: failed to write decimated mesh to {decimated_path}", file=sys.stderr)

    if export_all_variants and textured_mesh is not None:
        ensure_parent(str(textured_output_path))
        ok_tex = o3d.io.write_triangle_mesh(
            str(textured_output_path),
            textured_mesh,
            write_ascii=True,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=False,
        )
        if not ok_tex:
            print(
                f"warning: failed to write textured mesh to {textured_output_path}",
                file=sys.stderr,
            )
        else:
            ok_obj_tex, obj_mtl, obj_tex = write_obj_with_mtl(
                textured_mesh,
                textured_output_obj_path,
                obj_texture_max_size=int(args.obj_texture_max_size),
                obj_texture_min_tile=int(args.obj_texture_min_tile),
                obj_texture_max_tile=int(args.obj_texture_max_tile),
            )
            if ok_obj_tex:
                textured_obj_path = textured_output_obj_path
                textured_obj_mtl_path = obj_mtl
                textured_obj_texture_path = obj_tex
                if obj_tex is None:
                    print(
                        "warning: textured OBJ exported without map_Kd; "
                        "falling back to vertex-color OBJ records",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"warning: failed to write textured OBJ/MTL to {textured_output_obj_path}",
                    file=sys.stderr,
                )

    if (
        export_all_variants
        and decimated_mesh is not None
        and textured_decimated_mesh is not None
        and textured_decimated_path is not None
    ):
        ensure_parent(str(textured_decimated_path))
        ok_tex_dec = o3d.io.write_triangle_mesh(
            str(textured_decimated_path),
            textured_decimated_mesh,
            write_ascii=True,
            write_vertex_normals=True,
            write_vertex_colors=True,
            write_triangle_uvs=False,
        )
        if not ok_tex_dec:
            print(
                "warning: failed to write textured decimated mesh to "
                f"{textured_decimated_path}",
                file=sys.stderr,
            )
        elif textured_decimated_obj_path is not None:
            ok_obj_tex_dec, obj_dec_mtl, obj_dec_tex = write_obj_with_mtl(
                textured_decimated_mesh,
                textured_decimated_obj_path,
                obj_texture_max_size=int(args.obj_texture_max_size),
                obj_texture_min_tile=int(args.obj_texture_min_tile),
                obj_texture_max_tile=int(args.obj_texture_max_tile),
            )
            if ok_obj_tex_dec:
                textured_decimated_obj_path_written = textured_decimated_obj_path
                textured_decimated_mtl_path = obj_dec_mtl
                textured_decimated_texture_path = obj_dec_tex
                if obj_dec_tex is None:
                    print(
                        "warning: textured decimated OBJ exported without map_Kd; "
                        "falling back to vertex-color OBJ records",
                        file=sys.stderr,
                    )
            else:
                print(
                    "warning: failed to write textured decimated OBJ/MTL to "
                    f"{textured_decimated_obj_path}",
                    file=sys.stderr,
                )

    if export_all_variants and decimated_mesh is not None and gltf_decimated_path is not None:
        gltf_decimated_source = (
            textured_decimated_mesh if textured_decimated_mesh is not None else decimated_mesh
        )
        if write_gltf(gltf_decimated_source, gltf_decimated_path):
            gltf_decimated_path_written = gltf_decimated_path
        else:
            print(
                f"warning: failed to write decimated glTF mesh to {gltf_decimated_path}",
                file=sys.stderr,
            )

    _step("optional texture hook")
    texture_rc = run_texture_hook(
        texture_cmd,
        output_path,
        decimated_output_path_written,
        input_path,
    )
    if texture_cmd and texture_rc != 0:
        print(f"warning: texture hook exited with code {texture_rc}", file=sys.stderr)
    _step("done")

    print(f"points after clean:  {len(pcd.points)}")
    if args.mirror_backfill == "auto":
        print(f"mirror auto ratio:   {auto_ratio:.4f} (threshold={args.mirror_auto_threshold:g})")
    print(
        "mirror applied:      "
        f"{mirror_applied}"
        + (f" (pivot={mirror_pivot:.6f})" if mirror_applied != "none" else "")
    )
    print(
        "component filter:    "
        f"min_ratio={args.component_min_ratio:g}, "
        f"min_triangles={args.component_min_triangles}, "
        f"max_count={args.component_max_count}"
    )
    print(f"mesh vertices:       {len(mesh.vertices)}")
    print(f"mesh faces:          {len(mesh.triangles)}")
    if export_all_variants and decimated_mesh is not None:
        print(f"decimated faces:     {len(decimated_mesh.triangles)}")
    elif (
        export_all_variants
        and args.decimate_target_triangles > 0
        and len(mesh.triangles) <= args.decimate_target_triangles
    ):
        print("decimation:          skipped (mesh already below target)")
    if clean_cloud_path is not None:
        print(f"clean cloud output:  {clean_cloud_path}")
    if primary_output_path_written is not None:
        print(f"output:              {primary_output_path_written}")
    if primary_output_mtl_path is not None:
        print(f"mtl output:          {primary_output_mtl_path}")
    if primary_output_texture_path is not None:
        print(f"map output:          {primary_output_texture_path}")
    if ply_output_path_written is not None and ply_output_path_written != primary_output_path_written:
        print(f"ply output:          {ply_output_path_written}")
    if gltf_path_written is not None:
        print(f"gltf output:         {gltf_path_written}")
    if simplified_output_path_written is not None:
        print(f"simplified verts:    {simplified_vertex_count}")
        print(f"simplified tris:     {simplified_triangle_count}")
        print(f"simplified glb:      {simplified_output_path_written}")
    elif simplified_target_vertices > 0:
        print("simplified glb:      skipped (see warning above)")
    if export_all_variants and decimated_output_path_written is not None:
        print(f"decimated output:    {decimated_output_path_written}")
    if export_all_variants and gltf_decimated_path_written is not None:
        print(f"decimated gltf:      {gltf_decimated_path_written}")
    if export_all_variants and textured_mesh is not None:
        print(f"textured output:     {textured_output_path}")
    if export_all_variants and textured_obj_path is not None and textured_obj_mtl_path is not None:
        print(f"textured obj output: {textured_obj_path}")
        print(f"textured mtl output: {textured_obj_mtl_path}")
        if textured_obj_texture_path is not None:
            print(f"textured map output: {textured_obj_texture_path}")
    if export_all_variants and textured_decimated_mesh is not None and textured_decimated_path is not None:
        print(f"textured decimated:  {textured_decimated_path}")
    if (
        export_all_variants
        and textured_decimated_mesh is not None
        and textured_decimated_obj_path_written is not None
        and textured_decimated_mtl_path is not None
    ):
        print(f"textured dec obj:    {textured_decimated_obj_path_written}")
        print(f"textured dec mtl:    {textured_decimated_mtl_path}")
        if textured_decimated_texture_path is not None:
            print(f"textured dec map:    {textured_decimated_texture_path}")
    if texture_cmd:
        print(f"texture hook rc:     {texture_rc}")
    else:
        print("texture hook:        skipped (no command configured)")
    print(f"total time:          {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
