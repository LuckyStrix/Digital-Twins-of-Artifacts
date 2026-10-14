#!/usr/bin/env python3
"""Stage 2 driver: a cross-platform Python port of run.sh + src/main.sh.

run.sh needs bash, nproc and /proc/meminfo, so it only runs on Linux/WSL.
This script does the same job with the same options and FIPMESH_* env vars,
so the app can run Stage 2 on native Windows (no WSL, no bash):

  1. resolve the COLMAP binary (FIPMESH_COLMAP_BIN → ./colmap_local →
     ./colmap/COLMAP.bat (Windows zip layout) → colmap on PATH),
  2. strip image metadata with exiftool,
  3. pick thread/cache defaults from the CPU count and total RAM,
  4. run src/run_colmap_mvs.py to build <out>/fused.ply,
  5. run src/reconstruct_mesh.py on it (skipped when FIPMESH_SKIP_RECON=1,
     which the app always sets: it meshes in its own Stage 4).

Usage (same flags as run.sh):
    python src/run_stage2.py -i <image_dir> [-s <secondary_dir>] [-o <out_dir>]
                             [-m <mask_dir>] [-n <secondary_mask_dir>] [-v]

Unlike run.sh, -i/-s/-o/-m/-n may be absolute paths (run.sh always prepends
its own folder). Relative paths are taken relative to the modelingPipeline
folder, as in run.sh. Absolute paths matter on Windows: a path on another
drive (D:\\scans) has no relative form from C:.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROGRAM = "run_stage2.py"
WINDOWS = sys.platform == "win32"

# Same mesh settings src/main.sh passes to reconstruct_mesh.py.
RECON_ARGS = [
    "--max-input-points", "0",
    "--dbscan-max-points", "0",
    "--outlier-nb-neighbors", "32",
    "--outlier-std-ratio", "0",
    "--radius-outlier-nb-points", "0",
    "--radius-outlier-radius-factor", "2.2",
    "--dbscan-min-points", "0",
    "--dbscan-eps-factor", "2.2",
    "--dbscan-keep-largest", "1",
    "--dbscan-min-cluster-ratio", "0.02",
    "--normal-max-nn", "96",
    "--normal-orient-k", "64",
    "--poisson-depth", "10",
    "--poisson-linear-fit",
    "--density-trim-quantile", "0.03",
    "--component-min-ratio", "0.01",
    "--component-min-triangles", "1000",
    "--component-max-count", "2",
    "--smooth-iters", "1",
    "--decimate-target-triangles", "300000",
]


class Stage2Error(Exception):
    pass


def status(msg: str) -> None:
    print(f"{PROGRAM}: \x1b[32mstatus\x1b[0m: {msg}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"{PROGRAM}: \x1b[33mwarning\x1b[0m: {msg}", file=sys.stderr, flush=True)


# ── binaries ──────────────────────────────────────────────────────────────────

def resolve_colmap(env: dict) -> str | None:
    """COLMAP to use, in run.sh's order plus the Windows zip layout."""
    if env.get("FIPMESH_COLMAP_BIN"):
        return env["FIPMESH_COLMAP_BIN"]
    local = SCRIPT_DIR / "colmap_local"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    if WINDOWS:
        # The official Windows release unzips to a folder holding COLMAP.bat,
        # which puts the bundled DLLs on PATH before starting bin\colmap.exe.
        bat = SCRIPT_DIR / "colmap" / "COLMAP.bat"
        if bat.is_file():
            return str(bat)
    return shutil.which("colmap")


def resolve_exiftool(env: dict) -> str | None:
    if env.get("FIPMESH_EXIFTOOL_BIN"):
        return env["FIPMESH_EXIFTOOL_BIN"]
    found = shutil.which("exiftool")
    if found:
        return found
    if WINDOWS:
        # exiftool's Windows zip ships "exiftool(-k).exe", which waits for a
        # keypress when it finishes, so it must be renamed to exiftool.exe.
        for cand in (SCRIPT_DIR / "tools" / "exiftool.exe",
                     SCRIPT_DIR / "exiftool" / "exiftool.exe"):
            if cand.is_file():
                return str(cand)
    return None


# ── resource defaults (run.sh's nproc / MemTotal tiers) ───────────────────────

def total_mem_gb() -> int:
    """Total physical RAM in GB, rounded up; 0 if unknown."""
    try:
        if WINDOWS:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return 0
            total_kb = stat.ullTotalPhys // 1024
        else:
            total_kb = 0
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                        break
    except Exception:
        return 0
    return (total_kb + 1048575) // 1048576


def resource_defaults(cpus: int, mem_gb: int) -> dict:
    """Thread and cache defaults, tiered by RAM exactly as in run.sh."""
    cpus = max(1, cpus)
    extract = match = mapper = fusion = cpus
    patch_cache = fusion_cache = 64
    tiers = [  # (max GB, extract/match cap, mapper/fusion cap, cache GB)
        (18, 6, 8, 12),
        (24, 8, 10, 16),
        (32, 10, 12, 24),
    ]
    if mem_gb > 0:
        for max_gb, em_cap, mf_cap, cache in tiers:
            if mem_gb <= max_gb:
                extract = match = min(cpus, em_cap)
                mapper = fusion = min(cpus, mf_cap)
                patch_cache = fusion_cache = cache
                break
    return {
        "FIPMESH_COLMAP_EXTRACT_THREADS": extract,
        "FIPMESH_COLMAP_MATCH_THREADS": match,
        "FIPMESH_COLMAP_MAPPER_THREADS": mapper,
        "FIPMESH_COLMAP_FUSION_THREADS": fusion,
        "FIPMESH_COLMAP_PATCH_CACHE_SIZE": patch_cache,
        "FIPMESH_COLMAP_FUSION_CACHE_SIZE": fusion_cache,
    }


def apply_defaults(env: dict, secondary: bool, cpus: int, mem_gb: int) -> None:
    """Fill every FIPMESH_COLMAP_* setting the caller didn't set (run.sh's
    ${VAR:-default} block)."""
    defaults: dict[str, object] = {
        "FIPMESH_COLMAP_QUALITY": "high",
        "FIPMESH_COLMAP_USE_GPU": "1",
        "FIPMESH_COLMAP_GPU_INDEX": "-1",
        **resource_defaults(cpus, mem_gb),
        "FIPMESH_COLMAP_FUSION_USE_CACHE": "1",
        "FIPMESH_COLMAP_IMAGE_SCALE": "1",
        "FIPMESH_COLMAP_IMAGE_STRIDE": "3",
        "FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES": "16000",
        "FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD": "0.0045",
        "FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD": "12",
        "FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING": "1",
        "FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE": "0",
        "FIPMESH_COLMAP_MATCH_GUIDED": "1",
        "FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES": "65536",
    }
    if secondary:
        defaults.update({
            "FIPMESH_COLMAP_SECONDARY_ROTATE_DEG": "180",
            "FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS": "primary_frame_x",
            "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_X": "0",
            "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Y": "0",
            "FIPMESH_COLMAP_SECONDARY_EXTRA_ROTATE_Z": "0",
            "FIPMESH_COLMAP_SECONDARY_TRANSLATE_X": "0",
            "FIPMESH_COLMAP_SECONDARY_TRANSLATE_Y": "0",
            "FIPMESH_COLMAP_SECONDARY_TRANSLATE_Z": "0",
            "FIPMESH_COLMAP_SECONDARY_ALIGN_MODE": "auto",
        })
    for key, val in defaults.items():
        if not env.get(key):
            env[key] = str(val)


# ── steps ─────────────────────────────────────────────────────────────────────

def resolve_dir(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else SCRIPT_DIR / path


def strip_metadata(exiftool: str, img_dir: Path) -> None:
    rc = subprocess.run(
        [exiftool, "-overwrite_original", "-all:all=", "-r", str(img_dir)],
        stdout=subprocess.DEVNULL,
    ).returncode
    if rc != 0:
        raise Stage2Error(f"exiftool failed with exit code {rc}")
    status(f"exiftool metadata stripping succeeded for {img_dir}")


def run_mvs(p_imgdir: Path, s_imgdir: Path | None, out_dir: Path,
            dense_cloud: Path, env: dict) -> None:
    if env.get("FIPMESH_SKIP_MVS", "0") == "1":
        raise Stage2Error("skipping MVS pipeline (FIPMESH_SKIP_MVS=1); "
                          "legacy sparse triangulation disabled")
    status("running COLMAP MVS pipeline to densify cloud")
    cmd = [sys.executable, str(SCRIPT_DIR / "src" / "run_colmap_mvs.py"),
           "--images", str(p_imgdir)]
    if s_imgdir is not None:
        cmd += ["--images-secondary", str(s_imgdir)]
    cmd += ["--workspace", str(out_dir),
            "--dense-cloud-out", str(dense_cloud),
            "--single-camera-per-folder", "1",
            "--clean-workspace"]
    if subprocess.run(cmd, cwd=SCRIPT_DIR, env=env).returncode != 0:
        raise Stage2Error("COLMAP MVS pipeline failed")
    status(f"MVS dense cloud ready: {dense_cloud}")


def run_recon(dense_cloud: Path, out_dir: Path, env: dict) -> None:
    if env.get("FIPMESH_SKIP_RECON", "0") == "1":
        status("skipping reconstruction pipeline (FIPMESH_SKIP_RECON=1)")
        return
    out_mesh = out_dir / "recon_mesh_recon.obj"
    status("running Open3D reconstruction pipeline")
    cmd = [sys.executable, str(SCRIPT_DIR / "src" / "reconstruct_mesh.py"),
           "--input", str(dense_cloud),
           "--output", str(out_mesh),
           "--output-clean-cloud", str(out_dir / "clean_cloud.ply"),
           "--output-decimated", str(out_dir / "recon_mesh_recon_decimated.obj"),
           *RECON_ARGS]
    if subprocess.run(cmd, cwd=SCRIPT_DIR, env=env).returncode != 0:
        raise Stage2Error("reconstruction pipeline failed")
    status(f"reconstruction pipeline complete: {out_mesh}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog=PROGRAM, description=__doc__.split("\n")[0])
    ap.add_argument("-i", dest="primary", required=True, help="primary image directory")
    ap.add_argument("-s", dest="secondary", help="secondary image directory")
    ap.add_argument("-o", dest="out", default="out", help="output directory")
    ap.add_argument("-m", dest="mask", help="mask directory for -i")
    ap.add_argument("-n", dest="mask_secondary", help="mask directory for -s")
    ap.add_argument("-v", dest="verbose", action="store_true")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, env: dict) -> None:
    p_imgdir = resolve_dir(args.primary)
    s_imgdir = resolve_dir(args.secondary) if args.secondary else None
    out_dir = resolve_dir(args.out)

    if not p_imgdir.is_dir():
        raise Stage2Error(f'invalid primary image directory "{p_imgdir}"')
    if s_imgdir is not None and not s_imgdir.is_dir():
        raise Stage2Error(f'invalid secondary image directory "{s_imgdir}"')
    if args.mask:
        mask = resolve_dir(args.mask)
        if not mask.is_dir():
            raise Stage2Error(f'invalid primary mask directory "{mask}"')
        env["FIPMESH_COLMAP_MASK_PATH"] = str(mask)
    if args.mask_secondary:
        if s_imgdir is None:
            raise Stage2Error("-n given without -s")
        mask = resolve_dir(args.mask_secondary)
        if not mask.is_dir():
            raise Stage2Error(f'invalid secondary mask directory "{mask}"')
        env["FIPMESH_COLMAP_MASK_PATH_SECONDARY"] = str(mask)
    out_dir.mkdir(parents=True, exist_ok=True)

    colmap = resolve_colmap(env)
    exiftool = resolve_exiftool(env)
    missing = [name for name, path in (("colmap", colmap), ("exiftool", exiftool))
               if not path]
    if missing:
        raise Stage2Error(
            f"missing dependencies: {', '.join(missing)} "
            "(run `python check_setup.py` for details)")
    env["FIPMESH_COLMAP_BIN"] = colmap

    strip_metadata(exiftool, p_imgdir)
    if s_imgdir is not None:
        strip_metadata(exiftool, s_imgdir)

    mem_gb = total_mem_gb()
    apply_defaults(env, s_imgdir is not None, os.cpu_count() or 4, mem_gb)
    if s_imgdir is not None:
        env["FIPMESH_COLMAP_IMAGES_SECONDARY"] = str(s_imgdir)

    if args.verbose:
        status(f"using COLMAP: {colmap}")
        status(f"COLMAP quality: {env['FIPMESH_COLMAP_QUALITY']} "
               f"(use_gpu={env['FIPMESH_COLMAP_USE_GPU']})")
        status(f"COLMAP input_image_scale: {env['FIPMESH_COLMAP_IMAGE_SCALE']}")
        status(f"COLMAP input_image_stride: {env['FIPMESH_COLMAP_IMAGE_STRIDE']}")
        status(f"COLMAP gpu_index: {env['FIPMESH_COLMAP_GPU_INDEX']}")
        if mem_gb > 0:
            status(f"memory detected: {mem_gb}GB")
        status("COLMAP threads: extract={} match={} mapper={} fusion={}".format(
            env["FIPMESH_COLMAP_EXTRACT_THREADS"], env["FIPMESH_COLMAP_MATCH_THREADS"],
            env["FIPMESH_COLMAP_MAPPER_THREADS"], env["FIPMESH_COLMAP_FUSION_THREADS"]))
        status("COLMAP cache: patch={}GB fusion={}GB (fusion_use_cache={})".format(
            env["FIPMESH_COLMAP_PATCH_CACHE_SIZE"], env["FIPMESH_COLMAP_FUSION_CACHE_SIZE"],
            env["FIPMESH_COLMAP_FUSION_USE_CACHE"]))
        status("COLMAP sparse detect: sift_max_features={} peak={} edge={} dsp={} affine={}".format(
            env["FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES"], env["FIPMESH_COLMAP_SIFT_PEAK_THRESHOLD"],
            env["FIPMESH_COLMAP_SIFT_EDGE_THRESHOLD"],
            env["FIPMESH_COLMAP_SIFT_DOMAIN_SIZE_POOLING"],
            env["FIPMESH_COLMAP_SIFT_ESTIMATE_AFFINE_SHAPE"]))
        status("COLMAP sparse match: guided={} max_num_matches={}".format(
            env["FIPMESH_COLMAP_MATCH_GUIDED"], env["FIPMESH_COLMAP_MATCH_MAX_NUM_MATCHES"]))
    status(f"primary image directory: {p_imgdir}")
    if s_imgdir is not None:
        status(f"secondary image directory: {s_imgdir}")
    for key, label in (("FIPMESH_COLMAP_MASK_PATH", "primary"),
                       ("FIPMESH_COLMAP_MASK_PATH_SECONDARY", "secondary")):
        if env.get(key):
            status(f"{label} mask directory: {env[key]}")
    status(f"output directory: {out_dir}")

    dense_cloud = out_dir / "fused.ply"
    dense_cloud.touch()
    run_mvs(p_imgdir, s_imgdir, out_dir, dense_cloud, env)
    status(f"using dense MVS cloud for meshing: {dense_cloud}")
    run_recon(dense_cloud, out_dir, env)

    if s_imgdir is not None:
        status(f"successfully created initial cloud from {p_imgdir} and {s_imgdir}")
    else:
        status(f"successfully created initial cloud from {p_imgdir}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        run(args, env)
    except Stage2Error as exc:
        print(f"{PROGRAM}: \x1b[31merror\x1b[0m: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
