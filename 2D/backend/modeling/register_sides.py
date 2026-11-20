#!/usr/bin/env python3
"""
register_sides.py
=================
Align the back of a two-sided scan (side2) to its front (side1).

The two sides are photographed separately, with the sheet flipped over by hand
in between, so side2 lands at an arbitrary position and rotation on the stage.
The website shows them as one object: "Flip to other side" turns the sheet 180°
about its vertical axis and swaps side1.glb for side2.glb at the edge-on
midpoint. For that to read as one sheet turning over, side2's outline has to sit
exactly where side1's outline appears from behind - i.e. side1's outline
mirrored left-right. This script finds the rigid motion (rotation +
translation; same camera, so no scale) that puts side2's alpha mask there and
re-samples all of side2's render-ready maps with it, onto side1's canvas size so
both models also get identical plane dimensions.

  side1/maps/   untouched (the reference)
  side2/maps/   *_render.tiff replaced by the aligned versions
                unregistered/          the originals from the modeling pipeline
                registration.json      transform, rotation, overlap score
                registration_overlay.png   check image: red = side1 seen from
                                       behind, green = aligned side2, yellow =
                                       both. A good result is almost all yellow.

Re-running is safe. registration.json records a fingerprint (size + modified
time) of every map this script wrote and of side1's mask it aligned to:
  - side2's maps still match -> they are this script's output; it starts again
    from unregistered/ (or does nothing if side1 has not changed either);
  - they do not match -> modeling rewrote them; they become the new originals;
  - side1's mask changed -> side2 is re-aligned to the new outline.
Writes are crash-safe: the aligned maps are written to temporary files first
and swapped in under a marker file, so an interrupted run is detected next
time and redone from the untouched originals.

HOW THE MATCH IS FOUND
  1. Both masks are reduced to ~512 px and every rotation 0-359° (1° steps) is
     tried with centroids aligned, scoring by overlap (IoU). Searching all
     angles covers flipping the sheet top-to-bottom as well as left-to-right
     (the two differ by 180°).
  2. The best angle is refined with OpenCV's ECC alignment at ~1024 px on
     slightly blurred masks, which also fixes up the translation.
  3. A near-symmetric outline (e.g. a clean rectangle) cannot be resolved by
     shape alone - check the overlay for those.

NORMAL MAP
  The normal map stores directions, so moving its pixels is not enough: the
  vectors are rotated by the same angle. Directions are taken in the renderer's
  tangent-space convention (red = image right, green = image up; three.js loads
  the textures with flipY), so the result is correct for what the viewer draws.

Usage:
    python register_sides.py SIDE1_MAPS_DIR SIDE2_MAPS_DIR [--min-iou 0.8]
    python register_sides.py --check SIDE1_MAPS_DIR SIDE2_MAPS_DIR

Exit codes: 0 aligned (or already up to date), 1 bad input, 2 match too poor
to apply (side2's maps are left as they were). With --check nothing is
changed: 0 = alignment is current, 3 = it needs (re)doing.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

# Render map -> how to resample it.
MAPS = {
    "AlphaMask_render.tiff":     "nearest",
    "DiffuseMap_render.tiff":    "linear",
    "NormalMap_render.tiff":     "normal",
    "SpecularMap_render.tiff":   "linear",
    "RoughnessMap_render.tiff":  "linear",
    "HeightMap_render.tiff":     "linear",
}
MASK_NAME    = "AlphaMask_render.tiff"
ORIG_SUBDIR  = "unregistered"
INFO_NAME    = "registration.json"
OVERLAY_NAME = "registration_overlay.png"
MARKER_NAME  = ".registration_in_progress"   # exists only while maps are swapped
TEMP_SUFFIX  = ".aligned-tmp"

COARSE_PX   = 512     # long side for the exhaustive rotation search
FINE_PX     = 1024    # long side for the ECC refinement
WARN_IOU    = 0.93    # below this the result is applied but flagged


# ── Mask helpers ─────────────────────────────────────────────────────────────
def load_mask(path: Path) -> np.ndarray:
    m = tifffile.imread(str(path))
    if m.ndim == 3:
        m = m[..., 0]
    return m > (127 if m.dtype == np.uint8 else 0)


def resized(mask: np.ndarray, scale: float) -> np.ndarray:
    h, w = mask.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)


def centroid(m: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(m > 0.5)
    return np.array([xs.mean(), ys.mean()])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0.5, b > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def rigid(angle_deg: float, about: np.ndarray, to: np.ndarray) -> np.ndarray:
    """2x3 matrix: rotate by angle about `about`, then move `about` onto `to`."""
    A = cv2.getRotationMatrix2D((float(about[0]), float(about[1])), angle_deg, 1.0)
    A[:, 2] += to - about
    return A


def warp(img, A, shape, interp, border=0):
    return cv2.warpAffine(img, A, (shape[1], shape[0]), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


# ── Estimation ───────────────────────────────────────────────────────────────
def estimate(front: np.ndarray, back: np.ndarray):
    """Rigid 2x3 matrix (full-resolution pixels) taking `back` onto the mirror
    image of `front`, plus the final IoU and the rotation in degrees."""
    target_full = np.fliplr(front)

    # 1. Exhaustive rotation search at low resolution.
    s = COARSE_PX / max(*front.shape, *back.shape)
    t, m = resized(target_full, s), resized(back, s)
    ct, cm = centroid(t), centroid(m)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(0.0, 360.0, 1.0):
        score = iou(warp(m, rigid(angle, cm, ct), t.shape, cv2.INTER_LINEAR), t)
        if score > best_score:
            best_angle, best_score = angle, score
    for angle in np.arange(best_angle - 1.0, best_angle + 1.0, 0.1):
        score = iou(warp(m, rigid(angle, cm, ct), t.shape, cv2.INTER_LINEAR), t)
        if score > best_score:
            best_angle, best_score = angle, score
    A_small = rigid(best_angle, cm, ct)
    A = A_small.copy()
    A[:, 2] /= s

    # 2. ECC refinement at higher resolution. ECC's matrix maps template
    #    (target) coordinates to input (back) coordinates - the inverse of A.
    s2 = min(1.0, FINE_PX / max(*front.shape, *back.shape))
    t2 = cv2.GaussianBlur(resized(target_full, s2), (0, 0), 2)
    m2 = cv2.GaussianBlur(resized(back, s2), (0, 0), 2)
    A2 = A.copy()
    A2[:, 2] *= s2
    try:
        W = cv2.invertAffineTransform(A2).astype(np.float32)
        _, W = cv2.findTransformECC(t2, m2, W, cv2.MOTION_EUCLIDEAN,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                     200, 1e-6), None, 5)
        A_ecc = cv2.invertAffineTransform(W).astype(np.float64)
        A_ecc[:, 2] /= s2
        # Keep the refinement only if it actually improved the overlap.
        if score_full(back, target_full, A_ecc) >= score_full(back, target_full, A):
            A = A_ecc
    except cv2.error:
        pass   # ECC did not converge; the coarse estimate stands

    # Counter-clockwise as seen in the image, in (-180, 180].
    angle = float(-np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return A, score_full(back, target_full, A), angle


def score_full(back, target, A) -> float:
    """IoU of `back` warped by full-res A against `target`, at coarse scale."""
    s = COARSE_PX / max(*back.shape, *target.shape)
    As = A.copy()
    As[:, 2] *= s
    t = resized(target, s)
    return iou(warp(resized(back, s), As, t.shape, cv2.INTER_LINEAR), t)


# ── Applying ─────────────────────────────────────────────────────────────────
def warp_map(img: np.ndarray, kind: str, A: np.ndarray, shape) -> np.ndarray:
    if kind == "nearest":
        return warp(img, A, shape, cv2.INTER_NEAREST)
    if kind == "linear":
        return warp(img, A, shape, cv2.INTER_LINEAR)

    # Normal map: move the pixels, then rotate the vectors they hold.
    full = float(np.iinfo(img.dtype).max) if img.dtype.kind == "u" else 1.0
    flat = (full / 2, full / 2, full)          # (0, 0, 1) encoded
    moved = warp(img, A, shape, cv2.INTER_LINEAR, border=flat).astype(np.float64)
    n = moved[..., :3] / full * 2.0 - 1.0
    # Tangent (red = right, green = up) -> image-space direction (x right, y down).
    dx, dy = n[..., 0], -n[..., 1]
    R = A[:, :2]
    rx = R[0, 0] * dx + R[0, 1] * dy
    ry = R[1, 0] * dx + R[1, 1] * dy
    out = np.stack([rx, -ry, n[..., 2]], axis=-1)
    out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)
    enc = np.clip((out + 1.0) / 2.0 * full, 0, full)
    if moved.shape[-1] > 3:                    # keep any extra channel as moved
        enc = np.concatenate([enc, moved[..., 3:]], axis=-1)
    return np.round(enc).astype(img.dtype)


def fingerprint(path: Path):
    """(size, mtime_ns) - changes whenever a file is rewritten."""
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def read_info(side2: Path) -> dict:
    try:
        return json.loads((side2 / INFO_NAME).read_text())
    except (OSError, ValueError):
        return {}


def maps_are_ours(side2: Path, info: dict) -> bool:
    """True when side2's maps are exactly what this script last wrote."""
    if not info.get("applied") or (side2 / MARKER_NAME).exists():
        return False
    written = info.get("written", {})
    present = [n for n in MAPS if (side2 / n).is_file()]
    return bool(present) and all(
        written.get(n) == fingerprint(side2 / n) for n in present)


def is_current(side1: Path, side2: Path) -> bool:
    """Alignment is applied, untouched since, and made against this side1."""
    info = read_info(side2)
    mask1 = side1 / MASK_NAME
    return (maps_are_ours(side2, info) and mask1.is_file()
            and info.get("side1_mask") == fingerprint(mask1))


def originals_dir(side2: Path) -> Path:
    """Folder holding side2's unaligned maps.

    Refreshed from side2's current maps only when those are NOT this script's
    output - i.e. modeling rewrote them. After an interrupted run (marker file
    still present) the maps are a half-swapped mix, so unregistered/, which was
    complete before any swap began, is used as is.
    """
    orig = side2 / ORIG_SUBDIR
    for leftover in side2.glob("*" + TEMP_SUFFIX):
        leftover.unlink()
    interrupted = (side2 / MARKER_NAME).exists()
    if interrupted and (orig / MASK_NAME).is_file():
        print("Previous alignment was interrupted - redoing it from "
              f"{orig.name}/.")
        return orig
    if not maps_are_ours(side2, read_info(side2)):
        present = [side2 / n for n in MAPS if (side2 / n).is_file()]
        if present:
            orig.mkdir(exist_ok=True)
            for p in present:
                shutil.copy2(p, orig / p.name)
    (side2 / MARKER_NAME).unlink(missing_ok=True)
    return orig


def save_overlay(path: Path, front: np.ndarray, aligned_back: np.ndarray) -> None:
    s = COARSE_PX * 2 / max(front.shape)
    f = resized(np.fliplr(front), s) > 0.5
    b = resized(aligned_back, s) > 0.5
    img = np.zeros((*f.shape, 3), np.uint8)
    img[..., 2] = f * 255      # red   = side1 as seen from behind (BGR order)
    img[..., 1] = b * 255      # green = aligned side2
    cv2.imwrite(str(path), img)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("side1", type=Path, help="side1's maps/ folder (the reference)")
    ap.add_argument("side2", type=Path, help="side2's maps/ folder (gets aligned)")
    ap.add_argument("--min-iou", type=float, default=0.8,
                    help="refuse to apply a match with less overlap (default 0.8)")
    ap.add_argument("--check", action="store_true",
                    help="only report whether the alignment is current (exit 0) "
                         "or needs doing (exit 3); change nothing")
    args = ap.parse_args()

    if args.check:
        return 0 if is_current(args.side1, args.side2) else 3

    for d in (args.side1, args.side2):
        if not (d / MASK_NAME).is_file():
            print(f"ERROR: {d / MASK_NAME} not found - run modeling on both sides first.")
            return 1

    info_path = args.side2 / INFO_NAME
    if is_current(args.side1, args.side2):
        prev = read_info(args.side2)
        print(f"side2 is already aligned to side1 (rotation "
              f"{prev['rotation_deg_ccw']:+.1f}° ccw, overlap {prev['iou']:.3f}) - "
              "nothing to do.")
        return 0
    if maps_are_ours(args.side2, read_info(args.side2)):
        print("side1 has changed since side2 was aligned - re-aligning.")
    orig = originals_dir(args.side2)
    if not (orig / MASK_NAME).is_file():
        print(f"ERROR: no unaligned maps in {orig}.")
        return 1

    front = load_mask(args.side1 / MASK_NAME)
    back = load_mask(orig / MASK_NAME)
    print(f"Front mask {front.shape[1]}x{front.shape[0]}, "
          f"back mask {back.shape[1]}x{back.shape[0]}")
    if not front.any() or not back.any():
        print("ERROR: one of the alpha masks is empty.")
        return 1

    print("Searching rotations 0-359° for the best outline match...")
    A, score, angle = estimate(front, back)
    turn = "counter-clockwise" if angle >= 0 else "clockwise"
    print(f"Best match: turn side2 {abs(angle):.2f}° {turn}, overlap (IoU) {score:.3f}")

    info = {"applied": False, "iou": score, "rotation_deg_ccw": angle,
            "matrix_side2_to_side1": A.tolist(),
            "canvas": [int(front.shape[1]), int(front.shape[0])],
            "note": "side2 pixels are mapped onto side1's outline mirrored "
                    "left-right (how the website shows the back after a flip)"}

    aligned_mask = warp(back.astype(np.uint8), A, front.shape, cv2.INTER_NEAREST)
    save_overlay(args.side2 / OVERLAY_NAME, front, aligned_mask)

    if score < args.min_iou:
        print(f"ERROR: overlap {score:.3f} is below {args.min_iou} - the outlines "
              "do not match well enough to trust this. side2's maps were NOT "
              f"changed. Look at {args.side2 / OVERLAY_NAME}; usually one side's "
              "alpha mask is wrong (background included or part of the sheet "
              "cut away).")
        info_path.write_text(json.dumps(info, indent=2))
        return 2

    # 1. Write every aligned map to a temporary file; side2's maps are not
    #    touched until all of them have succeeded.
    temps = {}
    for name, kind in MAPS.items():
        src = orig / name
        if not src.is_file():
            print(f"  {name}: not found - skipped")
            continue
        img = tifffile.imread(str(src))
        tmp = args.side2 / (name + TEMP_SUFFIX)
        tifffile.imwrite(str(tmp), warp_map(img, kind, A, front.shape))
        temps[name] = tmp
        print(f"  {name}: aligned ({kind})")

    # 2. Swap them in under the marker, then record exactly what was written.
    #    A crash anywhere in here leaves the marker, which the next run sees.
    marker = args.side2 / MARKER_NAME
    marker.write_text("alignment in progress - delete only if no run is active\n")
    for name, tmp in temps.items():
        os.replace(tmp, args.side2 / name)
    info["applied"] = True
    info["written"] = {n: fingerprint(args.side2 / n) for n in temps}
    info["side1_mask"] = fingerprint(args.side1 / MASK_NAME)
    info_path.write_text(json.dumps(info, indent=2))
    marker.unlink()
    if score < WARN_IOU:
        print(f"WARNING: overlap {score:.3f} is lower than usual (<{WARN_IOU}). "
              f"Check {args.side2 / OVERLAY_NAME} before publishing.")
    print(f"Done. Check image: {args.side2 / OVERLAY_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
