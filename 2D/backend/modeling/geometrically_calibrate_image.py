#!/usr/bin/env python3
# =============================================================================
# undistort.py - correct lens distortion using an OpenCV calibration.json
# =============================================================================
#
# DEPENDENCIES
#   Python 3.8+
#   opencv-python   (pip install opencv-python)
#   numpy           (installed automatically with opencv-python)
#
#   Standard library only otherwise (argparse, json, glob, pathlib, sys).
#
# CALIBRATION FILE
#   Expects a JSON file with at least these keys:
#     image_width, image_height   - resolution the calibration was made at
#     camera_matrix               - 3x3 intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
#     dist_coeffs                 - OpenCV order [k1, k2, p1, p2, k3]
#   Valid only for the same camera, lens, focal length and focus distance
#   that were used for the checkerboard shots.
#
# USAGE
#   python undistort.py INPUT [INPUT ...] [options]
#
#   INPUT can be files, glob patterns (quote them so your shell doesn't
#   expand them), or directories (all supported images inside are used).
#
#   Options:
#     -c, --calibration PATH   calibration file          (default: calibration.json)
#     -o, --output-dir DIR     where to write results    (default: ./undistorted)
#     -a, --alpha FLOAT        0.0 = crop to valid pixels only (no black edges)
#                              1.0 = keep every source pixel (black borders)
#                              values in between blend the two   (default: 0.0)
#         --crop               with alpha > 0, also crop to the valid region
#         --suffix TEXT        appended to output filename stem  (default: _undistorted)
#         --interp NAME        nearest | linear | cubic | lanczos (default: lanczos)
#         --jpeg-quality INT   quality for .jpg output, 1-100    (default: 95)
#
#   Examples:
#     python undistort.py photo.tiff
#     python undistort.py shots/*.tiff -o fixed/
#     python undistort.py "shots/DSC_*.tiff" -c my_cal.json --alpha 1
#     python undistort.py shots/ --interp cubic
#
# OUTPUT FORMATTING
#   Files
#     - One output file per input, written to --output-dir (created if needed).
#     - Name:   <input stem><suffix><input extension>
#               e.g. DSC_0001.tiff -> undistorted/DSC_0001_undistorted.tiff
#     - Same file format as the input. Bit depth and channel count are
#       preserved (16-bit TIFF in -> 16-bit TIFF out). Originals are never
#       modified. Existing outputs with the same name are overwritten.
#     - Image size:  the output keeps the input resolution. With alpha=0 the
#       result is slightly zoomed so no black edges appear. With alpha>0 the
#       black borders are kept, unless --crop is given, in which case the
#       file is cropped to the valid region and is smaller than the input.
#
#   Console (stdout / stderr)
#     One line per image:
#       [ok]   <input> -> <output>  (<W>x<H>, <dtype>)
#       [skip] <input>  <reason>            (written to stderr)
#     followed by a one-line summary:
#       Done: <n_ok> written, <n_skipped> skipped -> <output dir>
#
#   Exit status
#     0 if at least one image was written and none failed to load/save,
#     1 if any image was skipped or nothing was processed,
#     2 for bad arguments or an unreadable calibration file.
#
# NOTES
#   - If an input has a different resolution than the calibration but the
#     same aspect ratio, the intrinsics are scaled to match (distortion
#     coefficients are scale-invariant). A different aspect ratio (i.e. a
#     crop) is skipped, since the principal point would be wrong.
#   - Interpolation alters pixel values. For radiometric work, undistort as
#     late as possible, or use cv2.undistortPoints on coordinates instead.
#   - Raw formats (NEF, DNG, ARW...) are not readable by OpenCV; convert to
#     TIFF/PNG first.
# =============================================================================

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["dist_coeffs"], dtype=np.float64).ravel()
    size = (int(cal["image_width"]), int(cal["image_height"]))
    return K, dist, size


def expand_inputs(items):
    files = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        else:
            matches = sorted(glob.glob(item))
            files += [Path(m) for m in matches] if matches else [p]
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def scaled_intrinsics(K, cal_size, img_size):
    """Scale K to a different resolution of the same aspect ratio."""
    cw, ch = cal_size
    w, h = img_size
    if (w, h) == (cw, ch):
        return K
    sx, sy = w / cw, h / ch
    if abs(sx - sy) / max(sx, sy) > 0.01:
        return None
    K2 = K.copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    return K2


def main():
    ap = argparse.ArgumentParser(description="Undistort images using calibration.json")
    ap.add_argument("inputs", nargs="+", help="image files, globs or directories")
    ap.add_argument("-c", "--calibration", default="calibration.json")
    ap.add_argument("-o", "--output-dir", default="undistorted")
    ap.add_argument("-a", "--alpha", type=float, default=0.0)
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--suffix", default="_undistorted")
    ap.add_argument("--interp", choices=INTERP, default="lanczos")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        ap.error("--alpha must be between 0 and 1")

    try:
        K, dist, cal_size = load_calibration(args.calibration)
    except (OSError, KeyError, ValueError) as e:
        print(f"Cannot read calibration '{args.calibration}': {e}", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = expand_inputs(args.inputs)
    n_ok = n_skip = 0
    cache = {}  # (w, h) -> (map1, map2, roi), so batches reuse the remap

    for path in files:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[skip] {path}  could not read as an image", file=sys.stderr)
            n_skip += 1
            continue

        h, w = img.shape[:2]
        if (w, h) not in cache:
            Ks = scaled_intrinsics(K, cal_size, (w, h))
            if Ks is None:
                cache[(w, h)] = None
            else:
                newK, roi = cv2.getOptimalNewCameraMatrix(Ks, dist, (w, h), args.alpha, (w, h))
                m1, m2 = cv2.initUndistortRectifyMap(Ks, dist, None, newK, (w, h), cv2.CV_32FC1)
                cache[(w, h)] = (m1, m2, roi)

        entry = cache[(w, h)]
        if entry is None:
            print(f"[skip] {path}  {w}x{h} has a different aspect ratio than "
                  f"calibration ({cal_size[0]}x{cal_size[1]})", file=sys.stderr)
            n_skip += 1
            continue

        m1, m2, roi = entry
        out = cv2.remap(img, m1, m2, INTERP[args.interp])

        if args.crop and args.alpha > 0:
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                out = out[y:y + rh, x:x + rw]

        out_path = out_dir / f"{path.stem}{args.suffix}{path.suffix}"
        params = []
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        if not cv2.imwrite(str(out_path), out, params):
            print(f"[skip] {path}  could not write {out_path}", file=sys.stderr)
            n_skip += 1
            continue

        oh, ow = out.shape[:2]
        print(f"[ok]   {path} -> {out_path}  ({ow}x{oh}, {out.dtype})")
        n_ok += 1

    print(f"Done: {n_ok} written, {n_skip} skipped -> {out_dir}")
    return 0 if n_ok and not n_skip else 1


if __name__ == "__main__":
    sys.exit(main())