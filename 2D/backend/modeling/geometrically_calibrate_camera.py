#!/usr/bin/env python3
"""Geometrically calibrate a camera + lens from photos of a checkerboard.

Board: 8 x 9 squares, 45 mm each  ->  7 x 8 inner corners (what OpenCV detects).

Usage:
    python geometrically_calibrate_camera.py /path/to/photos [-o calibration.json]

Requires: pip install opencv-python numpy

Output (JSON) uses the standard OpenCV / pinhole model:
    camera_matrix  K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]   (pixels)
    dist_coeffs    [k1, k2, p1, p2, k3]                        (OpenCV order)
plus the same values as a COLMAP "OPENCV" camera params string.
All photos must be the same resolution, taken with the same lens/zoom/focus.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 8, 9      # squares on the board
SQUARE_MM = 45.0
PATTERN = (SQUARES_X - 1, SQUARES_Y - 1)  # inner corners: (7, 8)
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("calibration.json"))
    args = ap.parse_args()

    # 3D corner positions on the flat board (z = 0), in millimetres.
    board = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    board[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM

    obj_pts, img_pts, names, size = [], [], [], None
    for path in sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTS):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"  skip (unreadable)  {path.name}")
            continue
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            print(f"  skip (size {gray.shape[::-1]} != {size})  {path.name}")
            continue
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, flags=cv2.CALIB_CB_ACCURACY)
        if not found:
            print(f"  skip (board not found)  {path.name}")
            continue
        print(f"  ok  {path.name}")
        obj_pts.append(board)
        img_pts.append(corners)
        names.append(path.name)

    if len(img_pts) < 5:
        raise SystemExit(f"Only {len(img_pts)} usable photos; need at least 5 (10-30 is better). Take more photos, then run again. Photos must be formatted as one of the following: {EXTS}")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    dist = dist.ravel()

    # Per-photo reprojection error (pixels) so bad shots are easy to spot.
    per_image = {}
    for name, o, i, r, t in zip(names, obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_image[name] = float(np.sqrt(np.mean(np.sum((proj - i) ** 2, axis=2))))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist[:5]
    result = {
        "image_width": size[0],
        "image_height": size[1],
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.tolist(),  # k1, k2, p1, p2, k3
        "rms_reprojection_error_px": rms,
        "num_images_used": len(names),
        "per_image_error_px": per_image,
        "board": {"squares": [SQUARES_X, SQUARES_Y], "inner_corners": list(PATTERN), "square_mm": SQUARE_MM},
        # COLMAP OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2 (k3 not supported there)
        "colmap_opencv_params": f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}",
    }
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\nUsed {len(names)} photos, RMS reprojection error {rms:.3f} px (under ~0.5 is good, over 1 is suspect)")
    worst = max(per_image, key=per_image.get)
    print(f"Worst photo: {worst} ({per_image[worst]:.3f} px)")
    print(f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"k1={k1:.5f} k2={k2:.5f} p1={p1:.5f} p2={p2:.5f} k3={k3:.5f}")
    print(f"Saved {args.out.resolve()}")


if __name__ == "__main__":
    main()
