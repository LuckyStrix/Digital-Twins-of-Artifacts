#!/usr/bin/env python3
"""
rotate_images.py
=================
Rotates every TIFF in input_images/scroll_images/ by 90 degrees clockwise,
in place.

Before rotating, an untouched copy of each file is saved to
input_images/scroll_images_original/ (created on first run).

This script is safe to re-run: each run re-derives the 90-degree-clockwise
version from the backed-up original, so repeated runs always produce the
same result instead of rotating further each time.

Usage:
    python3 rotate_images.py
"""

import shutil
from pathlib import Path

import cv2

ROOT     = Path(__file__).resolve().parent   # .../integrated/backend/modeling
APP_ROOT = ROOT.parent.parent                # .../integrated

SCROLL_DIR = APP_ROOT / "input_images" / "scroll_images"
BACKUP_DIR = APP_ROOT / "input_images" / "scroll_images_original"


def main() -> None:
    files = sorted(SCROLL_DIR.glob("*.tiff"))
    if not files:
        print(f"No .tiff files found in {SCROLL_DIR}")
        return

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    for f in files:
        backup = BACKUP_DIR / f.name
        if not backup.exists():
            shutil.copy2(f, backup)

        img = cv2.imread(str(backup), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: could not read {backup} — skipping {f.name}")
            continue

        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(str(f), rotated)
        print(f"  {f.name}: {img.shape[1]}x{img.shape[0]} -> "
              f"{rotated.shape[1]}x{rotated.shape[0]} (rotated 90° CW)")

    print(f"\nDone. Originals preserved in {BACKUP_DIR}")


if __name__ == "__main__":
    main()
