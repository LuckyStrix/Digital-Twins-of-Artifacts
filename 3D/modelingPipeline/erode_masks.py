#!/usr/bin/env python3
"""
Erode the alpha mask of processed (background-removed) photos inward by N px.

Usage:
    python erode_masks.py data/<folder> 5
    python erode_masks.py data/<folder> 5 --in-place
    python erode_masks.py data/<folder> 5 --workers 20

By default writes to data/<folder>/processed_eroded/ (mirroring the
processed/ subfolder structure). Pass --in-place to overwrite the
files in processed/ directly.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTS = {".png"}  # only PNGs carry an alpha channel


def erode_alpha(alpha: np.ndarray, px: int) -> np.ndarray:
    """Shrink the opaque region of an alpha channel inward by px pixels."""
    if px <= 0:
        return alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * px + 1, 2 * px + 1))
    return cv2.erode(alpha, kernel)


def _process_one(args: tuple[Path, Path, int]) -> tuple[str, str | None]:
    """Runs in a worker process: erode one file's alpha and save it.

    Returns (name, error). error is None on success. Read failures are
    retried a few times before giving up — on a WSL /mnt/c (DrvFs) mount,
    many processes reading concurrently can intermittently return short/
    truncated reads that succeed a moment later.
    """
    src, dst, px = args
    # Each worker is its own process; without this, OpenCV would spin up its
    # own internal thread pool per process and oversubscribe the CPU.
    cv2.setNumThreads(1)

    last_exc = None
    for attempt in range(3):
        try:
            img = Image.open(src).convert("RGBA")
            break
        except OSError as exc:
            last_exc = exc
            time.sleep(0.5 * (attempt + 1))
    else:
        return (src.name, f"read failed after retries: {last_exc}")

    try:
        arr = np.array(img)
        arr[:, :, 3] = erode_alpha(arr[:, :, 3], px)
        Image.fromarray(arr, mode="RGBA").save(dst, "PNG")
    except Exception as exc:
        return (src.name, f"processing failed: {exc}")

    return (src.name, None)


def process_folder(folder: Path, px: int, in_place: bool, workers: int) -> None:
    processed_dir = folder / "processed"
    if not processed_dir.exists():
        sys.exit(f"[error] No processed/ folder found under {folder}")

    files = sorted(
        f for f in processed_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        sys.exit(f"[error] No masked PNGs found in {processed_dir}")

    out_dir = processed_dir if in_place else folder / "processed_eroded"

    jobs = []
    for src in files:
        dst = src if in_place else out_dir / src.relative_to(processed_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((src, dst, px))

    workers = max(1, min(workers, len(jobs)))
    print(f"[info] {len(files)} image(s) in {processed_dir}")
    print(f"[info] Erosion: {px}px")
    print(f"[info] Output:  {out_dir}{' (in-place)' if in_place else ''}")
    print(f"[info] Workers: {workers}")

    failures = []
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            name, err = _process_one(job)
            if err:
                failures.append((name, err))
                print(f"  [{i}/{len(jobs)}] {name}  FAILED: {err}")
            else:
                print(f"  [{i}/{len(jobs)}] {name}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, job): job for job in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                name, err = fut.result()
                if err:
                    failures.append((name, err))
                    print(f"  [{i}/{len(jobs)}] {name}  FAILED: {err}")
                else:
                    print(f"  [{i}/{len(jobs)}] {name}")

    ok = len(files) - len(failures)
    print(f"[done] {ok}/{len(files)} succeeded, wrote to {out_dir}")
    if failures:
        print(f"[warn] {len(failures)} file(s) failed:")
        for name, err in failures:
            print(f"  - {name}: {err}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path,
                    help="Folder in data/ containing a processed/ subfolder")
    p.add_argument("px", type=int, help="Number of pixels to erode the mask inward")
    p.add_argument("--in-place", action="store_true",
                    help="Overwrite files in processed/ instead of writing to processed_eroded/")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Number of parallel worker processes (default: all CPU cores)")
    args = p.parse_args()

    if args.px < 0:
        sys.exit("[error] px must be >= 0")

    process_folder(args.folder, args.px, args.in_place, args.workers)


if __name__ == "__main__":
    main()
