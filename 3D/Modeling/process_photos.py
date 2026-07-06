#!/usr/bin/env python3
"""
BiRefNet background removal pipeline.

LOCAL MODE (default)
--------------------
Drop images into input/ and run:

    python process_photos.py

Structured input (dual-camera — recommended):
    input/side1/*.jpg  →  output/side1/*.jpg
    input/side2/*.jpg  →  output/side2/*.jpg

Flat input (single camera):
    input/*.jpg        →  output/*.jpg

Feed the output directly into the COLMAP pipeline:
    bash run.sh -i output/side1 -s output/side2 -o results/

NAS MODE (legacy)
-----------------
    python process_photos.py --nas

Reads the most recent folder from NAS_DATA, removes backgrounds, writes
to data/<folder>_masked/, and copies back to the NAS.

OPTIONS
-------
  --input DIR           Input folder (default: ./input)
  --output DIR          Output folder (default: ./output)
  --model NAME          rembg model (default: birefnet-general)
  --background          white | black | transparent (default: white)
  --black-threshold N   Zero out foreground pixels darker than N (0-255, default: 0 = off)
  --nas                 Run legacy NAS workflow instead of local mode
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove

SCRIPT_DIR = Path(__file__).parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

# NAS configuration (only used with --nas)
NAS_DATA = Path("/mnt/z/fip/Data/tabletCaptures")
LOCAL_DATA = SCRIPT_DIR / "data"


# ── arg parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BiRefNet background removal for tablet photos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        default=str(SCRIPT_DIR / "input"),
        help="Input folder (default: ./input). Ignored in --nas mode.",
    )
    p.add_argument(
        "--output",
        default=str(SCRIPT_DIR / "output"),
        help="Output folder (default: ./output). Ignored in --nas mode.",
    )
    p.add_argument(
        "--model",
        default="birefnet-general",
        help="rembg model name (default: birefnet-general).",
    )
    p.add_argument(
        "--background",
        default="white",
        choices=["white", "black", "transparent"],
        help="Replacement background colour (default: white).",
    )
    p.add_argument(
        "--black-threshold",
        type=int,
        default=0,
        metavar="N",
        help="Zero out foreground pixels whose brightness is below N (0-255). "
             "0 = disabled (default). Try 30-60 to remove black felt.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        help="Keep every Nth image per subfolder, matching COLMAP's image stride "
             "(N >= 1, default: 1 = keep all). Example: 3 keeps images 0, 3, 6, ...",
    )
    p.add_argument(
        "--nas",
        action="store_true",
        help="Run legacy NAS workflow (pull → process → push).",
    )
    return p.parse_args()


# ── image processing ─────────────────────────────────────────────────────────

def _apply_black_threshold(rgba: Image.Image, src_rgb: np.ndarray, threshold: int) -> Image.Image:
    """Zero out alpha for foreground pixels darker than threshold."""
    brightness = src_rgb.mean(axis=2)
    alpha = np.array(rgba.split()[3])
    alpha[brightness < threshold] = 0
    out = rgba.copy()
    out.putalpha(Image.fromarray(alpha))
    return out


def remove_background(src: Path, dst: Path, session, background: str,
                      black_threshold: int = 0) -> None:
    """Remove background from src and write to dst with the chosen fill."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_img = Image.open(src)
    result = remove(src_img, session=session)

    if black_threshold > 0:
        src_rgb = np.array(src_img.convert("RGB"))
        result = _apply_black_threshold(result, src_rgb, black_threshold)

    if background == "transparent":
        out = dst.with_suffix(".png")
        result.save(out, "PNG")
    else:
        bg_color = (255, 255, 255) if background == "white" else (0, 0, 0)
        bg = Image.new("RGB", result.size, bg_color)
        alpha = result.split()[3] if result.mode == "RGBA" else None
        bg.paste(result, mask=alpha)
        out = dst.with_suffix(".jpg")
        bg.save(out, "JPEG", quality=95)


# ── local mode ────────────────────────────────────────────────────────────────

def collect_images(input_dir: Path, stride: int = 1) -> list[tuple[Path, Path]]:
    """Return [(src_path, path_relative_to_input_dir), ...] with stride applied per subfolder.

    Stride is applied independently within each immediate subdirectory (mirroring
    COLMAP's per-set image_stride), so side1 and side2 each start their own index
    from 0. A flat directory is treated as one group.
    """
    all_pairs = sorted(
        (f, f.relative_to(input_dir))
        for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if stride <= 1:
        return all_pairs

    groups: dict[str, list] = {}
    for src, rel in all_pairs:
        key = rel.parts[0] if len(rel.parts) > 1 else ""
        groups.setdefault(key, []).append((src, rel))

    result = []
    for key in sorted(groups):
        result.extend(groups[key][::stride])
    return result


def run_local(input_dir: Path, output_dir: Path, model: str, background: str,
              black_threshold: int = 0, stride: int = 1) -> None:
    if not input_dir.exists():
        sys.exit(f"[error] Input folder not found: {input_dir}\n"
                 f"        Create it or pass --input <path>.")

    pairs = collect_images(input_dir, stride)
    if not pairs:
        sys.exit(f"[error] No images found in {input_dir}\n"
                 f"        Supported: {', '.join(sorted(IMAGE_EXTS))}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # summarise structure
    subdirs = sorted({rel.parts[0] for _, rel in pairs if len(rel.parts) > 1})
    if subdirs:
        print(f"[info]  Structure: {len(subdirs)} subfolder(s): {', '.join(subdirs)}")
    else:
        print(f"[info]  Structure: flat (all images at root)")

    print(f"[info]  Input:   {input_dir}  ({len(pairs)} image(s))")
    print(f"[info]  Output:  {output_dir}")
    print(f"[info]  Background: {background}")
    if stride > 1:
        print(f"[info]  Stride: {stride} (keeping every {stride}th image per subfolder)")
    if black_threshold > 0:
        print(f"[info]  Black threshold: {black_threshold}")
    print(f"[rembg] Loading model '{model}' ...")

    session = new_session(model)

    print(f"[rembg] Processing {len(pairs)} image(s) ...")
    errors = 0
    for i, (src, rel) in enumerate(pairs, 1):
        dst = output_dir / rel
        try:
            remove_background(src, dst, session, background, black_threshold=black_threshold)
            print(f"  [{i}/{len(pairs)}] {rel}")
        except Exception as exc:
            errors += 1
            print(f"  [{i}/{len(pairs)}] ERROR {rel}: {exc}", file=sys.stderr)

    if errors:
        print(f"[done]  {len(pairs) - errors}/{len(pairs)} succeeded, {errors} failed.")
        print(f"[done]  Output: {output_dir}")
    else:
        print(f"[done]  All {len(pairs)} images saved to {output_dir}")

    if subdirs:
        print()
        print("Next step — run the COLMAP pipeline:")
        if len(subdirs) >= 2:
            s1, s2 = subdirs[0], subdirs[1]
            print(f"  bash run.sh -i {output_dir}/{s1} -s {output_dir}/{s2} -o results/")
        else:
            print(f"  bash run.sh -i {output_dir}/{subdirs[0]} -o results/")


# ── NAS mode (legacy) ─────────────────────────────────────────────────────────

def _most_recent_folder(base: Path) -> Path:
    if not base.exists():
        sys.exit(f"[error] NAS path not accessible: {base}")
    folders = [f for f in base.iterdir() if f.is_dir()]
    if not folders:
        sys.exit(f"[error] No folders found in {base}")
    newest = max(folders, key=lambda f: f.stat().st_mtime)
    print(f"[scan]  Most recent folder: {newest.name}")
    return newest


def _copy_to_local(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[copy]  {dst.name} already exists locally — skipping.")
        return
    print(f"[copy]  {src}  →  {dst}")
    shutil.copytree(src, dst)
    print("[copy]  Done.")


def _copy_to_nas(src: Path, dst: Path) -> None:
    if dst.exists():
        print(f"[upload] {dst.name} already exists on NAS — skipping.")
        return
    print(f"[upload] {src}  →  {dst}")
    for side in src.iterdir():
        nas_side = dst / side.name
        nas_side.mkdir(parents=True, exist_ok=True)
        for f in side.iterdir():
            shutil.copyfile(f, nas_side / f.name)
    print("[upload] Done.")


def run_nas(model: str, background: str, black_threshold: int = 0) -> None:
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    source_folder = _most_recent_folder(NAS_DATA)
    folder_name = source_folder.name

    local_src = LOCAL_DATA / folder_name
    _copy_to_local(source_folder, local_src)

    local_masked = LOCAL_DATA / f"{folder_name}_masked"
    run_local(local_src, local_masked, model, background, black_threshold=black_threshold)

    nas_output = NAS_DATA / local_masked.name
    _copy_to_nas(local_masked, nas_output)

    print(f"\n[done]  Local:  {local_masked}")
    print(f"[done]  NAS:    {nas_output}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.nas:
        run_nas(args.model, args.background, black_threshold=args.black_threshold)
    else:
        run_local(Path(args.input), Path(args.output), args.model, args.background,
                  black_threshold=args.black_threshold, stride=args.stride)


if __name__ == "__main__":
    main()
