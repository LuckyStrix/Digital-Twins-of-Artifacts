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
  --white-threshold N   Zero out foreground pixels brighter than N (0-255, default: 0 = off)
  --value-threshold N   Zero out foreground pixels whose HSV Value (max channel) exceeds
                        N (0-255, default: 0 = off) — catches bright/tinted bleed that
                        mean-brightness (--white-threshold) misses
  --edge-band N         Only apply the thresholds above within N pixels of the mask
                        edge, leaving the interior of the foreground untouched
                        (0-255, default: 0 = apply to the whole mask)
  --passes N            Run background removal N times (default: 1). Each extra pass
                        flattens the background (per the previous pass's mask) to its
                        own average colour and re-runs rembg on the cleaner image,
                        then re-applies the refined mask to the original photo.
                        Slower (Nx rembg runs per image) but can sharpen mask edges
                        when the raw backdrop is noisy/textured.
  --seg-scale-pct N     Downscale the image to N%% of its original size before feeding
                        it to the background-removal model (1-100, default: 100 = full
                        resolution). The resulting mask is upscaled back and applied to
                        the original full-resolution photo, so saved output is always
                        full size. Note: rembg's bundled models (incl. BiRefNet) already
                        resize their input to a fixed internal working resolution before
                        running on the GPU, so this mainly cuts CPU/RAM time spent
                        decoding/resizing large source photos rather than GPU VRAM.
  --nas                 Run legacy NAS workflow instead of local mode
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
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
        "--white-threshold",
        type=int,
        default=0,
        metavar="N",
        help="Zero out foreground pixels whose brightness is above N (0-255). "
             "0 = disabled (default). Try 200-230 to remove white felt/backdrop bleed.",
    )
    p.add_argument(
        "--value-threshold",
        type=int,
        default=0,
        metavar="N",
        help="Zero out foreground pixels whose HSV Value (max of R,G,B) exceeds N "
             "(0-255). 0 = disabled (default). More aggressive than --white-threshold "
             "at catching bright/tinted backdrop bleed, since it looks at the single "
             "brightest channel instead of the RGB mean.",
    )
    p.add_argument(
        "--edge-band",
        type=int,
        default=0,
        metavar="N",
        help="Restrict --black/white/value-threshold to within N pixels of the mask "
             "edge (default: 0 = apply across the whole foreground). Use this to stop "
             "the thresholds from eating into the interior of the tablet while still "
             "cleaning up background bleed at the mask boundary.",
    )
    p.add_argument(
        "--passes",
        type=int,
        default=1,
        metavar="N",
        help="Run rembg N times per image (N >= 1, default: 1). Each extra pass "
             "flattens the background per the previous mask and re-segments the "
             "cleaner image; the final mask is applied to the original photo. "
             "Slower, but can sharpen masks against noisy/textured backdrops.",
    )
    p.add_argument(
        "--seg-scale-pct",
        type=int,
        default=100,
        metavar="N",
        help="Downscale images to N%% of original size before feeding them to the "
             "background-removal model (1-100, default: 100 = full resolution). "
             "Saved output stays full resolution regardless — only the model's "
             "input is downscaled. Mainly saves CPU/RAM time, not GPU VRAM, since "
             "rembg already resizes to a fixed internal resolution per model.",
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

def _mask_edge_band(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Boolean mask: True for foreground pixels within `radius` px of the alpha
    mask's edge. Erodes the binary foreground mask with a (2*radius+1) square
    min-filter — pixels present in the original mask but absent from the eroded
    one are the boundary ring."""
    binary = (alpha > 0).astype(np.uint8) * 255
    eroded = np.array(Image.fromarray(binary, mode="L")
                       .filter(ImageFilter.MinFilter(2 * radius + 1)))
    return (alpha > 0) & (eroded == 0)


def _apply_brightness_thresholds(rgba: Image.Image, src_rgb: np.ndarray,
                                  black_threshold: int, white_threshold: int,
                                  value_threshold: int, edge_band: int = 0) -> Image.Image:
    """Zero out alpha for foreground pixels darker than black_threshold, brighter
    (mean RGB) than white_threshold, or brighter (HSV Value = max channel) than
    value_threshold. Any of the three may be 0/disabled.

    If edge_band > 0, the thresholds only affect pixels within that many pixels
    of the foreground mask's edge, so the tablet's interior is never touched —
    only likely background bleed at the mask boundary."""
    alpha = np.array(rgba.split()[3])
    band = _mask_edge_band(alpha, edge_band) if edge_band > 0 \
        else np.ones_like(alpha, dtype=bool)

    if black_threshold > 0 or white_threshold > 0:
        brightness = src_rgb.mean(axis=2)
        if black_threshold > 0:
            alpha[band & (brightness < black_threshold)] = 0
        if white_threshold > 0:
            alpha[band & (brightness > white_threshold)] = 0
    if value_threshold > 0:
        value = src_rgb.max(axis=2)
        alpha[band & (value > value_threshold)] = 0
    out = rgba.copy()
    out.putalpha(Image.fromarray(alpha))
    return out


def _multi_pass_remove(src_img: Image.Image, session, passes: int,
                       seg_scale: float = 1.0) -> Image.Image:
    """Run rembg `passes` times. After each non-final pass, flatten the pixels
    the pass classified as background to their own average colour and re-run
    rembg on the cleaner image — this can sharpen the mask against a noisy or
    textured backdrop. The final RGBA always carries the *original* photo's
    colours, with only the last pass's alpha applied, so flattening never
    leaks into the saved output.

    If seg_scale < 1.0, the image handed to rembg (and all intermediate
    flattening) runs at that reduced size; the final pass's alpha is then
    upscaled back to src_img's original resolution before being applied, so
    the saved output is always full size regardless of seg_scale."""
    original_size = src_img.size
    current = src_img.convert("RGB")
    if seg_scale < 1.0:
        seg_size = (max(1, round(original_size[0] * seg_scale)),
                    max(1, round(original_size[1] * seg_scale)))
        current = current.resize(seg_size, Image.Resampling.LANCZOS)

    alpha = None
    for i in range(passes):
        result = remove(current, session=session)
        alpha = result.split()[3]
        if i < passes - 1:
            alpha_arr = np.array(alpha)
            bg = alpha_arr < 128
            if bg.any():
                arr = np.array(current).copy()
                arr[bg] = arr[bg].mean(axis=0).astype(np.uint8)
                current = Image.fromarray(arr)

    if alpha.size != original_size:
        alpha = alpha.resize(original_size, Image.Resampling.LANCZOS)

    final = src_img.convert("RGBA")
    final.putalpha(alpha)
    return final


def remove_background(src: Path, dst: Path, session, background: str,
                      black_threshold: int = 0, white_threshold: int = 0,
                      value_threshold: int = 0, edge_band: int = 0,
                      passes: int = 1, seg_scale_pct: int = 100) -> None:
    """Remove background from src and write to dst with the chosen fill."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_img = Image.open(src)
    seg_scale = max(1, min(100, seg_scale_pct)) / 100.0
    result = _multi_pass_remove(src_img, session, passes, seg_scale)

    if black_threshold > 0 or white_threshold > 0 or value_threshold > 0:
        src_rgb = np.array(src_img.convert("RGB"))
        result = _apply_brightness_thresholds(result, src_rgb, black_threshold,
                                               white_threshold, value_threshold, edge_band)

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
              black_threshold: int = 0, white_threshold: int = 0,
              value_threshold: int = 0, edge_band: int = 0, passes: int = 1,
              stride: int = 1, seg_scale_pct: int = 100) -> None:
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
    if white_threshold > 0:
        print(f"[info]  White threshold: {white_threshold}")
    if value_threshold > 0:
        print(f"[info]  Value threshold: {value_threshold}")
    if edge_band > 0:
        print(f"[info]  Edge band: {edge_band}px")
    if passes > 1:
        print(f"[info]  Passes: {passes}")
    if seg_scale_pct < 100:
        print(f"[info]  Segmentation input scale: {seg_scale_pct}% (output stays full res)")
    print(f"[rembg] Loading model '{model}' ...")

    session = new_session(model)

    print(f"[rembg] Processing {len(pairs)} image(s) ...")
    errors = 0
    for i, (src, rel) in enumerate(pairs, 1):
        dst = output_dir / rel
        try:
            remove_background(src, dst, session, background,
                               black_threshold=black_threshold, white_threshold=white_threshold,
                               value_threshold=value_threshold, edge_band=edge_band,
                               passes=passes, seg_scale_pct=seg_scale_pct)
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


def run_nas(model: str, background: str, black_threshold: int = 0,
            white_threshold: int = 0, value_threshold: int = 0,
            edge_band: int = 0, passes: int = 1, seg_scale_pct: int = 100) -> None:
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    source_folder = _most_recent_folder(NAS_DATA)
    folder_name = source_folder.name

    local_src = LOCAL_DATA / folder_name
    _copy_to_local(source_folder, local_src)

    local_masked = LOCAL_DATA / f"{folder_name}_masked"
    run_local(local_src, local_masked, model, background,
              black_threshold=black_threshold, white_threshold=white_threshold,
              value_threshold=value_threshold, edge_band=edge_band, passes=passes,
              seg_scale_pct=seg_scale_pct)

    nas_output = NAS_DATA / local_masked.name
    _copy_to_nas(local_masked, nas_output)

    print(f"\n[done]  Local:  {local_masked}")
    print(f"[done]  NAS:    {nas_output}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.nas:
        run_nas(args.model, args.background, black_threshold=args.black_threshold,
                white_threshold=args.white_threshold, value_threshold=args.value_threshold,
                edge_band=args.edge_band, passes=args.passes,
                seg_scale_pct=args.seg_scale_pct)
    else:
        run_local(Path(args.input), Path(args.output), args.model, args.background,
                  black_threshold=args.black_threshold, white_threshold=args.white_threshold,
                  value_threshold=args.value_threshold, edge_band=args.edge_band,
                  passes=args.passes, stride=args.stride, seg_scale_pct=args.seg_scale_pct)


if __name__ == "__main__":
    main()
