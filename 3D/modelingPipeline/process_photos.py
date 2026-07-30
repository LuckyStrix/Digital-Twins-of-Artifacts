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
  --hard-mask           Threshold rembg's soft alpha matte to a hard binary mask
                        (morphological open + blur + threshold at 127) instead of
                        leaving it as a continuous confidence gradient (default: on;
                        pass --no-hard-mask to keep the raw soft matte). The raw matte
                        blends real tablet-edge colour with backdrop colour at partial
                        alpha around the whole silhouette; since COLMAP has no notion of
                        alpha and only ever sees RGB, that blended ring survives as
                        photo-consistent (but wrong) content and can triangulate into
                        speckled noise hugging the reconstructed mesh's edges. Hard-mask
                        collapses that ring to a clean binary cut so background pixels
                        are fully replaced by the chosen --background colour instead of
                        blending with it.
  --black-threshold N   Zero out foreground pixels darker than N (0-255, default: 0 = off)
  --white-threshold N   Zero out foreground pixels brighter than N (0-255, default: 0 = off)
  --value-threshold N   Zero out foreground pixels whose HSV Value (max channel) exceeds
                        N (0-255, default: 0 = off) — catches bright/tinted bleed that
                        mean-brightness (--white-threshold) misses
  --chroma-threshold N  Zero out foreground pixels whose chroma (max(R,G,B) - min(R,G,B))
                        is below N (0-255, default: 0 = off) — removes neutral/grey
                        backdrop bleed regardless of how bright or dark it is, as an
                        alternative (or complement) to tuning --black/white-threshold.
                        Only safe if the tablet itself has some real colour; a grey/
                        monochrome tablet would get eaten too.
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
  --erode-px N          Shrink the foreground mask inward by N pixels after
                        segmentation (0-255, default: 0 = off). Trims thin halos of
                        background bleed left at the silhouette edge by rembg.
  --grow-chroma N       Grow the mask to include any region touching the current
                        foreground whose chroma (max(R,G,B) - min(R,G,B)) is >= N
                        (0-255, default: 0 = off). Fixes rembg dropping an attached,
                        differently-coloured piece (e.g. a mounting board) as
                        background: as long as the actual backdrop is neutral
                        (black/white/grey), anything with real colour next to the
                        tablet gets pulled in. No hue is hardcoded, so this works
                        for any tablet/board colour combination on the same style
                        of neutral backdrop. Chroma is used instead of HSV
                        saturation because saturation is a ratio (chroma / value)
                        that blows up from ordinary sensor noise on near-black
                        pixels — a black backdrop can misread as "saturated"
                        despite having almost no real colour. Try 40-60 to start.
  --grow-close-px N     Morphological closing radius (px) applied to the colour
                        mask before growing, to bridge small gaps/glare spots
                        within the coloured region (default: 15). Only matters
                        when --grow-chroma > 0.
  --grow-feather-px N   Soften the edge of the newly grown region with a blur of
                        this radius (px), since the colour mask is a hard
                        threshold unlike rembg's matted alpha (default: 3). Only
                        matters when --grow-chroma > 0.
  --grow-hull-fill      Replace the grown region with its convex hull, filled
                        solid (default: off). Bridges low-colour detail sitting
                        on top of it (e.g. a printed label) that a plain hole-fill
                        can't reach if it touches the region's own edge, but this
                        assumes the underlying piece (e.g. a mounting board) is
                        basically convex — off by default since most tablets
                        aren't board-backed, and turning it on for an irregular
                        board shape would bridge across real concavities too.
                        Only matters when --grow-chroma > 0.
  --nas                 Run legacy NAS workflow instead of local mode
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove

from erode_masks import erode_alpha

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
        "--hard-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Threshold rembg's soft alpha matte to a hard binary mask (morphological "
             "open + blur + threshold at 127) instead of a continuous confidence "
             "gradient (default: on). Pass --no-hard-mask to keep the raw soft matte. "
             "The soft matte blends real edge colour with backdrop colour at partial "
             "alpha around the whole silhouette; COLMAP has no notion of alpha and "
             "only sees RGB, so that blended ring survives as photo-consistent (but "
             "wrong) content and can triangulate into speckled noise at the "
             "reconstructed mesh's edges. Hard-mask removes the blend so background "
             "pixels are fully replaced by --background instead of blending with it.",
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
        "--chroma-threshold",
        type=int,
        default=0,
        metavar="N",
        help="Zero out foreground pixels whose chroma (max(R,G,B) - min(R,G,B)) is "
             "below N (0-255). 0 = disabled (default). Removes neutral/grey backdrop "
             "bleed regardless of brightness, which can be easier to dial in than "
             "--black-threshold/--white-threshold separately. Only safe if the "
             "tablet itself has some real colour. Try 15-30 to start.",
    )
    p.add_argument(
        "--edge-band",
        type=int,
        default=0,
        metavar="N",
        help="Restrict --black/white/value/chroma-threshold to within N pixels of the "
             "mask edge (default: 0 = apply across the whole foreground). Use this to "
             "stop the thresholds from eating into the interior of the tablet while "
             "still cleaning up background bleed at the mask boundary.",
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
        "--erode-px",
        type=int,
        default=0,
        metavar="N",
        help="Shrink the foreground mask inward by N pixels after segmentation "
             "(0-255, default: 0 = off). Trims thin halos/fringe of background bleed "
             "left at the silhouette edge by rembg.",
    )
    p.add_argument(
        "--grow-chroma",
        type=int,
        default=0,
        metavar="N",
        help="Grow the mask to include any region touching the current foreground "
             "with chroma (max(R,G,B) - min(R,G,B)) >= N (0-255, default: 0 = off). "
             "Recovers an attached, differently-coloured piece (e.g. a mounting "
             "board) that rembg dropped as background, as long as the real backdrop "
             "is neutral (black/white/grey). Try 40-60.",
    )
    p.add_argument(
        "--grow-close-px",
        type=int,
        default=15,
        metavar="N",
        help="Morphological closing radius (px) for the colour mask used by "
             "--grow-chroma, to bridge small gaps/glare within the coloured "
             "region (default: 15).",
    )
    p.add_argument(
        "--grow-feather-px",
        type=int,
        default=3,
        metavar="N",
        help="Blur radius (px) softening the edge of the region added by "
             "--grow-chroma (default: 3).",
    )
    p.add_argument(
        "--grow-hull-fill",
        action="store_true",
        help="Replace the grown region with its filled convex hull (default: off). "
             "Bridges low-colour detail on top of it (e.g. a printed label) even if "
             "it touches the region's own edge. Assumes the underlying piece is "
             "basically convex — off by default since most tablets aren't "
             "board-backed. Only matters when --grow-chroma > 0.",
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


def _convex_hull_fill(mask: np.ndarray) -> np.ndarray:
    """Replace each connected component of a binary mask with its filled convex
    hull.

    A simple hole-fill only recovers a gap that's fully enclosed by foreground
    on every side — it can't help if a low-colour spot (e.g. a label with no
    colour margin on one edge) forms an open notch reaching the region's own
    boundary rather than a sealed hole. Taking the convex hull instead bridges
    that too, on the assumption that the underlying object (a flat mounting
    board) is basically convex — true for a standard rectangular/oval mat, but
    this will bridge across *any* concavity in the recovered region, not just
    small ones, so it's a real assumption, not a free correctness win."""
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask_u8)
    for contour in contours:
        cv2.drawContours(filled, [cv2.convexHull(contour)], -1, 255, thickness=cv2.FILLED)
    return filled > 0


def _grow_mask_by_color(alpha: np.ndarray, src_rgb: np.ndarray, chroma_threshold: int,
                        close_px: int = 15, feather_px: int = 3,
                        hull_fill: bool = False) -> np.ndarray:
    """Expand alpha to cover any region touching the current foreground that has
    real colour (chroma = max(R,G,B) - min(R,G,B) >= chroma_threshold), whatever
    that colour is.

    Chroma, not HSV saturation, is what separates "colourful" from "neutral"
    here: saturation is a *ratio* (chroma / value), so it blows up from ordinary
    sensor noise on near-black pixels — a black backdrop can read as strongly
    "saturated" despite having almost no actual colour. Chroma stays low there
    since it looks at the raw channel spread instead.

    This targets a specific rembg failure mode: its saliency model picks out
    "the interesting object" and drops an attached, differently-coloured piece
    (e.g. a mounting board) as background. Since no hue is hardcoded — only
    "not neutral" — this generalises across any tablet/board colour combination,
    as long as the actual backdrop stays neutral (black/white/grey).

    Only colour-mask components that touch the existing foreground are kept, so
    unrelated coloured clutter elsewhere in frame isn't pulled in. If hull_fill
    is set, the kept region is then convex-hulled (see _convex_hull_fill) to
    bridge low-colour detail on top of it, like a printed label — off by
    default, since it assumes the recovered piece is convex."""
    if chroma_threshold <= 0:
        return alpha

    rgb16 = src_rgb.astype(np.int16)
    chroma = rgb16.max(axis=2) - rgb16.min(axis=2)
    color_mask = (chroma >= chroma_threshold).astype(np.uint8)
    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

    fg = alpha > 0
    num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
    keep = np.zeros_like(color_mask, dtype=bool)
    for label in range(1, num_labels):
        component = labels == label
        if np.any(component & fg):
            keep |= component

    if hull_fill:
        keep = _convex_hull_fill(keep)

    grown = keep.astype(np.uint8) * 255
    if feather_px > 0:
        k = 2 * feather_px + 1
        grown = cv2.GaussianBlur(grown, (k, k), 0)
    return np.maximum(alpha, grown)


def _apply_pixel_thresholds(rgba: Image.Image, src_rgb: np.ndarray,
                             black_threshold: int, white_threshold: int,
                             value_threshold: int, chroma_threshold: int,
                             edge_band: int = 0) -> Image.Image:
    """Zero out alpha for foreground pixels darker than black_threshold, brighter
    (mean RGB) than white_threshold, brighter (HSV Value = max channel) than
    value_threshold, or less colourful (chroma = max(R,G,B) - min(R,G,B)) than
    chroma_threshold. Any of the four may be 0/disabled.

    chroma_threshold targets the same neutral/grey backdrop bleed that
    black/white_threshold do, but in one pass instead of two brightness bands —
    it only assumes the tablet itself has some real colour, not a particular
    tone. Only safe to use where that holds; a grey/monochrome tablet would be
    zeroed out too.

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
    if chroma_threshold > 0:
        rgb16 = src_rgb.astype(np.int16)
        chroma = rgb16.max(axis=2) - rgb16.min(axis=2)
        alpha[band & (chroma < chroma_threshold)] = 0
    out = rgba.copy()
    out.putalpha(Image.fromarray(alpha))
    return out


def _multi_pass_remove(src_img: Image.Image, session, passes: int,
                       seg_scale: float = 1.0, hard_mask: bool = True) -> Image.Image:
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
        result = remove(current, session=session, post_process_mask=hard_mask)
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
                      value_threshold: int = 0, chroma_threshold: int = 0,
                      edge_band: int = 0,
                      passes: int = 1, seg_scale_pct: int = 100,
                      erode_px: int = 0, grow_chroma: int = 0,
                      grow_close_px: int = 15, grow_feather_px: int = 3,
                      grow_hull_fill: bool = False, hard_mask: bool = True) -> None:
    """Remove background from src and write to dst with the chosen fill."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_img = Image.open(src)
    seg_scale = max(1, min(100, seg_scale_pct)) / 100.0
    result = _multi_pass_remove(src_img, session, passes, seg_scale, hard_mask)

    need_src_rgb = (black_threshold > 0 or white_threshold > 0
                     or value_threshold > 0 or chroma_threshold > 0
                     or grow_chroma > 0)
    src_rgb = np.array(src_img.convert("RGB")) if need_src_rgb else None

    if grow_chroma > 0:
        alpha = _grow_mask_by_color(np.array(result.split()[3]), src_rgb,
                                     grow_chroma, grow_close_px, grow_feather_px,
                                     grow_hull_fill)
        result = result.copy()
        result.putalpha(Image.fromarray(alpha))

    if black_threshold > 0 or white_threshold > 0 or value_threshold > 0 or chroma_threshold > 0:
        result = _apply_pixel_thresholds(result, src_rgb, black_threshold,
                                          white_threshold, value_threshold,
                                          chroma_threshold, edge_band)

    if erode_px > 0:
        alpha = erode_alpha(np.array(result.split()[3]), erode_px)
        result = result.copy()
        result.putalpha(Image.fromarray(alpha))

    if background == "transparent":
        out = dst.with_suffix(".png")
        result.save(out, "PNG", compress_level=1)
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
              value_threshold: int = 0, chroma_threshold: int = 0,
              edge_band: int = 0, passes: int = 1,
              stride: int = 1, seg_scale_pct: int = 100, erode_px: int = 0,
              grow_chroma: int = 0, grow_close_px: int = 15,
              grow_feather_px: int = 3, grow_hull_fill: bool = False,
              hard_mask: bool = True) -> None:
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
    print(f"[info]  Hard mask: {'on' if hard_mask else 'off (raw soft matte)'}")
    if stride > 1:
        print(f"[info]  Stride: {stride} (keeping every {stride}th image per subfolder)")
    if black_threshold > 0:
        print(f"[info]  Black threshold: {black_threshold}")
    if white_threshold > 0:
        print(f"[info]  White threshold: {white_threshold}")
    if value_threshold > 0:
        print(f"[info]  Value threshold: {value_threshold}")
    if chroma_threshold > 0:
        print(f"[info]  Chroma threshold: {chroma_threshold}")
    if edge_band > 0:
        print(f"[info]  Edge band: {edge_band}px")
    if passes > 1:
        print(f"[info]  Passes: {passes}")
    if seg_scale_pct < 100:
        print(f"[info]  Segmentation input scale: {seg_scale_pct}% (output stays full res)")
    if erode_px > 0:
        print(f"[info]  Mask erosion: {erode_px}px")
    if grow_chroma > 0:
        print(f"[info]  Grow by colour: chroma >= {grow_chroma} "
              f"(close {grow_close_px}px, feather {grow_feather_px}px"
              f"{', hull-fill' if grow_hull_fill else ''})")
    print(f"[rembg] Loading model '{model}' ...")

    session = new_session(model)

    print(f"[rembg] Processing {len(pairs)} image(s) ...")
    errors = 0
    for i, (src, rel) in enumerate(pairs, 1):
        dst = output_dir / rel
        try:
            remove_background(src, dst, session, background,
                               black_threshold=black_threshold, white_threshold=white_threshold,
                               value_threshold=value_threshold, chroma_threshold=chroma_threshold,
                               edge_band=edge_band,
                               passes=passes, seg_scale_pct=seg_scale_pct, erode_px=erode_px,
                               grow_chroma=grow_chroma, grow_close_px=grow_close_px,
                               grow_feather_px=grow_feather_px, grow_hull_fill=grow_hull_fill,
                               hard_mask=hard_mask)
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
            chroma_threshold: int = 0,
            edge_band: int = 0, passes: int = 1, seg_scale_pct: int = 100,
            erode_px: int = 0, grow_chroma: int = 0, grow_close_px: int = 15,
            grow_feather_px: int = 3, grow_hull_fill: bool = False,
            hard_mask: bool = True) -> None:
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    source_folder = _most_recent_folder(NAS_DATA)
    folder_name = source_folder.name

    local_src = LOCAL_DATA / folder_name
    _copy_to_local(source_folder, local_src)

    local_masked = LOCAL_DATA / f"{folder_name}_masked"
    run_local(local_src, local_masked, model, background,
              black_threshold=black_threshold, white_threshold=white_threshold,
              value_threshold=value_threshold, chroma_threshold=chroma_threshold,
              edge_band=edge_band, passes=passes,
              seg_scale_pct=seg_scale_pct, erode_px=erode_px,
              grow_chroma=grow_chroma, grow_close_px=grow_close_px,
              grow_feather_px=grow_feather_px, grow_hull_fill=grow_hull_fill,
              hard_mask=hard_mask)

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
                chroma_threshold=args.chroma_threshold,
                edge_band=args.edge_band, passes=args.passes,
                seg_scale_pct=args.seg_scale_pct, erode_px=args.erode_px,
                grow_chroma=args.grow_chroma, grow_close_px=args.grow_close_px,
                grow_feather_px=args.grow_feather_px, grow_hull_fill=args.grow_hull_fill,
                hard_mask=args.hard_mask)
    else:
        run_local(Path(args.input), Path(args.output), args.model, args.background,
                  black_threshold=args.black_threshold, white_threshold=args.white_threshold,
                  value_threshold=args.value_threshold, chroma_threshold=args.chroma_threshold,
                  edge_band=args.edge_band,
                  passes=args.passes, stride=args.stride, seg_scale_pct=args.seg_scale_pct,
                  erode_px=args.erode_px, grow_chroma=args.grow_chroma,
                  grow_close_px=args.grow_close_px, grow_feather_px=args.grow_feather_px,
                  grow_hull_fill=args.grow_hull_fill, hard_mask=args.hard_mask)


if __name__ == "__main__":
    main()
