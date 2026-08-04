#!/usr/bin/env python3
"""Convert a raw MISHA multispectral export into small web-deployable assets.

MISHA (Multispectral Imaging System for Historical Artifacts) is an
independent imaging project at RIT's Cultural Heritage Imaging, Preservation,
and Research (CHIPR) program (https://www.rit.edu/chipr/misha) — it is NOT
one of this repo's own capture systems (see 2D/ and 3D/), it's an external
collaborator whose exports this site displays. A raw MISHA export is a set of
single-wavelength 16-bit TIFFs, one per captured band, typically named like
"raw_365 nm.tif". A full export runs ~40MB/band x ~16 bands = ~600MB+, so it
must never be committed as-is — this script produces small derived assets
under website/artifacts/<artifact-id>/msi/ instead. Keep the raw export
itself under the repo's gitignored data/ folder (e.g.
data/msi/<artifact-id>-raw/), the same convention the 2D/3D pipelines use for
their own raw captures.

Usage:

    python3 tools/build_msi_assets.py <raw-export-dir> <artifact-id> \\
        [--max-dim 2048] [--low-pct 0.5] [--high-pct 99.5]

<raw-export-dir> is searched recursively for files named like
"raw_<wavelength> nm.tif" (case-insensitive, whitespace before "nm"
optional) — point it at the zip's extracted root, a "Raw/" folder, or the
"raw/" folder directly, all work. The wavelength is read from the filename,
not any embedded metadata (a MISHA export may also include an ENVI-format
"<name>_cube"/".hdr" pair — that cube's own wavelength metadata is typically
empty in practice, and its bands are a resampled 8-bit copy of the same raw
TIFFs, so it isn't used by this script at all).

Each band is independently processed:
  1. Read as a 16-bit grayscale array (tifffile, not Pillow, since MISHA
     TIFFs may be LZW-compressed and multi-page).
  2. Contrast-stretched: clipped to the [--low-pct, --high-pct] percentile
     range of that band's own pixel values, then linearly rescaled to 0-255.
     This is a LINEAR stretch, not histogram equalization/CLAHE — nonlinear
     stretches can fabricate apparent structure, which matters here because
     the whole point of this feature is surfacing real faint signal (e.g.
     undertext) that a curator or visitor can trust.
  3. Downsampled (Lanczos) so its long edge is --max-dim.
  4. Saved as lossless WebP (lossy compression risks the same
     fabricate-or-hide-real-signal problem as nonlinear stretching).

Output, under website/artifacts/<artifact-id>/msi/:
  bands/<wavelength>nm.webp   one per detected band
  msi_manifest.json           regenerated in full every run — safe to rerun

This script never reads or writes msi/recipes.json — that file is
hand-authored by a curator (band-weight/exposure/contrast presets for the
web viewer's equalizer) and must survive reruns of this script untouched.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

WEBSITE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WEBSITE_ROOT / "artifacts"

BAND_FILE_RE = re.compile(r"raw_(\d+)\s*nm\.tif$", re.IGNORECASE)
MISHA_SOURCE = "MISHA (Multispectral Imaging System for Historical Artifacts), RIT CHIPR — https://www.rit.edu/chipr/misha"


def find_band_files(raw_dir):
    """Return [(wavelength_nm, path), ...] sorted by wavelength."""
    bands = []
    for path in raw_dir.rglob("*.tif"):
        match = BAND_FILE_RE.search(path.name)
        if not match:
            continue
        bands.append((int(match.group(1)), path))
    bands.sort(key=lambda b: b[0])
    return bands


def region_for_wavelength(wavelength):
    if wavelength < 400:
        return "uv"
    if wavelength <= 700:
        return "visible"
    return "nir"


def process_band(path, low_pct, high_pct, max_dim):
    """Read a 16-bit band TIFF, contrast-stretch and downsample it.

    Returns (PIL.Image in mode "L", stretch metadata dict).
    """
    arr = tifffile.imread(str(path))
    if arr.ndim > 2:
        # Defend against an accidental multi-page/multi-channel TIFF — MISHA
        # raw bands are single-channel, so the first plane is the real data.
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]

    arr = arr.astype(np.float32)
    low, high = np.percentile(arr, [low_pct, high_pct])
    span = max(high - low, 1.0)
    stretched = np.clip((arr - low) / span * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(stretched, mode="L")
    if max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    stretch_meta = {
        "lowPercentile": low_pct,
        "highPercentile": high_pct,
        "lowDN": round(float(low), 1),
        "highDN": round(float(high), 1),
    }
    return img, stretch_meta


def build(raw_dir, artifact_id, max_dim, low_pct, high_pct):
    if not raw_dir.exists():
        print(f"raw export directory not found: {raw_dir}", file=sys.stderr)
        return 1

    band_files = find_band_files(raw_dir)
    if not band_files:
        print(f"no 'raw_<wavelength> nm.tif' files found under {raw_dir}", file=sys.stderr)
        return 1

    msi_dir = ARTIFACTS_DIR / artifact_id / "msi"
    bands_dir = msi_dir / "bands"
    bands_dir.mkdir(parents=True, exist_ok=True)

    band_entries = []
    for wavelength, path in band_files:
        print(f"processing {wavelength}nm ({path.name})…")
        img, stretch_meta = process_band(path, low_pct, high_pct, max_dim)

        out_name = f"{wavelength}nm.webp"
        out_path = bands_dir / out_name
        img.save(out_path, "WEBP", lossless=True, method=6)

        band_entries.append({
            "wavelength": wavelength,
            "region": region_for_wavelength(wavelength),
            "file": f"bands/{out_name}",
            "width": img.width,
            "height": img.height,
            "stretch": stretch_meta,
        })

    manifest = {
        "artifactId": artifact_id,
        "source": MISHA_SOURCE,
        "sourceNote": (
            "Each band is independently contrast-stretched for web visibility "
            "— pixel brightness is not radiometrically calibrated and is not "
            "comparable across bands."
        ),
        "maxDimension": max_dim,
        "bands": band_entries,
    }
    manifest_path = msi_dir / "msi_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    recipes_path = msi_dir / "recipes.json"
    highlights_note = "found" if recipes_path.exists() else "none yet — the web viewer opens in free-adjust mode"
    print(f"curator recipes.json: {highlights_note}")
    print(f"Wrote {len(band_entries)} band(s) to {manifest_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Build web assets from a raw MISHA multispectral export.")
    parser.add_argument("raw_export_dir", type=Path, help="Folder to search recursively for raw_<nm>nm.tif files")
    parser.add_argument("artifact_id", help="Artifact folder name under website/artifacts/")
    parser.add_argument("--max-dim", type=int, default=2048, help="Max long-edge pixel size for output bands (default: 2048)")
    parser.add_argument("--low-pct", type=float, default=0.5, help="Low percentile clip for contrast stretch (default: 0.5)")
    parser.add_argument("--high-pct", type=float, default=99.5, help="High percentile clip for contrast stretch (default: 99.5)")
    args = parser.parse_args()

    return build(args.raw_export_dir, args.artifact_id, args.max_dim, args.low_pct, args.high_pct)


if __name__ == "__main__":
    sys.exit(main())
