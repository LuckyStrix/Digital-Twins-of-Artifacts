"""
registration_state.py
=====================
The on-disk state of two-sided alignment (register_sides.py), in one place.
Standard library only, so the launcher can use it without loading OpenCV.

Rules - explicit, never inferred from timestamps (copying a scan folder to a
NAS or another machine changes those):

  * side2/maps/unregistered/ exists  ->  it holds side2's TRUE originals.
    Only the modeling pipeline removes it (clear(), called by Stage 4 when it
    writes new maps), because only modeling knows the maps are fresh.
  * unregistered/ does not exist      ->  side2's current maps ARE the originals.
  * registration.json "applied": true ->  side2's maps are aligned to the side1
    mask whose SHA-256 is recorded there; a different side1 mask (side1 was
    re-modelled) means the alignment is stale.
  * .registration_in_progress exists  ->  a run died while swapping maps in;
    side2's maps are a mix and must be restored from unregistered/.
"""

import hashlib
import json
import shutil
from pathlib import Path

MASK_NAME    = "AlphaMask_render.tiff"
MAP_NAMES    = ("AlphaMask_render.tiff", "DiffuseMap_render.tiff",
                "NormalMap_render.tiff", "SpecularMap_render.tiff",
                "RoughnessMap_render.tiff", "HeightMap_render.tiff")
ORIG_SUBDIR  = "unregistered"
INFO_NAME    = "registration.json"
OVERLAY_NAME = "registration_overlay.png"
MARKER_NAME  = ".registration_in_progress"
TEMP_SUFFIX  = ".aligned-tmp"


def file_hash(path: Path) -> str:
    """SHA-256 of a file's contents (unaffected by copying or timestamps)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_info(maps_dir: Path) -> dict:
    try:
        return json.loads((Path(maps_dir) / INFO_NAME).read_text())
    except (OSError, ValueError):
        return {}


def is_current(side1_maps: Path, side2_maps: Path) -> bool:
    """side2 is aligned, not mid-swap, and aligned to side1's current mask."""
    side1_maps, side2_maps = Path(side1_maps), Path(side2_maps)
    info = read_info(side2_maps)
    mask1 = side1_maps / MASK_NAME
    return bool(info.get("applied")
                and not (side2_maps / MARKER_NAME).exists()
                and (side2_maps / ORIG_SUBDIR / MASK_NAME).is_file()
                and mask1.is_file()
                and info.get("side1_mask_sha256") == file_hash(mask1))


def clear(maps_dir: Path) -> None:
    """Forget any alignment state in `maps_dir`. Called by the modeling
    pipeline right before it writes fresh render maps there."""
    maps_dir = Path(maps_dir)
    shutil.rmtree(maps_dir / ORIG_SUBDIR, ignore_errors=True)
    for name in (INFO_NAME, OVERLAY_NAME, MARKER_NAME):
        (maps_dir / name).unlink(missing_ok=True)
    for leftover in maps_dir.glob("*" + TEMP_SUFFIX):
        leftover.unlink(missing_ok=True)
