"""Session folder layout and input-structure detection."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .options import IMAGE_EXTS


def is_dir(p: Path) -> bool:
    """Path.is_dir() that treats unreadable parents (PermissionError) as False."""
    try:
        return Path(p).is_dir()
    except OSError:
        return False


def is_file(p: Path) -> bool:
    try:
        return Path(p).is_file()
    except OSError:
        return False


def has_images(d: Path) -> bool:
    try:
        return any(f.suffix.lower() in IMAGE_EXTS for f in d.rglob("*") if f.is_file())
    except OSError:
        return False


def detect_sides(input_dir: Path) -> list[str]:
    if not is_dir(input_dir):
        return []
    try:
        subs = list(input_dir.iterdir())
    except OSError:
        return []
    return sorted(sub.name for sub in subs if is_dir(sub) and has_images(sub))


def default_output_for(input_dir: str) -> str:
    """``<input>_recon`` next to the input folder."""
    p = Path(input_dir)
    return str(p.parent / (p.name + "_recon"))


def describe_sides(sides: list[str]) -> tuple[str, str, str]:
    """(side1, side2, status line) for the sides found by ``detect_sides``."""
    if len(sides) >= 2:
        return sides[0], sides[1], f"Detected sides: {', '.join(sides)}"
    if len(sides) == 1:
        return sides[0], "", f"Detected: 1 side ({sides[0]}), no alignment stage"
    return "", "", "Flat structure — images at root, no alignment stage"


class SessionPaths:
    """Where each stage reads and writes, derived from the current settings.

    Reads the settings live, so it always reflects the latest input/output
    folder and side names.
    """

    def __init__(self, settings: Mapping):
        self.s = settings

    def _get(self, key: str) -> str:
        return str(self.s.get(key, "")).strip()

    def session_dir(self) -> Path:
        out = self._get("output_var")
        if out:
            return Path(out)
        inp = self._get("input_var")
        if inp:
            return Path(default_output_for(inp))
        return Path.home() / "tablet_recon"

    def processed_dir(self) -> Path:
        return self.session_dir() / "processed"

    def masks_dir(self) -> Path:
        # Matches process_photos.py's default --mask-dir (<output>_masks,
        # a sibling of --output) when --output is processed_dir().
        return self.session_dir() / "processed_masks"

    def colmap_dir(self, side: str | None = None) -> Path:
        if side:
            return self.session_dir() / f"colmap_{side}"
        return self.session_dir() / "colmap_out"

    def merged_ply(self) -> Path:
        return self.session_dir() / "aligned_cloud" / "merged_fpfh.ply"

    def icp_report_path(self) -> Path:
        return self.merged_ply().parent / "icp_report.json"

    def recon_dir(self) -> Path:
        return self.session_dir() / "recon"

    def intrinsics_path(self) -> Path:
        return self.session_dir() / "camera_intrinsics.json"

    def active_sides(self) -> tuple[str, str]:
        return self._get("side1_var"), self._get("side2_var")

    def input_ply_for_recon(self) -> Path:
        s1, s2 = self.active_sides()
        if s2:
            return self.merged_ply()
        return self.colmap_dir(s1 if s1 else None) / "fused.ply"

    def side1_camera_centers_path(self) -> Path:
        s1, _ = self.active_sides()
        return self.colmap_dir(s1 if s1 else None) / "camera_centers.json"

    def expected_output_for_stage(self, idx: int) -> Path:
        s1, _ = self.active_sides()
        if idx == 0:
            return self.processed_dir()
        if idx == 1:
            return (self.colmap_dir(s1) if s1 else self.colmap_dir()) / "fused.ply"
        if idx == 2:
            return self.merged_ply()
        return self.recon_dir() / "recon_mesh_recon.obj"

    def stage_output_ready(self, idx: int) -> bool:
        if idx == 1:
            # Both sides' COLMAP output must be ready in a two-sided session,
            # not just side1's (which is all expected_output_for_stage(1) checks).
            s1, s2 = self.active_sides()
            paths = [self.colmap_dir(s1) / "fused.ply" if s1 else self.colmap_dir() / "fused.ply"]
            if s2:
                paths.append(self.colmap_dir(s2) / "fused.ply")
            try:
                return all(p.is_file() and p.stat().st_size > 0 for p in paths)
            except OSError:
                return False
        p = self.expected_output_for_stage(idx)
        try:
            if idx == 0:
                return p.is_dir() and has_images(p)
            return p.is_file() and p.stat().st_size > 0
        except OSError:
            return False
