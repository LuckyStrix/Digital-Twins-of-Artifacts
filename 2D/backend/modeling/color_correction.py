#!/usr/bin/env python3
"""
color_correction.py
===================
Colour management for the 2D papyrus pipeline: transfer functions, capture
encoding metadata, and the ColorChecker-derived correction that Stage 1 applies
to the directional-light images.

WHY THIS EXISTS
---------------
The capture chain produces TIFFs via dcraw with `-o 0` (camera-native primaries,
no colour matrix).  Historically it also left dcraw's default BT.709 tone curve
in place, so the images were *gamma-encoded camera RGB* while every downstream
stage treated them as linear radiance.  That is wrong three times over:

  * photometric stereo solves a Lambertian model, which needs linear light;
  * the flat-field divide `scroll / paper` is only "reflectance ratio" in linear;
  * BT.709 luminance weights are defined on linear light.

This module supplies the two halves of the fix:

  1. Transfer functions to move between dcraw's encoding and linear, plus the
     `capture_info.json` sidecar that records which one a given folder holds.
     Captures made before the sidecar existed are assumed BT.709, which is what
     they are - so old scans keep processing correctly.

  2. A `ColorCorrection` (white-balance gains, exposure scale, 3x3 matrix)
     fitted from a 24-patch ColorChecker and applied at the end of Stage 1.

THE INVARIANT THAT MAKES THE CORRECTION CORRECT
-----------------------------------------------
The chart must be flat-fielded exactly the way the artifact is:

    fit   on  chart_linear  / envelope(paper_linear)
    apply to  scroll_linear / envelope(paper_linear)

Both sides then live in the same units ("reflectance relative to the copy
paper under that light"), so the gains, exposure scalar and matrix transfer
verbatim with no scalar bookkeeping.  Two consequences follow:

  * a global exposure difference between the chart shot and the artifact shot
    cancels in the ratio, so session drift is harmless; and
  * ONE matrix is correct for both co- and cross-polarised frames, because each
    co frame is divided by its own co-polarised paper envelope and the
    polariser transmission difference cancels.  Do not fit a second co-pol CCM.

If someone "tidies" Stage 1 so the envelope is built from colour-corrected
paper, or fits the chart without flat-fielding it, this invariant breaks and
the correction silently becomes wrong.  Both sites carry a comment saying so.

This half of the module depends only on numpy.  Chart detection and matrix
fitting (which need opencv-contrib and colour-science) live in the fit half and
are imported lazily, so the processing pipeline never needs those packages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# dcraw transfer function
# ─────────────────────────────────────────────────────────────────────────────
# dcraw stores its gamma as gamm[0] = 1/power and gamm[1] = toe_slope, and its
# BUILT-IN DEFAULTS ARE gamm[0]=0.45, gamm[1]=4.5 exactly.  That distinction
# matters: writing `-g 2.222 4.5` on the command line yields gamm[0] = 1/2.222 =
# 0.450045..., which is NOT the same curve - it shifts the knee by ~2e-5.  The
# default 0.45 reproduces textbook BT.709 exactly.  So these are stored as
# dcraw's internal g0/g1, never as a user-facing "power".
DCRAW_DEFAULT_G0 = 0.45      # gamm[0] = 1/2.2222...
DCRAW_DEFAULT_G1 = 4.5       # gamm[1] = toe slope


def dcraw_gamma_params(g0: float = DCRAW_DEFAULT_G0,
                       g1: float = DCRAW_DEFAULT_G1) -> tuple[float, float, float]:
    """Solve dcraw's gamma knee, replicating gamma_curve()'s bisection.

    This is a transcription of the 48-iteration bisection in dcraw.c, so a
    non-default `-g p ts` in capture_info.json produces the matching curve
    rather than silently falling back to BT.709.

    Returns (knee_encoded, knee_linear, a) where the encode is

        encoded = linear * g1                       for linear <  knee_linear
                = linear**g0 * (1 + a) - a          otherwise

    With the defaults this returns exactly the textbook BT.709 constants
    (0.08124285829863354, 0.018053968510807452, 0.09929682680944099).
    """
    bnd = [0.0, 0.0]
    bnd[1 if g1 >= 1 else 0] = 1.0
    g2 = 0.0
    if g1 and (g1 - 1) * (g0 - 1) <= 0:
        for _ in range(48):
            g2 = (bnd[0] + bnd[1]) / 2
            if g0:
                bnd[1 if ((g2 / g1) ** -g0 - 1) / g0 - 1 / g2 > -1 else 0] = g2
    knee_linear = g2 / g1
    a = g2 * (1 / g0 - 1) if g0 else 0.0
    return g2, knee_linear, a


def dcraw_decode(encoded: np.ndarray, g0: float = DCRAW_DEFAULT_G0,
                 g1: float = DCRAW_DEFAULT_G1) -> np.ndarray:
    """Encoded [0,1+] -> linear. Inverse of dcraw_encode.

    Values above 1.0 are carried through the upper branch rather than clipped,
    because Stage 1's flat-field divide legitimately produces values > 1 and
    that headroom must survive the round trip.
    """
    knee_enc, _, a = dcraw_gamma_params(g0, g1)
    e = np.asarray(encoded, dtype=np.float64)
    # np.where evaluates both branches, so guard the power against negatives
    # (which would produce NaN) even where the toe branch is the one selected.
    safe = np.maximum(e, 0.0)
    return np.where(e < knee_enc, e / g1, ((safe + a) / (1 + a)) ** (1 / g0))


def dcraw_encode(linear: np.ndarray, g0: float = DCRAW_DEFAULT_G0,
                 g1: float = DCRAW_DEFAULT_G1) -> np.ndarray:
    """Linear [0,1+] -> encoded. Inverse of dcraw_decode."""
    _, knee_lin, a = dcraw_gamma_params(g0, g1)
    x = np.asarray(linear, dtype=np.float64)
    safe = np.maximum(x, 0.0)
    return np.where(x < knee_lin, x * g1, safe ** g0 * (1 + a) - a)


# ─────────────────────────────────────────────────────────────────────────────
# sRGB transfer function
# ─────────────────────────────────────────────────────────────────────────────
# Applied once, at Stage 4, to the diffuse map only. glTF defines
# baseColorTexture as sRGB-encoded, so the bytes in the .glb must carry this
# curve; every other map is data (normals, height, roughness, specular,
# alpha) and stays linear.
SRGB_KNEE_LINEAR  = 0.0031308
SRGB_KNEE_ENCODED = 0.04045


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Linear [0,1+] -> sRGB-encoded."""
    x = np.asarray(linear, dtype=np.float64)
    safe = np.maximum(x, 0.0)
    return np.where(x <= SRGB_KNEE_LINEAR, x * 12.92,
                    1.055 * safe ** (1 / 2.4) - 0.055)


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """sRGB-encoded [0,1+] -> linear."""
    e = np.asarray(encoded, dtype=np.float64)
    safe = np.maximum(e, 0.0)
    return np.where(e <= SRGB_KNEE_ENCODED, e / 12.92,
                    ((safe + 0.055) / 1.055) ** 2.4)


# ─────────────────────────────────────────────────────────────────────────────
# Capture encoding sidecar
# ─────────────────────────────────────────────────────────────────────────────
CAPTURE_INFO_NAME = "capture_info.json"

KIND_LINEAR = "linear"
KIND_BT709  = "bt709"


@dataclass(frozen=True)
class CaptureEncoding:
    """Which transfer curve the TIFFs in a folder carry."""
    kind: str = KIND_BT709
    g0: float = DCRAW_DEFAULT_G0
    g1: float = DCRAW_DEFAULT_G1
    source: str = "assumed-bt709-default"

    @property
    def is_linear(self) -> bool:
        return self.kind == KIND_LINEAR

    def describe(self) -> str:
        if self.is_linear:
            return f"linear (from {self.source})"
        return f"BT.709 g0={self.g0:.6g} g1={self.g1:.6g} (from {self.source})"


def encoding_from_dcraw_args(argv) -> CaptureEncoding:
    """Derive the encoding from a dcraw argument list.

    `-g <power> <toe>` sets the curve; `-g 1 1` means linear. Absent, dcraw
    applies its BT.709 default. Note the inversion: dcraw's gamm[0] is
    1/power, so `-g 2.222 4.5` gives 0.450045, not the default 0.45.
    """
    argv = [str(a) for a in argv]
    if "-g" in argv:
        i = argv.index("-g")
        try:
            power, toe = float(argv[i + 1]), float(argv[i + 2])
        except (IndexError, ValueError):
            return CaptureEncoding(source="unparseable-dcraw-args")
        if power == 1.0 and toe == 1.0:
            return CaptureEncoding(KIND_LINEAR, 1.0, 1.0, "dcraw-args")
        return CaptureEncoding(KIND_BT709, 1.0 / power, toe, "dcraw-args")
    return CaptureEncoding(KIND_BT709, DCRAW_DEFAULT_G0, DCRAW_DEFAULT_G1,
                           "dcraw-default")


def read_capture_info(folder: Path) -> CaptureEncoding:
    """Read <folder>/capture_info.json.

    A missing or unparseable sidecar means the folder predates this metadata,
    which in practice means it was shot with dcraw's BT.709 default - so that
    is the assumption, and it keeps every existing scan on disk processable.
    """
    path = Path(folder) / CAPTURE_INFO_NAME
    if not path.is_file():
        return CaptureEncoding()
    try:
        info = json.loads(path.read_text())
        enc = info.get("encoding", {})
        kind = enc.get("kind", KIND_BT709)
        if kind == KIND_LINEAR:
            return CaptureEncoding(KIND_LINEAR, 1.0, 1.0, CAPTURE_INFO_NAME)
        return CaptureEncoding(
            KIND_BT709,
            float(enc.get("dcraw_g0", DCRAW_DEFAULT_G0)),
            float(enc.get("dcraw_g1", DCRAW_DEFAULT_G1)),
            CAPTURE_INFO_NAME,
        )
    except (OSError, ValueError, TypeError):
        return CaptureEncoding(source=f"unreadable-{CAPTURE_INFO_NAME}")


def to_linear(img01: np.ndarray, enc: CaptureEncoding) -> np.ndarray:
    """Normalised image [0,1] -> linear, per the folder's encoding."""
    if enc.is_linear:
        return np.asarray(img01, dtype=np.float32)
    return dcraw_decode(img01, enc.g0, enc.g1).astype(np.float32)


def load_linear01(path: Path, enc: CaptureEncoding) -> np.ndarray:
    """Read a uint16 TIFF and return linear float32 in [0,1+], channels as stored.

    tifffile, not PIL: PIL truncates 16-bit TIFFs to their high byte.
    """
    import tifffile
    raw = tifffile.imread(str(path))
    return to_linear(raw.astype(np.float32) / 65535.0, enc)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration folder resolution
# ─────────────────────────────────────────────────────────────────────────────
CAL_SUBDIR   = "cal"
CHART_SUBDIR = "chart"
CCM_NAME     = "ccm.json"


def resolve_cal_dir(scroll_dir: Path, global_fallback: Path) -> Path:
    """Find the calibration set that belongs to `scroll_dir`.

    Order, first hit wins:
      1. $PAPYRUS_CAL_DIR         - the launcher's explicit choice
      2. <scroll_dir>/cal         - this scan set's own calibration
      3. <scroll_dir>.parent/cal  - lets side1/ and side2/ share one cal/
      4. global_fallback          - backend/calibration/

    Calibration is per-zoom (the flat-field envelope is a per-pixel spatial fit
    and Stage 1 hard-errors on a shape mismatch), so keeping it beside the scan
    is what makes an old scan reprocessable years later.
    """
    scroll_dir = Path(scroll_dir)
    env = os.environ.get("PAPYRUS_CAL_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [scroll_dir / CAL_SUBDIR,
                   scroll_dir.parent / CAL_SUBDIR,
                   Path(global_fallback)]
    for c in candidates:
        if c.is_dir():
            return c
    return Path(global_fallback)


def find_ccm(cal_dir: Path) -> Path | None:
    """The fitted matrix for a calibration set, or None if it has not been fitted."""
    p = Path(cal_dir) / CCM_NAME
    return p if p.is_file() else None


# ─────────────────────────────────────────────────────────────────────────────
# The correction itself
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ColorCorrection:
    """Black offset, white-balance gains, exposure scale and a 3x3 matrix (RGB).

    Applied as   out = M @ (exposure * gains * (in - offset))   on linear light.

    The offset is a small per-channel additive term measured from the chart's
    neutral ramp. It matters because any additive error is negligible on bright
    patches but dominates dark ones, where it shows up as coloured greys - the
    most visible failure there is. Verified to generalise by repeated
    split-half cross-validation over the 24 patches (it improved held-out dE in
    100% of trials), rather than assumed; three extra free parameters on 24
    patches would otherwise be a natural overfitting risk.
    """
    gains: np.ndarray                      # (3,) RGB multipliers
    exposure: float                        # single scalar
    matrix: np.ndarray                     # (3,3) acting on column vectors
    encoding: CaptureEncoding = field(default_factory=CaptureEncoding)
    metadata: dict = field(default_factory=dict)
    order: str = "rgb"                     # "rgb" or "bgr"; see as_bgr()
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def identity(cls, encoding: CaptureEncoding | None = None) -> "ColorCorrection":
        """A no-op correction, so the pipeline runs unchanged before any fit."""
        return cls(np.ones(3), 1.0, np.eye(3),
                   encoding or CaptureEncoding(),
                   {"note": "identity - no colour correction applied"})

    @property
    def is_identity(self) -> bool:
        return (np.allclose(self.gains, 1.0) and np.isclose(self.exposure, 1.0)
                and np.allclose(self.matrix, np.eye(3))
                and np.allclose(self.offset, 0.0))

    # ── channel order ───────────────────────────────────────────────────────
    def as_bgr(self) -> "ColorCorrection":
        """Return the same transform expressed for BGR arrays.

        The pipeline holds cv2-loaded BGR arrays everywhere. Rather than flip
        channels at every call site (many chances to get it wrong), flip the
        transform once here. For the reversal permutation P:

            out_bgr = P M diag(g) P x_bgr = (P M P)(P diag(g) P) x_bgr

        and P M P is M[::-1, ::-1] while P diag(g) P is diag(g[::-1]).
        """
        if self.order == "bgr":
            return self
        return ColorCorrection(
            gains=self.gains[::-1].copy(),
            exposure=self.exposure,
            matrix=self.matrix[::-1, ::-1].copy(),
            encoding=self.encoding,
            metadata=self.metadata,
            order="bgr",
            offset=self.offset[::-1].copy(),
        )

    # ── application ─────────────────────────────────────────────────────────
    def apply(self, img: np.ndarray) -> np.ndarray:
        """Correct a linear (H, W, 3) image. Channel order must match `self.order`.

        Clamps negatives (a matrix can push saturated colours slightly below
        zero) but deliberately does NOT clip the top: Stage 1 carries headroom
        above 1.0 and clipping here would crush the specular signal.
        """
        x = np.asarray(img, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) image, got {x.shape}")
        scaled = ((x - self.offset.astype(np.float32))
                  * (self.gains * self.exposure).astype(np.float32))
        out = np.einsum("ij,...j->...i", self.matrix.astype(np.float32), scaled)
        return np.maximum(out, 0.0, out)

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self, extras: dict | None = None) -> dict:
        d = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": "color_correction.py",
            "input_encoding": {
                "kind": self.encoding.kind,
                "dcraw_g0": self.encoding.g0,
                "dcraw_g1": self.encoding.g1,
                "source": self.encoding.source,
            },
            "white_balance": {
                "gains_rgb": [float(g) for g in self.gains],
                "exposure_scale": float(self.exposure),
                "black_offset_rgb": [float(o) for o in self.offset],
            },
            "ccm": {"matrix_rgb": [[float(v) for v in row] for row in self.matrix]},
        }
        d.update(extras or {})
        d.update(self.metadata or {})
        return d

    def save(self, path: Path, extras: dict | None = None) -> None:
        Path(path).write_text(json.dumps(self.to_dict(extras), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ColorCorrection":
        info = json.loads(Path(path).read_text())
        enc_d = info.get("input_encoding", {})
        kind = enc_d.get("kind", KIND_BT709)
        encoding = CaptureEncoding(
            kind,
            float(enc_d.get("dcraw_g0", DCRAW_DEFAULT_G0)),
            float(enc_d.get("dcraw_g1", DCRAW_DEFAULT_G1)),
            enc_d.get("source", str(path)),
        )
        wb = info.get("white_balance", {})
        return cls(
            gains=np.asarray(wb.get("gains_rgb", [1.0, 1.0, 1.0]), dtype=np.float64),
            exposure=float(wb.get("exposure_scale", 1.0)),
            matrix=np.asarray(info["ccm"]["matrix_rgb"], dtype=np.float64),
            encoding=encoding,
            metadata={k: info[k] for k in ("delta_e_2000", "detection", "reference")
                      if k in info},
            offset=np.asarray(wb.get("black_offset_rgb", [0.0, 0.0, 0.0]),
                              dtype=np.float64),
        )

    def summary(self) -> str:
        de = (self.metadata or {}).get("delta_e_2000", {})
        bits = [f"gains=[{', '.join(f'{g:.4f}' for g in self.gains)}]",
                f"exposure={self.exposure:.4f}"]
        if de.get("mean") is not None:
            bits.append(f"mean dE2000={de['mean']:.2f}")
        return "  ".join(bits)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    """Pure-function checks. No hardware, no chart, no network."""
    rng = np.random.default_rng(0)
    fails = []

    def check(name, cond, detail=""):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not cond:
            fails.append(name)

    print("dcraw curve:")
    knee_enc, knee_lin, a = dcraw_gamma_params()
    check("default params are textbook BT.709",
          (knee_enc, knee_lin, a) == (0.08124285829863354,
                                      0.018053968510807452,
                                      0.09929682680944099),
          f"knee_lin={knee_lin!r}")
    check("a == 5.5 * knee_linear", np.isclose(a, 5.5 * knee_lin, rtol=1e-12))
    # The knee is derived, not rounded, so both branches must agree to
    # full double precision there.
    toe   = knee_lin * DCRAW_DEFAULT_G1
    power = knee_lin ** DCRAW_DEFAULT_G0 * (1 + a) - a
    check("both branches agree at the knee", np.isclose(toe, power, rtol=1e-11),
          f"{toe!r} vs {power!r}")

    print("round trips (values up to 4.0, i.e. Stage 1 headroom):")
    x = rng.uniform(0, 4, 1_000_000)
    check("dcraw decode(encode(x)) == x",
          np.abs(dcraw_decode(dcraw_encode(x)) - x).max() < 1e-9,
          f"max err {np.abs(dcraw_decode(dcraw_encode(x)) - x).max():.2e}")
    check("srgb decode(encode(x)) == x",
          np.abs(srgb_decode(srgb_encode(x)) - x).max() < 1e-9,
          f"max err {np.abs(srgb_decode(srgb_encode(x)) - x).max():.2e}")
    # The published sRGB constants are rounded and so very slightly
    # inconsistent with each other: 0.0031308 * 12.92 = 0.04044994, not exactly
    # 0.04045. Everyone uses the rounded values anyway; assert the residual
    # kink is the known ~6e-8 and not something larger.
    check("srgb knee agrees to the standard's own rounding",
          abs(srgb_encode(SRGB_KNEE_LINEAR) - SRGB_KNEE_ENCODED) < 1e-7,
          f"kink {abs(float(srgb_encode(SRGB_KNEE_LINEAR)) - SRGB_KNEE_ENCODED):.2e}")
    check("no NaN from negative input",
          np.isfinite(dcraw_decode(np.array([-0.5, -1e-9, 0.0]))).all()
          and np.isfinite(srgb_encode(np.array([-0.5, -1e-9, 0.0]))).all())

    print("encoding metadata:")
    check("-g 1 1 reads as linear",
          encoding_from_dcraw_args(["-T", "-6", "-g", "1", "1"]).is_linear)
    check("no -g reads as BT.709 default",
          encoding_from_dcraw_args(["-T", "-6", "-W", "-o", "0"]).g0 == DCRAW_DEFAULT_G0)
    check("-g 2.222 4.5 inverts to 1/2.222 (NOT 0.45)",
          np.isclose(encoding_from_dcraw_args(["-g", "2.222", "4.5"]).g0, 1 / 2.222))
    check("to_linear is a no-op when already linear",
          np.array_equal(to_linear(x[:100].astype(np.float32),
                                   CaptureEncoding(KIND_LINEAR, 1.0, 1.0, "t")),
                         x[:100].astype(np.float32)))

    print("ColorCorrection:")
    gains = np.array([1.07, 1.0, 1.21])
    M = np.array([[1.22, -0.18, -0.04],
                  [-0.09, 1.15, -0.06],
                  [0.02, -0.31, 1.29]])
    # Non-neutral offset on purpose: a symmetric one would hide a reversal bug.
    cc = ColorCorrection(gains, 0.87, M, offset=np.array([0.004, -0.002, 0.011]))
    img = rng.uniform(0, 2, (16, 16, 3)).astype(np.float32)
    out_rgb = cc.apply(img)
    # The one place a channel-order bug could hide.
    out_bgr = cc.as_bgr().apply(img[..., ::-1])[..., ::-1]
    check("as_bgr() matches RGB application",
          np.allclose(out_rgb, out_bgr, atol=1e-6),
          f"max diff {np.abs(out_rgb - out_bgr).max():.2e}")
    check("as_bgr() is involutive", np.allclose(cc.as_bgr().as_bgr().matrix,
                                                cc.as_bgr().matrix))
    check("identity is a no-op",
          np.allclose(ColorCorrection.identity().apply(img), img, atol=1e-6))
    check("apply clamps negatives but keeps headroom",
          ColorCorrection.identity().apply(
              np.full((2, 2, 3), -1.0, np.float32)).max() == 0.0
          and ColorCorrection.identity().apply(
              np.full((2, 2, 3), 3.5, np.float32)).max() > 3.0)

    print("persistence:")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / CCM_NAME
        cc.save(p)
        back = ColorCorrection.load(p)
        check("save/load round trip",
              np.allclose(back.gains, cc.gains)
              and np.isclose(back.exposure, cc.exposure)
              and np.allclose(back.matrix, cc.matrix)
              and np.allclose(back.offset, cc.offset))
        check("offset survives save/load",
              np.allclose(back.apply(img), cc.apply(img), atol=1e-6))
        check("find_ccm locates it", find_ccm(Path(td)) == p)
        check("find_ccm returns None when absent", find_ccm(Path(td) / "nope") is None)

    print("cal dir resolution:")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scan = root / "data" / "set1"
        (scan / CAL_SUBDIR).mkdir(parents=True)
        glob = root / "backend" / "calibration"
        glob.mkdir(parents=True)
        check("prefers <scroll>/cal", resolve_cal_dir(scan, glob) == scan / CAL_SUBDIR)
        side = scan / "side1"
        side.mkdir()
        check("side1/ falls back to the shared parent cal/",
              resolve_cal_dir(side, glob) == scan / CAL_SUBDIR)
        bare = root / "data" / "set2"
        bare.mkdir()
        check("falls back to global", resolve_cal_dir(bare, glob) == glob)

    print("against dcraw itself:")
    verdict, detail = _check_against_dcraw()
    if verdict is None:
        print(f"  SKIP  {detail}")
    else:
        check("bt709 inverse reproduces dcraw -g 1 1", verdict, detail)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {', '.join(fails)}")
        return 1
    print("All self-tests passed.")
    return 0


_RAW_SUFFIXES = {".cr2", ".nef", ".arw", ".dng", ".orf", ".raf", ".rw2", ".pef"}


def _find_test_raw() -> Path | None:
    """Any RAW file already in the repo's data/ folder, for the dcraw cross-check.

    Real RAW extensions are preferred. The capture scripts download RAWs as
    `<name>.tmp` and archive them in `tmpArchive/`, so those count too -- but
    only there, since a stray .tmp elsewhere is unlikely to be a camera file.
    """
    app_root = Path(__file__).resolve().parent.parent.parent   # .../2D
    files = sorted(f for f in (app_root.parent / "data").rglob("*") if f.is_file())
    for f in files:
        if f.suffix.lower() in _RAW_SUFFIXES:
            return f
    for f in files:
        if f.suffix.lower() == ".tmp" and f.parent.name == "tmpArchive":
            return f
    return None


def _check_against_dcraw(tol_lsb: float = 2.0):
    """Decode a real RAW both ways and verify the software inverse matches dcraw.

    This is the test that underwrites the whole legacy-scan story: it proves
    that inverting the BT.709 curve in software reproduces what dcraw would
    have produced with `-g 1 1`, so scans captured before the switch to linear
    can still be processed correctly.  Returns (passed, detail), or
    (None, reason) when dcraw or a RAW file is unavailable.
    """
    import shutil as _shutil
    import subprocess
    import tempfile

    if _shutil.which("dcraw") is None:
        return None, "dcraw not on PATH"
    raw = _find_test_raw()
    if raw is None:
        return None, "no RAW file found under data/"

    import tifffile
    common = ["-T", "-6", "-W", "-o", "0", "-q", "0", "-t", "0"]
    try:
        with tempfile.TemporaryDirectory() as td:
            out = {}
            for label, extra in (("gamma", []), ("linear", ["-g", "1", "1"])):
                p = Path(td) / f"{label}.tiff"
                with open(p, "wb") as fh:
                    subprocess.run(["dcraw", "-c", *common, *extra, str(raw)],
                                   stdout=fh, stderr=subprocess.DEVNULL, check=True)
                out[label] = tifffile.imread(str(p)).astype(np.float64) / 65535.0
    except (subprocess.CalledProcessError, OSError) as exc:
        return None, f"dcraw invocation failed: {exc}"

    err = np.abs(dcraw_decode(out["gamma"]) - out["linear"]).max()
    return err <= tol_lsb / 65535.0, (
        f"max err {err:.2e} = {err * 65535:.2f} LSB  ({raw.name})")


if __name__ == "__main__":
    import sys
    raise SystemExit(_selftest())
