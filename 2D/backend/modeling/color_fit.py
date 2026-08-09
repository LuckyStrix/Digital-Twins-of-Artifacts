#!/usr/bin/env python3
"""
color_fit.py
============
Fits the ColorChecker correction that Stage 1 applies, and writes ccm.json.

Split out from color_correction.py deliberately.  This half needs
opencv-contrib (for cv2.mcc chart detection) and colour-science (reference
colorimetry, CIE Lab, dE2000); the pipeline half needs neither and only ever
reads a JSON and multiplies a 3x3.  Keeping them apart means a machine that
merely *processes* scans never has to install contrib OpenCV - which matters,
because rembg depends on plain opencv-python-headless and pip will happily
install both into the same cv2/ directory with last-writer-wins.

Run it once per calibration session:

    python color_fit.py fit    --cal-dir <working>/cal
    python color_fit.py report --ccm    <working>/cal/ccm.json

THE INVARIANT
-------------
The chart is flat-fielded exactly the way the artifact will be: divided by the
envelope built from the *uncorrected* copy paper.  Both sides then live in the
same units, so the gains/exposure/matrix transfer to Stage 1 verbatim.  See the
header of color_correction.py.  If you change how the envelope is built here,
change it there too.

WHY DETECTION USES cv2.mcc FOR GEOMETRY ONLY
--------------------------------------------
cv2.mcc's getChartsRGB() computes patch means on the 8-bit, display-encoded
image handed to process().  Those numbers are useless for fitting a matrix.
This module uses mcc purely to locate the chart, then samples the full
precision linear array itself.  --box bypasses detection entirely when mcc
cannot find the chart, which happens more often than you would like under flat
studio lighting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import color_correction as cc

# The chart's neutral ramp, patches 19-24 (white 9.5 .. black 2), 0-indexed.
NEUTRAL_IDX = np.arange(18, 24)

# Reference dataset. A 2005 GretagMacbeth chart predates the November 2014
# reformulation, so it needs the "Before" colorimetry - X-Rite changed the
# colorants and published separate data for the two eras.
DEFAULT_CHART = "ColorChecker24 - Before November 2014"

# Patch 24 ("black 2") sits low enough that veiling glare dominates it, which
# drags the whole neutral fit. Excluded from the white-balance solve by default.
DROP_FROM_WB = {23}


class ChartNotFound(RuntimeError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Reference colorimetry
# ─────────────────────────────────────────────────────────────────────────────
def reference_patches(chart: str = DEFAULT_CHART):
    """(linear sRGB (24,3), XYZ D65 (24,3), names (24,)) for the chart.

    colour-science stores the checkers as xyY under the ICC D50 illuminant, so
    this converts and chromatically adapts to D65, which is sRGB's white point.
    """
    import colour

    data = colour.CCS_COLOURCHECKERS[chart]
    names = list(data.data.keys())
    xyY = np.array([data.data[n] for n in names], dtype=np.float64)
    XYZ = colour.xyY_to_XYZ(xyY)
    rgb = colour.XYZ_to_sRGB(XYZ, illuminant=data.illuminant,
                             apply_cctf_encoding=False)      # keep it linear
    XYZ_d65 = colour.sRGB_to_XYZ(rgb, apply_cctf_decoding=False)
    return np.asarray(rgb, np.float64), np.asarray(XYZ_d65, np.float64), names


def load_reference_csv(path: Path):
    """Reference values measured from YOUR chart, overriding the published set.

    A 21-year-old chart has drifted; if you ever get spectrophotometer access
    this is the cheapest way to cut residual dE. Expects 24 rows of
    `name,L,a,b` or `name,R,G,B` (linear sRGB), header optional.
    """
    import colour

    rows = [r.strip().split(",") for r in Path(path).read_text().splitlines()
            if r.strip() and not r.lstrip().startswith("#")]
    if rows and not _is_number(rows[0][1]):
        rows = rows[1:]                                        # drop a header
    if len(rows) != 24:
        raise ValueError(f"{path}: expected 24 patches, found {len(rows)}")
    names = [r[0] for r in rows]
    vals = np.array([[float(v) for v in r[1:4]] for r in rows])
    if vals[:, 0].max() > 3.0:                                 # looks like L*a*b*
        XYZ = colour.Lab_to_XYZ(vals)
        rgb = colour.XYZ_to_sRGB(XYZ, apply_cctf_encoding=False)
    else:
        rgb = vals
        XYZ = colour.sRGB_to_XYZ(rgb, apply_cctf_decoding=False)
    return np.asarray(rgb, np.float64), np.asarray(XYZ, np.float64), names


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Chart detection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ChartDetection:
    box: np.ndarray          # (4,2) chart corners in image pixels
    method: str
    flipped: bool = False    # True when the chart was found upside-down


def _order_box(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)                       # counter-clockwise from -pi
    pts = pts[order]
    start = np.argmin(pts.sum(axis=1))            # top-left has the smallest x+y
    return np.roll(pts, -start, axis=0)


def detect_chart(display_bgr_u8: np.ndarray) -> ChartDetection:
    """Locate the chart with cv2.mcc. Raises ChartNotFound."""
    import cv2

    if not hasattr(cv2, "mcc"):
        raise ChartNotFound(
            "This OpenCV build has no 'mcc' module, so the chart cannot be "
            "detected automatically. Install opencv-contrib-python-headless, "
            "or pass --box with the chart's four corners.")
    detector = cv2.mcc.CCheckerDetector_create()
    if not detector.process(display_bgr_u8, cv2.mcc.MCC24):
        raise ChartNotFound(
            "cv2.mcc could not find a 24-patch chart. Its detector is "
            "unreliable under flat studio lighting. Re-run with --box "
            "\"x0,y0,x1,y1,x2,y2,x3,y3\" giving the chart's outer corners "
            "(clockwise from top-left), which you can read off the overlay PNG.")
    checker = detector.getBestColorChecker()
    return ChartDetection(_order_box(np.array(checker.getBox())), "cv2.mcc.MCC24")


def sample_patches(linear_rgb: np.ndarray, det: ChartDetection,
                   frac: float = 0.5, clip_ceiling: float = 0.98,
                   clip_ref: np.ndarray | None = None):
    """Median linear RGB of each of the 24 patches, in reading order.

    Samples the centre `frac` of every patch cell from the FULL-PRECISION
    LINEAR image - never from the 8-bit preview the detector ran on. Median
    rather than mean so dust and the odd specular fleck do not shift a patch.

    `clip_ref` is the RAW chart (pre-flat-field), which is where saturation
    actually means something: 1.0 there is the sensor ceiling. Testing the
    flat-fielded array instead would flag the white patch on every well-exposed
    chart, since dividing by the paper envelope legitimately pushes it past 1.

    Returns (means (24,3), stds (24,3), clipped (24,) bool).
    """
    import cv2

    h_cells, w_cells = 4, 6
    dst = det.box.astype(np.float32)
    src = np.array([[0, 0], [w_cells, 0], [w_cells, h_cells], [0, h_cells]],
                   np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    means = np.zeros((24, 3))
    stds = np.zeros((24, 3))
    clipped = np.zeros(24, bool)
    m = (1.0 - frac) / 2.0
    for r in range(h_cells):
        for c in range(w_cells):
            # Corners of the centre sub-rectangle of this cell, in chart space.
            quad = np.array([[c + m, r + m], [c + 1 - m, r + m],
                             [c + 1 - m, r + 1 - m], [c + m, r + 1 - m]],
                            np.float32).reshape(-1, 1, 2)
            px = cv2.perspectiveTransform(quad, H).reshape(4, 2)
            x0, y0 = np.floor(px.min(axis=0)).astype(int)
            x1, y1 = np.ceil(px.max(axis=0)).astype(int)
            x0, y0 = max(x0, 0), max(y0, 0)
            x1 = min(x1, linear_rgb.shape[1])
            y1 = min(y1, linear_rgb.shape[0])
            patch = linear_rgb[y0:y1, x0:x1].reshape(-1, 3)
            i = r * w_cells + c
            means[i] = np.median(patch, axis=0)
            stds[i] = patch.std(axis=0)
            ref = (clip_ref[y0:y1, x0:x1].reshape(-1, 3)
                   if clip_ref is not None else patch)
            clipped[i] = bool((ref >= clip_ceiling).mean() > 0.01)
    return means, stds, clipped


def orient_patches(measured: np.ndarray, det: ChartDetection) -> np.ndarray:
    """Flip the patch order if the chart was detected upside-down.

    The neutral ramp runs light -> dark left to right along the bottom row. If
    it reads dark -> light the chart is rotated 180 degrees, and reversing the
    whole array puts every patch back where the reference expects it.
    """
    neutrals = measured[NEUTRAL_IDX].mean(axis=1)
    if neutrals[0] < neutrals[-1]:
        det.flipped = True
        return measured[::-1].copy()
    return measured


# ─────────────────────────────────────────────────────────────────────────────
# The fit
# ─────────────────────────────────────────────────────────────────────────────
def fit_black_offset(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-channel additive offset, from an affine fit down the neutral ramp.

    Fits measured = slope * reference + offset per channel and returns the
    offsets. Any additive term is negligible on bright patches and dominant on
    dark ones, so it surfaces as greys that drift coloured as they get darker -
    which is exactly what this data shows (chroma rising from 0.3 at neutral 5
    to 6.1 at black 2).

    Measured here as a NEGATIVE offset, so it is not veiling glare (which adds
    light). The likely cause is cross-polarisation removing a surface component
    that the reference's 45/0 measurement geometry retains, plus drift in a
    21-year-old chart. Whatever the cause, repeated split-half cross-validation
    over the patches showed removing it improves held-out dE in 100% of trials,
    so it is real signal and not three free parameters fitting noise.

    Weighted by 1/reference, i.e. minimising RELATIVE rather than absolute
    error. Plain least squares is dominated by the white end, where the offset
    is irrelevant, and the resulting estimate overshoots the dark patches badly
    (black 2 reached dE 8.0 against 4.8 with no offset at all). Relative
    weighting balances the ramp and is better on every metric: mean 2.34 vs
    2.61, p95 4.14 vs 5.51, worst patch 5.7 vs 8.0.
    """
    idx = np.array([i for i in NEUTRAL_IDX if i not in DROP_FROM_WB])
    out = np.zeros(3)
    for c in range(3):
        A = np.stack([reference[idx, c], np.ones(len(idx))], axis=1)
        w = 1.0 / np.maximum(reference[idx, c], 1e-3)
        out[c] = np.linalg.lstsq(A * w[:, None], measured[idx, c] * w,
                                 rcond=None)[0][1]
    return out


def fit_white_balance(measured: np.ndarray, reference: np.ndarray):
    """Per-channel gains plus one exposure scalar, from the neutral ramp.

    The gains make the neutrals neutral; the scalar puts them at the right
    absolute level. Normalised so gains[G] == 1, leaving green as the reference
    channel. Returns (gains (3,), exposure).
    """
    idx = np.array([i for i in NEUTRAL_IDX if i not in DROP_FROM_WB])
    m, r = measured[idx], reference[idx]
    # Least-squares scale per channel through the origin: the ramp is a line
    # from black to white, so its slope IS the channel's response.
    per_channel = (m * r).sum(axis=0) / np.maximum((m * m).sum(axis=0), 1e-12)
    gains = per_channel / per_channel[1]           # green == 1
    exposure = float(per_channel[1])
    return gains, exposure


def fit_ccm(measured_wb: np.ndarray, reference: np.ndarray,
            white_preserving: bool = True, exclude: np.ndarray | None = None):
    """3x3 matrix mapping white-balanced camera RGB to linear sRGB.

    white_preserving constrains every row to sum to 1, so M @ [1,1,1] == [1,1,1]
    exactly. Without it the least-squares solution is free to introduce a global
    cast that undoes the white balance, which shows up as tinted greys - the
    most visible failure mode there is.
    """
    keep = np.ones(len(measured_wb), bool)
    if exclude is not None:
        keep &= ~exclude
    A, B = measured_wb[keep], reference[keep]

    if not white_preserving:
        M, *_ = np.linalg.lstsq(A, B, rcond=None)
        return M.T

    # Row i: out_i = m0*R + m1*G + (1-m0-m1)*B  =>  out_i - B = m0*(R-B) + m1*(G-B)
    X = np.stack([A[:, 0] - A[:, 2], A[:, 1] - A[:, 2]], axis=1)
    M = np.zeros((3, 3))
    for i in range(3):
        y = B[:, i] - A[:, 2]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        M[i] = [coef[0], coef[1], 1.0 - coef[0] - coef[1]]
    return M


def delta_e_report(corrected: np.ndarray, reference: np.ndarray, names):
    """dE2000 per patch between two sets of linear sRGB values."""
    import colour

    def to_lab(rgb):
        XYZ = colour.sRGB_to_XYZ(np.clip(rgb, 0, None), apply_cctf_decoding=False)
        return colour.XYZ_to_Lab(XYZ)

    de = colour.difference.delta_E_CIE2000(to_lab(corrected), to_lab(reference))
    de = np.asarray(de, np.float64)
    worst = int(np.argmax(de))
    return {
        "per_patch": [float(v) for v in de],
        "patch_names": list(names),
        "mean": float(de.mean()),
        "median": float(np.median(de)),
        "max": float(de.max()),
        "p95": float(np.percentile(de, 95)),
        "neutrals_mean": float(de[NEUTRAL_IDX].mean()),
        "worst_patch": names[worst],
        "worst_value": float(de[worst]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flat-fielding the chart (must mirror Stage 1)
# ─────────────────────────────────────────────────────────────────────────────
def flat_field_chart(cal_dir: Path, frame: str = "allLight.tiff",
                     smooth_sigma: float = 200.0):
    """Chart / envelope(copy paper), both linear. Returns (rgb, diagnostics).

    Mirrors run_calibration() in modeling_pipeline.py: the envelope is the
    luminance of the UNCORRECTED linear paper, blurred. This is the invariant
    that lets the fitted matrix transfer to Stage 1 unchanged.
    """
    import tifffile
    from scipy.ndimage import gaussian_filter

    chart_dir = cal_dir / cc.CHART_SUBDIR
    chart_path = chart_dir / frame
    paper_path = cal_dir / frame
    if not chart_path.is_file():
        raise FileNotFoundError(
            f"No chart frame at {chart_path}. Capture the colour chart "
            "(Step 0b) before fitting.")

    chart_enc = cc.read_capture_info(chart_dir)
    chart = cc.to_linear(
        tifffile.imread(str(chart_path)).astype(np.float32) / 65535.0, chart_enc)

    diag = {"chart_encoding": chart_enc.kind, "flat_fielded": False,
            "chart_frame": frame}

    if not paper_path.is_file():
        print(f"  WARNING: no copy-paper frame at {paper_path}. Fitting on the "
              "raw chart instead.")
        print("           The matrix will still be usable, but it will not "
              "match Stage 1's flat-fielded units exactly, and any vignetting "
              "across the chart is left in.")
        return chart, diag

    paper_enc = cc.read_capture_info(cal_dir)
    paper = cc.to_linear(
        tifffile.imread(str(paper_path)).astype(np.float32) / 65535.0, paper_enc)
    if paper.shape != chart.shape:
        raise ValueError(
            f"Paper {paper.shape} and chart {chart.shape} differ in size. They "
            "must be shot at the same zoom and orientation.")

    # tifffile gives RGB; the BT.709 weights below are in that order.
    lum = (0.2126 * paper[..., 0] + 0.7152 * paper[..., 1]
           + 0.0722 * paper[..., 2])
    env = gaussian_filter(lum.astype(np.float64), sigma=smooth_sigma)
    env = np.maximum(env, 0.02 * float(env.max()))

    diag.update({"flat_fielded": True, "paper_encoding": paper_enc.kind,
                 "smooth_sigma": smooth_sigma,
                 "envelope_range": [float(env.min()), float(env.max())]})
    diag["_raw"] = chart
    return (chart / env[..., None]).astype(np.float32), diag


def fit_exposure(measured_ff_wb: np.ndarray, reference: np.ndarray) -> float:
    """The single exposure scalar, measured in Stage 1's flat-fielded units.

    Split out from the gains/matrix on purpose. A flat-field is a per-pixel
    SCALAR, and a scalar commutes with the 3x3, so the matrix and the channel
    gains are unaffected by whether the chart was flat-fielded - which means
    they are better fitted on the raw chart, where they do not inherit the
    envelope's spatial residual (measured at ~2.6% between shots here).

    Only the absolute level depends on the flat-field, so only this one number
    is measured in flat-fielded units. That keeps the invariant exact while
    keeping the envelope's noise out of the colour fit.
    """
    idx = np.array([i for i in NEUTRAL_IDX if i not in DROP_FROM_WB])
    m, r = measured_ff_wb[idx, 1], reference[idx, 1]       # green channel
    return float((m * r).sum() / max((m * m).sum(), 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def write_overlay(linear_rgb, det, out_path: Path, frac: float = 0.5):
    """Draw the detected chart and the sampled regions on a viewable preview.

    The single most useful artifact this produces: one glance confirms the
    patches were sampled where you think they were.
    """
    import cv2

    disp = (np.clip(cc.srgb_encode(linear_rgb), 0, 1) * 255).astype(np.uint8)
    disp = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
    cv2.polylines(disp, [det.box.astype(np.int32)], True, (0, 255, 0), 6)

    dst = det.box.astype(np.float32)
    src = np.array([[0, 0], [6, 0], [6, 4], [0, 4]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    m = (1.0 - frac) / 2.0
    for r in range(4):
        for c in range(6):
            quad = np.array([[c + m, r + m], [c + 1 - m, r + m],
                             [c + 1 - m, r + 1 - m], [c + m, r + 1 - m]],
                            np.float32).reshape(-1, 1, 2)
            px = cv2.perspectiveTransform(quad, H).reshape(4, 2).astype(np.int32)
            cv2.polylines(disp, [px], True, (0, 0, 255), 3)
            cv2.putText(disp, str(r * 6 + c + 1), tuple(px[0] + [8, 34]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    scale = 1600 / disp.shape[1]
    small = cv2.resize(disp, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), small)
    return out_path


def print_report(rep: dict, measured_wb=None, corrected=None, reference=None):
    print()
    print(f"  {'patch':<22}{'dE2000':>8}   {'measured (lin sRGB)':>24}"
          f"{'reference':>24}")
    for i, n in enumerate(rep["patch_names"]):
        row = f"  {n:<22}{rep['per_patch'][i]:>8.2f}"
        if corrected is not None and reference is not None:
            row += ("   " + " ".join(f"{v:6.3f}" for v in corrected[i])
                    + "   " + " ".join(f"{v:6.3f}" for v in reference[i]))
        print(row)
    print()
    print(f"  mean dE2000 {rep['mean']:.2f}   median {rep['median']:.2f}   "
          f"p95 {rep['p95']:.2f}   max {rep['max']:.2f} ({rep['worst_patch']})")
    print(f"  neutrals    {rep['neutrals_mean']:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def cmd_fit(args) -> int:
    import tifffile

    cal_dir = Path(args.cal_dir).resolve()
    print(f"Calibration set : {cal_dir}")

    chart_lin, diag = flat_field_chart(cal_dir, args.frame, args.smooth_sigma)
    print(f"Chart frame     : {diag['chart_frame']}  "
          f"(encoding: {diag['chart_encoding']})")
    print(f"Flat-fielded    : {diag['flat_fielded']}"
          + (f"  (paper envelope, sigma {diag['smooth_sigma']:g})"
             if diag["flat_fielded"] else "  <- NOT matching Stage 1 units"))

    # Detection runs on a display-encoded 8-bit copy; measurement does not.
    import cv2
    disp = (np.clip(cc.srgb_encode(chart_lin / max(np.percentile(chart_lin, 99), 1e-6)),
                    0, 1) * 255).astype(np.uint8)
    disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)

    if args.box:
        vals = [float(v) for v in args.box.replace(" ", "").split(",")]
        if len(vals) != 8:
            print("ERROR: --box needs 8 numbers: x0,y0,x1,y1,x2,y2,x3,y3")
            return 2
        det = ChartDetection(_order_box(np.array(vals).reshape(4, 2)), "manual-box")
        print("Detection       : manual --box")
    else:
        try:
            det = detect_chart(disp_bgr)
            print(f"Detection       : {det.method}")
        except ChartNotFound as exc:
            print(f"\nERROR: {exc}")
            fallback = cal_dir / "chart_detect_FAILED.png"
            import cv2 as _cv2
            _cv2.imwrite(str(fallback), _cv2.resize(
                disp_bgr, None, fx=1600 / disp_bgr.shape[1],
                fy=1600 / disp_bgr.shape[1], interpolation=_cv2.INTER_AREA))
            print(f"Wrote {fallback} — read the corner coordinates off it for --box "
                  "(they are in this image's scale; multiply by "
                  f"{disp_bgr.shape[1] / 1600:.3f}).")
            return 1

    raw_chart = diag.pop("_raw", None)
    # Colour (gains + matrix) is fitted on the RAW chart; only the exposure
    # scalar is measured in flat-fielded units. See fit_exposure.
    measured_ff, stds, clipped = sample_patches(
        chart_lin, det, args.patch_fraction, clip_ref=raw_chart)
    measured = (measured_ff if raw_chart is None else
                sample_patches(raw_chart, det, args.patch_fraction,
                               clip_ref=raw_chart)[0])
    measured = orient_patches(measured, det)
    measured_ff = measured_ff[::-1].copy() if det.flipped else measured_ff
    if det.flipped:
        clipped = clipped[::-1].copy()
        print("                  chart was upside-down; patch order reversed")

    if args.reference_csv:
        ref_rgb, ref_XYZ, names = load_reference_csv(Path(args.reference_csv))
        ref_src = str(args.reference_csv)
    else:
        ref_rgb, ref_XYZ, names = reference_patches(args.chart)
        ref_src = args.chart
    print(f"Reference        : {ref_src}")

    if clipped.any():
        bad = [names[i] for i in np.flatnonzero(clipped)]
        print(f"  WARNING: clipped patches excluded from the fit: {', '.join(bad)}")
        print("           A clipped patch biases the whole matrix. Re-shoot "
              "with less exposure if this includes a neutral.")

    # Offset is additive, so unlike the gains and matrix it does NOT survive the
    # flat-field's per-pixel scaling and has to be converted between domains.
    #
    # Fit it on the RAW chart and convert, rather than fitting it on the
    # flat-fielded chart directly. Fitting in the flat-fielded domain looks
    # equivalent but is not: the envelope varies ~1.2x across the chart, so each
    # patch is scaled differently and the affine fit -- which least squares
    # weights toward the bright end -- comes out roughly 2.5x too large. That
    # overshoots the darkest patch badly (black 2 went to dE 8.1). The raw
    # domain has no such per-patch scaling, and dividing by the mean envelope
    # over the patches is the correct conversion.
    if args.no_black_offset:
        off_raw = off_ff = np.zeros(3)
    else:
        off_raw = fit_black_offset(measured, ref_rgb)
        env_at_chart = float(np.median(measured / np.maximum(measured_ff, 1e-9)))
        off_ff = off_raw / env_at_chart

    gains, exposure_raw = fit_white_balance(measured - off_raw, ref_rgb)
    M = fit_ccm((measured - off_raw) * gains * exposure_raw, ref_rgb,
                white_preserving=not args.no_white_preserving, exclude=clipped)

    # Re-measure the level in the units Stage 1 will actually hand us.
    exposure = (fit_exposure((measured_ff - off_ff) * gains, ref_rgb)
                if diag["flat_fielded"] else exposure_raw)
    wb = (measured_ff - off_ff) * gains * exposure
    corrected = wb @ M.T

    before = delta_e_report(measured_ff * exposure, ref_rgb, names)
    after = delta_e_report(corrected, ref_rgb, names)

    print(f"\nBlack offset     : R/G/B = "
          + "/".join(f"{o:+.5f}" for o in off_ff)
          + ("   (disabled)" if args.no_black_offset else
             f"   ({off_ff.max() - off_ff.min():.5f} spread => "
             + ("coloured" if off_ff.max() - off_ff.min() > 2e-3 else "neutral") + ")"))
    print(f"White balance    : gains R/G/B = "
          + "/".join(f"{g:.4f}" for g in gains)
          + f"   exposure = {exposure:.4f}")
    print("Matrix           :")
    for row in M:
        print("                   [" + "  ".join(f"{v:+.4f}" for v in row) + "]")
    print(f"                   row sums = "
          + ", ".join(f"{s:.6f}" for s in M.sum(axis=1))
          + "  (1.0 => neutrals stay neutral)")

    if args.verbose:
        print_report(after, wb, corrected, ref_rgb)
    else:
        print_report(after)
    print(f"\n  before correction: mean dE2000 {before['mean']:.2f}"
          f"   ->  after: {after['mean']:.2f}")

    overlay = write_overlay(chart_lin, det, cal_dir / "chart_detect_overlay.png",
                            args.patch_fraction)
    print(f"\nWrote {overlay}  <- CHECK THIS: confirm the red squares sit inside "
          "the patches")

    enc = cc.read_capture_info(cal_dir / cc.CHART_SUBDIR)
    correction = cc.ColorCorrection(gains=gains, exposure=exposure, matrix=M,
                                    encoding=enc, offset=off_ff)
    out = cal_dir / cc.CCM_NAME
    correction.save(out, extras={
        "source": {"cal_dir": str(cal_dir), **diag},
        "reference": {"dataset": ref_src, "target_space": "sRGB",
                      "target_encoding": "linear", "target_illuminant": "D65"},
        "detection": {"method": det.method,
                      "box": det.box.tolist(),
                      "patch_sample_fraction": args.patch_fraction,
                      "orientation_flipped": bool(det.flipped)},
        "ccm": {"matrix_rgb": M.tolist(),
                "white_preserving": not args.no_white_preserving,
                "row_sums": [float(s) for s in M.sum(axis=1)]},
        "measured_linear_rgb": measured_ff.tolist(),
        "measured_raw_linear_rgb": measured.tolist(),
        "reference_linear_srgb": ref_rgb.tolist(),
        "delta_e_2000": {**after, "before_correction_mean": before["mean"]},
        "diagnostics": {"overlay_png": overlay.name,
                        "patches_clipped": [names[i] for i in np.flatnonzero(clipped)],
                        "patch_std_max": float(stds.max())},
    })
    print(f"Wrote {out}")

    if after["mean"] > args.max_delta_e:
        print(f"\nFAILED: mean dE2000 {after['mean']:.2f} exceeds "
              f"--max-delta-e {args.max_delta_e:.2f}. Something is structurally "
              "wrong - check the overlay for misregistration, look for clipped "
              "or glared patches, and confirm the chart era matches --chart.")
        return 1
    return 0


def cmd_report(args) -> int:
    info = json.loads(Path(args.ccm).read_text())
    de = info.get("delta_e_2000", {})
    print(f"{args.ccm}")
    print(f"  created   : {info.get('created_utc')}")
    print(f"  reference : {info.get('reference', {}).get('dataset')}")
    print(f"  detection : {info.get('detection', {}).get('method')}")
    wb = info.get("white_balance", {})
    print(f"  gains     : " + "/".join(f"{g:.4f}" for g in wb.get("gains_rgb", []))
          + f"   exposure {wb.get('exposure_scale', float('nan')):.4f}")
    if de:
        print_report(de)
        print(f"\n  before correction: {de.get('before_correction_mean', float('nan')):.2f}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit a colour matrix from a chart capture")
    f.add_argument("--cal-dir", required=True,
                   help="calibration folder holding the paper set and chart/")
    f.add_argument("--frame", default="allLight.tiff",
                   help="which frame to fit from (default: allLight.tiff, which "
                        "is shot cross-polarised)")
    f.add_argument("--box", help="manual chart corners x0,y0,...,x3,y3, "
                                 "bypassing automatic detection")
    f.add_argument("--chart", default=DEFAULT_CHART,
                   help="reference dataset (default suits pre-Nov-2014 charts)")
    f.add_argument("--reference-csv", help="measured reference values for YOUR chart")
    f.add_argument("--patch-fraction", type=float, default=0.5,
                   help="fraction of each cell to sample (default 0.5)")
    f.add_argument("--smooth-sigma", type=float, default=200.0,
                   help="flat-field blur; must match SMOOTH_SIGMA in the pipeline")
    f.add_argument("--max-delta-e", type=float, default=5.0,
                   help="exit non-zero above this mean dE2000 (default 5.0)")
    f.add_argument("--no-black-offset", action="store_true",
                   help="skip the neutral-ramp black offset (see fit_black_offset)")
    f.add_argument("--no-white-preserving", action="store_true",
                   help="allow rows that do not sum to 1 (not recommended)")
    f.add_argument("-v", "--verbose", action="store_true",
                   help="print measured and reference values per patch")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("report", help="print the dE report from an existing ccm.json")
    r.add_argument("--ccm", required=True)
    r.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
