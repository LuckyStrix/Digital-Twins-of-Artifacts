"""Stage 2 hook: scale a side's COLMAP reconstruction to millimeters.

``run_colmap_mvs.py`` calls ``scale_from_env`` right after ``colmap mapper``,
before the dense steps, so depth maps, ``fused.ply``, the ``.vis`` files and
``camera_centers.json`` all come out in millimeters.

Rules (from the design):
- every sparse model must fit the ring, or none is scaled;
- any failure leaves the side unscaled with a warning, never fails the run;
- ``<workspace>/scale.json`` records the result, including each model's raw
  ring radii in model units, so a later, better ring measurement can rescale
  old scans exactly.

Command line (for testing the ring on an existing COLMAP workspace):

    python -m scale.apply fit --workspace data/x_recon/colmap_side1 \\
        --raw data/x/side1 [--processed data/x_recon/processed/side1] \\
        [--inner-mm 200 --outer-mm 216] [--apply]
    python -m scale.apply unscale <workspace>

Scaling a session that already finished Stage 2 with the Scale tab off
(rescales the sparse models and the dense outputs; then re-run Stages 3-4):

    python -m scale.apply session --session data/x_recon --scan data/x [--fit-only]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .colmap_model import (
    read_model_txt, rescale_camera_centers, rescale_images_bin, rescale_model_txt,
    rescale_ply, rescale_points3d_bin,
)
from .ring import RingFit, RingSpec, fit_ring

SCALE_JSON = "scale.json"
UNSCALED_DIR = "sparse_unscaled"     # outside sparse/, so it's never read as a model


def env_spec(env=os.environ) -> RingSpec:
    def num(key, default):
        try:
            return float(env.get(key, "") or default)
        except ValueError:
            return default
    return RingSpec(inner_mm=num("FIPMESH_SCALE_RING_INNER_MM", 200.0),
                    outer_mm=num("FIPMESH_SCALE_RING_OUTER_MM", 216.0),
                    width_mm=num("FIPMESH_SCALE_RING_WIDTH_MM", 0.0))


def _convert(colmap_bin: str, src: Path, dst: Path, out_type: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run([colmap_bin, "model_converter", "--input_path", str(src),
                    "--output_path", str(dst), "--output_type", out_type],
                   check=True, capture_output=True)


def write_report(workspace: Path, report: dict) -> Path:
    path = Path(workspace) / SCALE_JSON
    path.write_text(json.dumps(report, indent=1))
    return path


def read_report(workspace: Path) -> dict | None:
    try:
        return json.loads((Path(workspace) / SCALE_JSON).read_text())
    except (OSError, ValueError):
        return None


def _ring_mm(fit: RingFit) -> dict:
    """The ring in the scaled (millimeter) frame."""
    s = fit.scale
    return {"center": [v * s for v in fit.center], "normal": list(fit.normal),
            "r_inner": fit.r_in * s, "r_outer": fit.r_out * s}


def scale_models(colmap_bin: str, model_dirs: list[Path], workspace: Path, raw_root: Path,
                 processed_root: Path | None, spec: RingSpec, apply: bool = True,
                 log=print) -> dict:
    """Fit the ring for every model; if all fit and ``apply``, rescale them in
    place (originals kept in ``<workspace>/sparse_unscaled/``). Returns the report."""
    workspace = Path(workspace)
    work = workspace / "scale"
    report: dict = {"status": "unscaled", "units": "model", "scale_mm_per_unit": None,
                    "sigma_pct": None, "ring_spec": spec.__dict__.copy(), "ring_mm": None,
                    "sparse_models": len(model_dirs), "raw_images": str(raw_root),
                    "models": [], "reason": "", "warnings": []}
    if len(model_dirs) > 1:
        report["warnings"].append(f"COLMAP split this side into {len(model_dirs)} separate models")
    fits: list[RingFit] = []
    for i, md in enumerate(model_dirs):
        txt = work / f"model_{i}_txt"
        if txt.exists():
            shutil.rmtree(txt)
        _convert(colmap_bin, md, txt, "TXT")
        model = read_model_txt(txt)
        log(f"[scale] model {i}: {len(model.images)} photos; looking for the ring")
        fit = fit_ring(model, raw_root, processed_root, spec,
                       debug_dir=work / "debug" / f"model_{i}", log=log)
        fits.append(fit)
        report["models"].append({"dir": str(md), "fit": fit.to_json()})
        for w in fit.warnings:
            report["warnings"].append(f"model {i}: {w}")
        if fit.ok:
            log(f"[scale] model {i}: 1 model unit = {fit.scale:.6g} mm "
                f"(+-{fit.sigma_pct:.3f}%, {fit.n_views} photos, band {fit.width_mm:.2f} mm)")
        else:
            log(f"[scale] model {i}: ring fit failed: {fit.reason}")

    bad = [i for i, f in enumerate(fits) if not f.ok]
    if bad:
        reasons = "; ".join(f"model {i}: {fits[i].reason}" for i in bad)
        report["reason"] = (reasons if len(fits) == 1 else
                            f"{len(bad)} of {len(fits)} models couldn't be scaled, so none were "
                            f"({reasons})")
        return report

    first = fits[0]
    report.update(scale_mm_per_unit=first.scale, sigma_pct=first.sigma_pct,
                  ring_mm=_ring_mm(first))
    if not apply:
        report["status"] = "fit_only"
        return report
    _rescale_sparse(colmap_bin, model_dirs, [f.scale for f in fits], workspace)
    report.update(status="scaled", units="mm")
    return report


def _rescale_sparse(colmap_bin: str, model_dirs: list[Path], scales: list[float],
                    workspace: Path) -> None:
    """Rescale each sparse model from the TXT copy ``scale_models`` made,
    keeping the originals in ``<workspace>/sparse_unscaled/``."""
    workspace = Path(workspace)
    backup = workspace / UNSCALED_DIR
    for i, (md, s) in enumerate(zip(model_dirs, scales)):
        dst = backup / md.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(md, dst)
        txt = workspace / "scale" / f"model_{i}_txt"
        rescale_model_txt(txt, s)
        _convert(colmap_bin, txt, md, "BIN")


def _rescale_dense(workspace: Path, k: float) -> None:
    """Multiply the finished outputs Stage 3/4 read by ``k``: the fused
    clouds, the dense sparse models and camera_centers.json. Depth maps are
    left alone (nothing after Stage 2 reads them)."""
    clouds = [workspace / "fused.ply"] + sorted((workspace / "dense").glob("fused_component_*.ply"))
    for p in clouds:
        if p.is_file() and p.stat().st_size > 0:
            rescale_ply(p, k)
    for sp in [workspace / "dense" / "sparse"] + sorted((workspace / "dense").glob("component_*/sparse")):
        if (sp / "images.bin").is_file():
            rescale_images_bin(sp / "images.bin", k)
        if (sp / "points3D.bin").is_file():
            rescale_points3d_bin(sp / "points3D.bin", k)
    cc = workspace / "camera_centers.json"
    if cc.is_file():
        rescale_camera_centers(cc, k)


def scale_from_env(colmap_bin: str, model_dirs: list[Path], workspace: Path, log=print) -> dict:
    """The Stage 2 hook. Never raises: any error leaves the side unscaled."""
    workspace = Path(workspace)
    if os.environ.get("FIPMESH_SCALE_RING", "0").strip() != "1":
        report = {"status": "disabled", "units": "model"}
        write_report(workspace, report)
        log("[scale] off (Scale tab); this side stays in model units")
        return report
    raw = os.environ.get("FIPMESH_SCALE_RAW_IMAGES", "").strip()
    proc = os.environ.get("FIPMESH_SCALE_PROCESSED_IMAGES", "").strip()
    spec = env_spec()
    try:
        if not raw or not Path(raw).is_dir():
            raise RuntimeError(f"raw photo folder not found: {raw or '(not set)'}")
        report = scale_models(colmap_bin, [Path(m) for m in model_dirs], workspace, Path(raw),
                              Path(proc) if proc else None, spec, log=log)
    except Exception as exc:
        report = {"status": "unscaled", "units": "model", "scale_mm_per_unit": None,
                  "ring_spec": spec.__dict__.copy(), "reason": f"scale step error: {exc}",
                  "warnings": []}
    write_report(workspace, report)
    if report["status"] == "scaled":
        log(f"[scale] side scaled to millimeters (1 model unit = "
            f"{report['scale_mm_per_unit']:.6g} mm)")
    else:
        log(f"[scale] WARNING: continuing unscaled: {report['reason']}")
    for w in report.get("warnings", []):
        log(f"[scale] WARNING: {w}")
    return report


def unscale_workspace(workspace: Path, log=print) -> bool:
    """Undo a side's scaling after Stage 2 finished (used when the other side
    couldn't be scaled, so both sides stay in matching units). Rescales the
    dense outputs Stage 3/4 read and restores the original sparse models."""
    workspace = Path(workspace)
    report = read_report(workspace)
    if not report or report.get("status") != "scaled":
        return False
    _rescale_dense(workspace, 1.0 / float(report["scale_mm_per_unit"]))
    backup = workspace / UNSCALED_DIR
    if backup.is_dir():
        for md in backup.iterdir():
            dst = workspace / "sparse" / md.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(md, dst)
    report.update(status="reverted", units="model",
                  reason="the other side couldn't be scaled; reverted so both sides match")
    write_report(workspace, report)
    log(f"[scale] reverted {workspace.name} to model units")
    return True


def _model_dirs(workspace: Path) -> list[Path]:
    root = Path(workspace) / "sparse"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith("_")
                  and ((p / "images.bin").exists() or (p / "images.txt").exists()))


def session_sides(session: Path, scan: Path) -> list[tuple[str, Path, Path, Path]]:
    """(label, workspace, raw photos, Stage 1 photos) for each COLMAP side of
    a session, laid out the way the app writes it."""
    session, scan = Path(session), Path(scan)
    out = []
    for ws in sorted(session.glob("colmap_*")):
        if not ws.is_dir():
            continue
        if ws.name == "colmap_out":                       # flat scan, no sides
            out.append(("scan", ws, scan, session / "processed"))
        else:
            side = ws.name[len("colmap_"):]
            out.append((side, ws, scan / side, session / "processed" / side))
    return out


def scale_finished_session(colmap_bin: str, session: Path, scan: Path, spec: RingSpec,
                           apply: bool = True, log=print) -> dict:
    """Scale a session whose Stage 2 already ran with the Scale tab off, so
    COLMAP doesn't have to run again. Every side must fit the ring or nothing
    changes. Afterwards re-run Stages 3 and 4 with the Scale tab on."""
    sides = session_sides(session, scan)
    result: dict = {"status": "unscaled", "sides": {}, "reason": ""}
    if not sides:
        result["reason"] = f"no colmap_* folders in {session}"
        return result
    problems, fitted = [], []
    for label, ws, raw, proc in sides:
        prior = read_report(ws) or {}
        models = _model_dirs(ws)
        if prior.get("status") == "scaled":
            problems.append(f"{label}: already scaled")
        elif not (ws / "fused.ply").is_file():
            problems.append(f"{label}: no fused.ply (Stage 2 didn't finish)")
        elif not raw.is_dir():
            problems.append(f"{label}: raw photo folder not found: {raw}")
        elif len(models) != 1:
            problems.append(f"{label}: {len(models)} sparse models; scaling a finished side "
                            f"needs exactly 1 (re-run Stage 2 with the Scale tab on instead)")
        else:
            fitted.append((label, ws, raw, proc, models))
    if problems:
        result["reason"] = "; ".join(problems)
        return result

    reports = {}
    for label, ws, raw, proc, models in fitted:
        log(f"[scale] {label}: fitting the ring")
        rep = scale_models(colmap_bin, models, ws, raw, proc if proc.is_dir() else None,
                           spec, apply=False, log=log)
        reports[label] = rep
        result["sides"][label] = {k: rep.get(k) for k in
                                  ("status", "scale_mm_per_unit", "sigma_pct", "reason", "warnings")}
        for w in rep.get("warnings", []):
            log(f"[scale] {label}: WARNING: {w}")
    bad = [f"{l}: {r['reason']}" for l, r in reports.items() if r["status"] != "fit_only"]
    if bad:
        result["reason"] = "; ".join(bad)
        return result
    if not apply:
        result["status"] = "fit_only"
        return result

    for label, ws, raw, proc, models in fitted:
        rep = reports[label]
        s = float(rep["scale_mm_per_unit"])
        _rescale_sparse(colmap_bin, models, [s], ws)
        _rescale_dense(ws, s)
        rep.update(status="scaled", units="mm", scaled_after_stage2=True)
        write_report(ws, rep)
        result["sides"][label]["status"] = "scaled"
        log(f"[scale] {label}: scaled to millimeters (1 model unit = {s:.6g} mm)")
    result["status"] = "scaled"
    return result


def _resolve_colmap() -> str:
    env = os.environ.get("FIPMESH_COLMAP_BIN", "")
    if env:
        return env
    here = Path(__file__).resolve().parent.parent
    local = here / "colmap_local"
    if local.is_file():
        return str(local)
    bat = here / "colmap" / "COLMAP.bat"
    if bat.is_file():
        return str(bat)
    return shutil.which("colmap") or "colmap"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scale.apply", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the ring on an existing COLMAP workspace")
    f.add_argument("--workspace", required=True, help="a side's COLMAP folder (colmap_side1)")
    f.add_argument("--raw", required=True, help="that side's raw photos (scan/side1)")
    f.add_argument("--processed", default="", help="that side's Stage 1 photos (for the tablet mask)")
    f.add_argument("--inner-mm", type=float, default=200.0)
    f.add_argument("--outer-mm", type=float, default=216.0)
    f.add_argument("--width-mm", type=float, default=0.0)
    f.add_argument("--apply", action="store_true",
                   help="also rescale the sparse models (only useful before the dense steps)")
    f.add_argument("--colmap-bin", default="")
    u = sub.add_parser("unscale", help="undo a side's scaling")
    u.add_argument("workspace")
    se = sub.add_parser("session", help="scale a session whose Stage 2 ran with the Scale tab off")
    se.add_argument("--session", required=True, help="the output folder (scan_recon)")
    se.add_argument("--scan", required=True, help="the input folder with the raw photos")
    se.add_argument("--inner-mm", type=float, default=200.0)
    se.add_argument("--outer-mm", type=float, default=216.0)
    se.add_argument("--width-mm", type=float, default=0.0)
    se.add_argument("--fit-only", action="store_true", help="fit and report, change nothing")
    se.add_argument("--colmap-bin", default="")
    a = ap.parse_args(argv)
    if a.cmd == "unscale":
        return 0 if unscale_workspace(Path(a.workspace)) else 1
    if a.cmd == "session":
        res = scale_finished_session(a.colmap_bin or _resolve_colmap(), Path(a.session),
                                     Path(a.scan), RingSpec(a.inner_mm, a.outer_mm, a.width_mm),
                                     apply=not a.fit_only)
        print(json.dumps(res, indent=1))
        if res["status"] == "scaled":
            print("done: turn on the Scale tab, then Run Stage on Stage 3 and Stage 4")
        elif res["status"] != "fit_only":
            print(f"nothing changed: {res['reason']}", file=sys.stderr)
        return 0 if res["status"] in ("scaled", "fit_only") else 2
    ws = Path(a.workspace)
    models = _model_dirs(ws)
    if not models:
        print(f"no sparse models under {ws / 'sparse'}", file=sys.stderr)
        return 1
    report = scale_models(a.colmap_bin or _resolve_colmap(), models, ws, Path(a.raw),
                          Path(a.processed) if a.processed else None,
                          RingSpec(a.inner_mm, a.outer_mm, a.width_mm), apply=a.apply)
    if a.apply:
        write_report(ws, report)
    out = ws / "scale" / "fit_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("status", "scale_mm_per_unit", "sigma_pct",
                                             "reason", "warnings")}, indent=1))
    print(f"full report: {out}\ndebug images: {ws / 'scale' / 'debug'}")
    return 0 if report["status"] in ("scaled", "fit_only") else 2


if __name__ == "__main__":
    sys.exit(main())
