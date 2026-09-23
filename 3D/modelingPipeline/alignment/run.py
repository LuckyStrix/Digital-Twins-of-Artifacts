"""
run.py -- headless CLI (no GUI). Aligns B onto A and writes a merged PLY.

  ./venv/bin/python run.py A.ply B.ply -o merged.ply [--method opening|fpfh|both]
                                                     [--voxel 0.5] [--view]
"""
import argparse
import json
import os
import align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A", help="fixed / target PLY")
    ap.add_argument("B", help="moving / source PLY")
    ap.add_argument("-o", "--out", default="merged.ply")
    ap.add_argument("--method",
                    choices=["opening", "fpfh", "collapse", "all"], default="all")
    ap.add_argument("--voxel", type=float, default=None)
    ap.add_argument("--samples", type=int, default=60000)
    g = ap.add_argument_group("ICP refinement (coarse-to-fine, off by default)")
    g.add_argument("--refine", action="store_true",
                   help="polish the chosen pose with a multi-stage ICP on dense clouds")
    g.add_argument("--refine-thresholds", default="3,1,0.5,0.25",
                   help="comma list of correspondence distances, in voxels, one per stage")
    g.add_argument("--refine-iters", type=int, default=100, help="max ICP iterations per stage")
    g.add_argument("--refine-tol", type=float, default=1e-7, help="relative convergence tolerance")
    g.add_argument("--refine-points", type=int, default=300000, help="dense points per side")
    g.add_argument("--refine-band", type=float, default=0.0,
                   help="0=off; else keep only points within band*threshold of the other side")
    g.add_argument("--refine-robust", type=float, default=0.0,
                   help="0=off; else Tukey robust loss with this sigma (in voxels)")
    sg = ap.add_argument_group("Seam resolution (drop the less confident side where the halves disagree)")
    sg.add_argument("--resolve-seam", action="store_true",
                    help="needs COLMAP .vis + images.bin next to each fused.ply (auto-found)")
    sg.add_argument("--seam-tau", type=float, default=0.5,
                    help="gap in voxels beyond which overlapping points count as a conflict")
    sg.add_argument("--seam-window", type=float, default=10.0, help="local window size in voxels")
    sg.add_argument("--seam-min-count", type=int, default=20,
                    help="other side must have this many points nearby for a conflict")
    sg.add_argument("--seam-margin", type=float, default=0.0,
                    help="local-confidence lead required before dropping a point")
    sg.add_argument("--seam-passes", type=int, default=3, help="re-evaluate after dropping, up to N times")
    sg.add_argument("--seam-mode", choices=["patch", "point"], default="point",
                    help="patch: compare competing sheets' local mean confidence; "
                         "point: drop points below the other side's local mean")
    sg.add_argument("--seam-min-conf", type=float, default=0.0,
                    help="also drop any point below this confidence wherever the other side "
                         "covers the surface (0=off; median confidence is about 0.5)")
    sg.add_argument("--seam-global-min-conf", type=float, default=0.0,
                    help="drop ANY point below this confidence, overlap or not (0=off)")
    sg.add_argument("--views-cap", type=int, default=8,
                    help="view count at which the view-count term of confidence saturates")
    g.add_argument("--report", default=None,
                   help="write ICP diagnostics JSON here (default: icp_report.json next to --out)")
    ap.add_argument("--view", action="store_true", help="open an Open3D window")
    args = ap.parse_args()

    A = align.load_mesh(args.A, n_sample=args.samples)
    B = align.load_mesh(args.B, n_sample=args.samples)

    fns = {"opening": align.align_opening, "fpfh": align.align_fpfh,
           "collapse": align.align_collapse}
    methods = list(fns) if args.method == "all" else [args.method]
    results = {}
    for m in methods:
        print(f"\n=== {m} ===")
        results[m] = fns[m](A, B, voxel=args.voxel)

    best = max(results.values(),
               key=lambda r: (round(r.closed, 3) >= 0.6, round(r.detail, 3)))
    print(f"\nBest method: {best.method}  closed={best.closed:.3f}  "
          f"detail_fit={best.detail:.3f}  fit={best.fitness:.3f}  rmse={best.rmse:.4g}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    T = best.transform
    if args.refine:
        print("\n=== refine ===")
        voxel = args.voxel or min(A.extent, B.extent) * 0.01
        params = align.RefineParams(
            thresholds=align.RefineParams.parse_thresholds(args.refine_thresholds),
            max_iters=args.refine_iters, tol=args.refine_tol, points=args.refine_points,
            band=args.refine_band, robust=args.refine_robust)
        T, report = align.refine_icp(A, B, T, voxel, params)
        report["method"] = best.method
        report["selection"] = {"fitness": best.fitness, "rmse": best.rmse,
                               "closed": best.closed, "detail": best.detail}
        rp = args.report or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                         "icp_report.json")
        with open(rp, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[icp] report -> {rp}")
    keepA = keepB = None
    if args.resolve_seam:
        import confidence
        print("\n=== resolve seam ===")
        cAll = confidence.confidence_for_cloud(args.A, args.views_cap)
        cBll = confidence.confidence_for_cloud(args.B, args.views_cap)
        if cAll is None or cBll is None or A.is_mesh or B.is_mesh or A.keep_idx is None:
            print("[seam] skipped: confidence data unavailable (need point-cloud PLYs with COLMAP .vis files)")
        else:
            cA, cB = cAll["conf"][A.keep_idx], cBll["conf"][B.keep_idx]
            voxel = args.voxel or min(A.extent, B.extent) * 0.01
            PA, PB = align.centered_points(A, B, T)
            keepA, keepB, srep = align.resolve_seam(
                PA, cA, PB, cB, voxel,
                align.SeamParams(args.seam_tau, args.seam_window, args.seam_min_count,
                                 args.seam_margin, args.seam_passes, args.seam_min_conf,
                                 args.seam_global_min_conf, args.seam_mode))
            base = os.path.splitext(os.path.abspath(args.out))[0]
            with open(base + "_seam_report.json", "w") as fh:
                json.dump(srep, fh, indent=1)
            align.save_seam_audit(A, B, T, keepA, keepB, base + "_seam_audit.ply")
            print(f"[seam] audit cloud -> {base}_seam_audit.ply")
    align.save_merged(A, B, T, args.out, keepA, keepB)
    print(f"Saved merged -> {args.out}")

    if args.view:
        import open3d as o3d
        o3d.visualization.draw_geometries(
            align.view_geometries(A, B, T=T),
            window_name=f"{best.method}  green=A red=B")


if __name__ == "__main__":
    main()
