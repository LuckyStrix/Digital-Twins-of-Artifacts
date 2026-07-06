"""
run.py -- headless CLI (no GUI). Aligns B onto A and writes a merged PLY.

  ./venv/bin/python run.py A.ply B.ply -o merged.ply [--method opening|fpfh|both]
                                                     [--voxel 0.5] [--view]
"""
import argparse
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
    align.save_merged(A, B, best.transform, args.out)
    print(f"Saved merged -> {args.out}")

    if args.view:
        import open3d as o3d
        o3d.visualization.draw_geometries(
            align.view_geometries(A, B, T=best.transform),
            window_name=f"{best.method}  green=A red=B")


if __name__ == "__main__":
    main()
