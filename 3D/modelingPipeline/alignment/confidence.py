"""
confidence.py -- per-point confidence for COLMAP fused point clouds.

COLMAP's stereo_fusion writes, next to each fused PLY, a `.ply.vis` file listing
which images observed every fused point. Together with the camera poses in the
sparse model's images.bin that gives two independent, physically meaningful
confidence cues per point:

  n_views  how many images agreed on the point (more = better constrained)
  cos      mean |cos| between the surface normal and the viewing rays. Grazing
           views (cos -> 0) have large depth uncertainty; head-on views are sharp.

Nothing here moves a point; confidence is only ever used to decide which points
to KEEP.
"""
import struct
import numpy as np
import open3d as o3d


def read_images_bin(path):
    """images.bin -> (image_ids sorted, camera centers Nx3 in the same order)."""
    ids, centers = [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (iid,) = struct.unpack("<I", f.read(4))
            q = np.array(struct.unpack("<4d", f.read(32)))          # w x y z (world->cam)
            t = np.array(struct.unpack("<3d", f.read(24)))
            f.read(4)                                               # camera_id
            while f.read(1) != b"\x00":                             # image name
                pass
            (n2d,) = struct.unpack("<Q", f.read(8))
            f.seek(n2d * 24, 1)                                     # skip points2D
            w, x, y, z = q
            R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                          [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                          [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])
            ids.append(iid)
            centers.append(-R.T @ t)                                # camera center
    order = np.argsort(ids)
    return np.asarray(ids)[order], np.asarray(centers)[order]


def read_vis(path):
    """.ply.vis -> list of int arrays (image indices per point)."""
    out = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        buf = np.frombuffer(f.read(), dtype="<u4")
    pos = 0
    for _ in range(n):
        k = int(buf[pos]); out.append(buf[pos + 1: pos + 1 + k]); pos += 1 + k
    return out


def load_with_confidence(ply, vis, images_bin, views_cap=8):
    """Returns (pcd, info) where info has per-point n_views, cos, conf in [0,1]."""
    pcd = o3d.io.read_point_cloud(ply)
    P, N = np.asarray(pcd.points), np.asarray(pcd.normals)
    _, cams = read_images_bin(images_bin)
    vis_list = read_vis(vis)
    if len(vis_list) != len(P):
        raise ValueError(f"{vis}: {len(vis_list)} vis entries but {len(P)} points")
    nv = np.array([len(v) for v in vis_list])
    cos = np.zeros(len(P))
    for i, v in enumerate(vis_list):
        v = v[v < len(cams)]
        if len(v) == 0:
            continue
        d = cams[v] - P[i]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        cos[i] = np.abs(d @ N[i]).mean()
    conf = np.clip(nv / views_cap, 0, 1) * cos
    return pcd, {"n_views": nv, "cos": cos, "conf": conf}


def find_colmap_inputs(fused_ply):
    """Locate the .vis file and images.bin that belong to a side's fused.ply.
    run_colmap_mvs.py writes one of two layouts:
      * several sparse models (the usual case with a stray small component):
          dense/fused_component_00.ply(.vis) + dense/component_00/sparse/images.bin
          (fused.ply is the merge of the components)
      * a single sparse model:
          fused.ply(.vis) itself + dense/sparse/images.bin
    Returns (ply, vis, images_bin) or None."""
    import os
    d = os.path.dirname(os.path.abspath(fused_ply))
    candidates = [
        (os.path.join(d, "dense", "fused_component_00.ply"),
         os.path.join(d, "dense", "component_00", "sparse", "images.bin")),
        (os.path.abspath(fused_ply),
         os.path.join(d, "dense", "sparse", "images.bin")),
    ]
    for ply, ib in candidates:
        vis = ply + ".vis"
        if all(os.path.exists(p) for p in (ply, vis, ib)):
            return ply, vis, ib
    return None


def confidence_for_cloud(fused_ply, views_cap=8, log=print):
    """Per-point confidence for EVERY point of `fused_ply` (same order), or None
    if the COLMAP side files are missing. Points not found in component 00
    (stragglers from other components) get confidence 0."""
    found = find_colmap_inputs(fused_ply)
    if found is None:
        log(f"  [conf] no .vis / images.bin next to {fused_ply}; confidence unavailable")
        return None
    try:
        comp, info = load_with_confidence(*found, views_cap=views_cap)
    except Exception as exc:                      # unreadable/mismatched .vis etc.
        log(f"  [conf] could not read confidence data for {fused_ply}: {exc}")
        return None
    fused = o3d.io.read_point_cloud(fused_ply)
    tree = o3d.geometry.KDTreeFlann(comp)
    F = np.asarray(fused.points)
    idx = np.empty(len(F), dtype=np.int64)
    miss = np.zeros(len(F), dtype=bool)
    C = np.asarray(comp.points)
    for i, p in enumerate(F):
        _, nn, d2 = tree.search_knn_vector_3d(p, 1)
        idx[i] = nn[0]
        miss[i] = d2[0] > 1e-12
    out = {k: v[idx] for k, v in info.items()}
    for k in out:
        out[k] = out[k].astype(float)
        out[k][miss] = 0.0
    log(f"  [conf] {fused_ply}: {len(F) - miss.sum()}/{len(F)} points matched "
        f"(median views={np.median(out['n_views'][~miss]):.0f}, "
        f"median cos={np.median(out['cos'][~miss]):.2f})")
    return out
