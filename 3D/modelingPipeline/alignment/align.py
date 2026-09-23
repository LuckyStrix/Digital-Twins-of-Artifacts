"""
align.py -- programmatic alignment of two half-scans of a cuneiform tablet.

Two methods are implemented, both ending in ICP refinement:

  1. "opening"  -- Geometry-aware initializer.
                   * Finds each mesh's OPEN FACE direction two ways:
                       (a) area-weighted normal sum  (closed surface => sum = 0,
                           so an open shell's net normal points opposite the hole)
                       (b) boundary-loop (rim) centroid
                   * Builds a PCA frame (long / mid / short axes).
                   * Generates a small set of discrete candidate poses that make
                     the two open faces face each other and line up the long axes,
                     translating so the rim seams coincide (this is the "shift
                     inward" step, sized automatically instead of by a magic number).
                   * Runs ICP from each candidate and keeps the best by fitness.

  2. "fpfh"     -- Feature-based global registration (FPFH + RANSAC) followed by
                   ICP. Needs no knowledge of the open face; relies on the
                   overlapping side-wall detail.

Both return an AlignResult (4x4 transform mapping SOURCE onto TARGET, plus
fitness / rmse from a common evaluation threshold).

Convention: TARGET (a.k.a. mesh A) is fixed; SOURCE (mesh B) is moved onto it.
"""

from dataclasses import dataclass, field
import copy
import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration


# --------------------------------------------------------------------------- #
#  Small linear-algebra helpers
# --------------------------------------------------------------------------- #
def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def _perp(v):
    """Return some unit vector perpendicular to v."""
    a = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _normalize(np.cross(v, a))


def _frame(d_open, e_long):
    """Right-handed 3x3 rotation whose 3rd column is the opening direction and
    1st column is the long PCA axis (projected orthogonal to the opening)."""
    u3 = _normalize(d_open)
    u1 = e_long - np.dot(e_long, u3) * u3
    if np.linalg.norm(u1) < 1e-6:        # long axis ~parallel to opening: fall back
        u1 = _perp(u3)
    u1 = _normalize(u1)
    u2 = np.cross(u3, u1)
    return np.column_stack([u1, u2, u3])


def _T(R=np.eye(3), t=np.zeros(3)):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _flip_about(p0, axis):
    """4x4 transform: 180-deg rotation about the line through p0 along `axis`."""
    a = _normalize(axis)
    R = 2.0 * np.outer(a, a) - np.eye(3)      # 180-deg rotation about a
    return _T(R, p0 - R @ p0)


# --------------------------------------------------------------------------- #
#  Mesh container + feature extraction
# --------------------------------------------------------------------------- #
@dataclass
class MeshData:
    path: str
    mesh: o3d.geometry.TriangleMesh       # None when the input was a point cloud
    pcd: o3d.geometry.PointCloud          # sampled, normals estimated, CENTERED
    centroid: np.ndarray                  # original centroid (added back at the end)
    d_open: np.ndarray                    # unit opening direction (centered frame)
    c_rim: np.ndarray                     # rim/seam centroid (centered frame)
    e_long: np.ndarray                    # long PCA axis
    e_mid: np.ndarray
    e_short: np.ndarray
    extent: float                         # bbox diagonal (for scale / voxel)
    is_mesh: bool = True
    full: object = None                   # full-res geometry for output (mesh or cloud)


def _opening_direction(mesh):
    """(a) area-weighted normal sum.  For a closed surface the area-weighted
    normal integral is zero; an open shell's net normal therefore points
    OPPOSITE the hole, so the opening is the negative of that sum.
    Robust to winding sign because callers also try the flipped sign."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    cross = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])  # = 2*area*unit_n
    net = cross.sum(axis=0)
    return _normalize(-net)


def _rim_centroid(mesh, centroid, d_open, half_extent):
    """(b) boundary-loop centroid: edges used by exactly one triangle bound the
    open face. Returns its centroid (the seam). Falls back to a synthetic seam
    a bit out along the opening axis if the mesh has no boundary."""
    t = np.asarray(mesh.triangles)
    v = np.asarray(mesh.vertices)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    boundary = uniq[cnt == 1]
    if boundary.shape[0] == 0:
        return centroid + d_open * (0.5 * half_extent)   # synthetic seam
    bverts = np.unique(boundary.reshape(-1))
    return v[bverts].mean(axis=0)


def _pca(points):
    cov = np.cov(points.T)
    _, V = np.linalg.eigh(cov)            # ascending eigenvalues
    return V[:, 2], V[:, 1], V[:, 0]      # long, mid, short


def load_mesh(path, n_sample=60000):
    """Accepts either a triangle-mesh PLY or a point-cloud PLY (e.g. a COLMAP
    'fused.ply'). Returns a MeshData usable by both alignment methods."""
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        return _load_from_mesh(path, mesh, n_sample)
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"{path}: no triangles and no points (unreadable PLY?)")
    return _load_from_cloud(path, pcd, n_sample)


def _load_from_mesh(path, mesh, n_sample):
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    centroid = mesh.get_center()
    extent = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))

    d_open = _opening_direction(mesh)
    proj = np.asarray(mesh.vertices) @ d_open
    half_extent = float(proj.max() - proj.min()) * 0.5
    c_rim = _rim_centroid(mesh, centroid, d_open, half_extent) - centroid

    pcd = mesh.sample_points_uniformly(number_of_points=n_sample)
    pcd.translate(-centroid)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=extent * 0.02, max_nn=30))

    e_long, e_mid, e_short = _pca(np.asarray(pcd.points))
    return MeshData(path, mesh, pcd, centroid, d_open, c_rim,
                    e_long, e_mid, e_short, extent, is_mesh=True, full=mesh)


def _load_from_cloud(path, pcd, n_sample, clean=True):
    if clean and len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    centroid = pcd.get_center()
    extent = float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))

    if not pcd.has_normals():
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
            radius=extent * 0.02, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)

    # opening dir: net of (consistent) normals points opposite the hole.
    # A global sign flip is harmless -- the os=+/-1 candidates cover it.
    n = np.asarray(pcd.normals)
    d_open = _normalize(-n.sum(axis=0))
    proj = np.asarray(pcd.points) @ d_open
    half_extent = float(proj.max() - proj.min()) * 0.5
    c_rim = d_open * (0.5 * half_extent)          # synthetic seam (no boundary loop)

    # registration cloud: downsample big clouds to ~n_sample, center, ensure normals
    reg_pcd = o3d.geometry.PointCloud(pcd)
    if len(reg_pcd.points) > n_sample:
        every = max(1, len(reg_pcd.points) // n_sample)
        reg_pcd = reg_pcd.uniform_down_sample(every)
    reg_pcd.translate(-centroid)
    reg_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=extent * 0.02, max_nn=30))

    e_long, e_mid, e_short = _pca(np.asarray(reg_pcd.points))
    return MeshData(path, None, reg_pcd, centroid, d_open, c_rim,
                    e_long, e_mid, e_short, extent, is_mesh=False, full=pcd)


# --------------------------------------------------------------------------- #
#  Results + evaluation
# --------------------------------------------------------------------------- #
@dataclass
class AlignResult:
    method: str
    transform: np.ndarray                 # maps SOURCE (centered) onto TARGET (centered)
    fitness: float
    rmse: float
    closed: float = 0.0
    detail: float = 0.0                   # fine-scale overlap fitness (spin decider)
    log: list = field(default_factory=list)
    candidates: list = field(default_factory=list)   # [(name, fit, rmse, closed), ...]


def _evaluate(src, tgt, T, thr):
    r = reg.evaluate_registration(src, tgt, thr, T)
    return r.fitness, r.inlier_rmse


def _icp(src, tgt, T_init, thr, iters=60):
    return reg.registration_icp(
        src, tgt, thr, T_init,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=iters))


def _closedness(A: MeshData, B: MeshData, T):
    """How closed the merged surface is. The two halves only fit together one
    way: the correct assembly completes the (closed) tablet surface, so the
    union's unit normals nearly cancel (-> 1.0). A degenerate 'collapse' that
    stacks one half onto the other facing the same way leaves an open double
    cap whose normals do NOT cancel (-> low). Invariant to a global normal flip,
    so it works even when point-cloud normals were only consistently oriented."""
    R = T[:3, :3]
    ns = np.asarray(B.pcd.normals) @ R.T
    nt = np.asarray(A.pcd.normals)
    n = np.vstack([ns, nt])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return float(1.0 - np.linalg.norm(n.mean(axis=0)))


def _select(scored, fit_floor, A, B, voxel, log=print):
    """scored: list of (name, T, fit_coarse, rmse_coarse, closed).

    Two-stage choice, because two different ambiguities need two different signals:
      1. CLOSEDNESS gates out the 'collapse' (one half stacked on the other facing
         the same way) -- it leaves an open surface, so closedness is low.
      2. Among the surviving CLOSED poses there is still a 180-deg in-plane spin
         about the seam-normal axis. Both spins close equally well, so closedness
         can't choose; only the actual surface DETAIL coincides in the right spin.
         We tighten ICP and score FINE-scale overlap fitness -- the wrong spin
         loses inliers because its detail is misregistered.
    Returns (name, T, fit_coarse, rmse_coarse, closed, detail_fit)."""
    src, tgt = B.pcd, A.pcd
    cmax = max(c[4] for c in scored)
    survivors = [c for c in scored
                 if c[2] >= fit_floor and c[4] >= max(0.6, cmax - 0.2)]
    if not survivors:                       # nothing closed: fall back to overlap
        survivors = [c for c in scored if c[2] >= fit_floor] or list(scored)

    thr_fine = voxel * 0.6
    ranked = []
    for name, T, f, e, c in survivors:
        r = _icp(src, tgt, T, voxel * 1.0, iters=40)      # tighten onto the detail
        Tf = r.transformation
        cc = _closedness(A, B, Tf)
        df = reg.evaluate_registration(src, tgt, thr_fine, Tf)
        cf = reg.evaluate_registration(src, tgt, voxel * 1.5, Tf)
        ranked.append((name, Tf, cf.fitness, cf.inlier_rmse, cc, df.fitness, df.inlier_rmse))
        log(f"   refine {name:18s} closed={cc:.3f}  detail_fit={df.fitness:.3f}")
    best = max(ranked, key=lambda r: (round(r[5], 3), -r[6]))
    return best[:6]


# --------------------------------------------------------------------------- #
#  Method 1: opening-direction candidates + multi-start ICP
# --------------------------------------------------------------------------- #
def align_opening(A: MeshData, B: MeshData, voxel=None, log=print, fit_floor=0.12):
    """A = target (fixed), B = source (moved)."""
    scale = min(A.extent, B.extent)
    voxel = voxel or scale * 0.01
    thr_icp = voxel * 3.0
    thr_eval = voxel * 1.5
    src, tgt = B.pcd, A.pcd

    R_A_long = A.e_long
    cands = []
    # enumerate: long-axis sign s, and B-opening sign os (covers winding ambiguity)
    for os in (+1.0, -1.0):
        R_B = _frame(B.d_open * os, B.e_long)
        c_rim_B = B.c_rim                       # rotated below
        for s in (+1.0, -1.0):
            u3_t = -A.d_open                     # B opening opposite A opening
            u1_t = s * _normalize(R_A_long - np.dot(R_A_long, u3_t) * u3_t)
            if np.linalg.norm(u1_t) < 1e-6:
                u1_t = _perp(u3_t)
            u2_t = np.cross(u3_t, u1_t)
            Tg = np.column_stack([u1_t, u2_t, u3_t])
            Rot = Tg @ R_B.T
            # translate so B's rim seam lands on A's rim seam (the "shift inward")
            t = A.c_rim - Rot @ c_rim_B
            cands.append((f"open[os={int(os):+d},long={int(s):+d}]", _T(Rot, t)))

    scored = []
    for name, T0 in cands:
        res = _icp(src, tgt, T0, thr_icp)
        f, e = _evaluate(src, tgt, res.transformation, thr_eval)
        c = _closedness(A, B, res.transformation)
        scored.append((name, res.transformation, f, e, c))
        log(f"  {name:24s} fit={f:.3f}  rmse={e:.4g}  closed={c:.3f}")

    name, T, f, e, c, d = _select(scored, fit_floor, A, B, voxel, log)
    log(f"  chosen {name}: closed={c:.3f} detail_fit={d:.3f} fit={f:.3f}")
    return AlignResult("opening", T, f, e, closed=c, detail=d,
                       candidates=[(s[0], s[2], s[3], s[4]) for s in scored])


# --------------------------------------------------------------------------- #
#  Method 2: FPFH + RANSAC global registration + ICP
# --------------------------------------------------------------------------- #
def _fpfh(pcd, voxel):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * 2.0, max_nn=30))
    f = reg.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * 5.0, max_nn=100))
    return down, f


def align_fpfh(A: MeshData, B: MeshData, voxel=None, log=print, fit_floor=0.12):
    scale = min(A.extent, B.extent)
    voxel = voxel or scale * 0.012
    thr_eval = voxel * 1.5
    src, tgt = B.pcd, A.pcd

    sdown, sf = _fpfh(src, voxel)
    tdown, tf = _fpfh(tgt, voxel)
    log(f"  FPFH on {len(sdown.points)} / {len(tdown.points)} pts (voxel={voxel:.4g})")

    ransac = reg.registration_ransac_based_on_feature_matching(
        sdown, tdown, sf, tf, True, voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        reg.RANSACConvergenceCriteria(400000, 0.999))
    log(f"  RANSAC fitness={ransac.fitness:.3f}")

    # FPFH/RANSAC maximizes inliers, so on these symmetric halves it favors the
    # 'collapse' (one half stacked onto the other, facing the same way). Try the
    # RANSAC pose plus 180-deg flips about the SEAM (rim centroid) -- the pivot
    # that swings one half to the opposite side -- using A's principal axes.
    # Flipping about the centroid instead would throw the half off the object.
    p0 = A.c_rim
    flips = {"id": None,
             "seamLong": A.e_long, "seamMid": A.e_mid, "seamShort": A.e_short}
    scored = []
    for fn, axis in flips.items():
        T0 = (ransac.transformation if axis is None
              else _flip_about(p0, axis) @ ransac.transformation)
        res = _icp(src, tgt, T0, voxel * 3.0)
        f, e = _evaluate(src, tgt, res.transformation, thr_eval)
        c = _closedness(A, B, res.transformation)
        scored.append((f"fpfh[{fn}]", res.transformation, f, e, c))
        log(f"  {f'fpfh[{fn}]':24s} fit={f:.3f}  rmse={e:.4g}  closed={c:.3f}")

    name, T, f, e, c, d = _select(scored, fit_floor, A, B, voxel, log)
    log(f"  chosen {name}: closed={c:.3f} detail_fit={d:.3f} fit={f:.3f}")
    return AlignResult("fpfh", T, f, e, closed=c, detail=d,
                       candidates=[(s[0], s[2], s[3], s[4]) for s in scored])


# --------------------------------------------------------------------------- #
#  Method 3: deliberate collapse -> 180-deg seam flip -> ICP
# --------------------------------------------------------------------------- #
def _pca_frame(md: MeshData):
    return np.column_stack([md.e_long, md.e_mid, md.e_short])


def align_collapse(A: MeshData, B: MeshData, voxel=None, log=print, fit_floor=0.12):
    """The robust 'collapse then flip' idea: first deliberately stack the two
    halves on top of each other facing the same way (the easy, reliable pose --
    it's the overlap maximum), then flip one half 180 deg about the seam to swing
    it to the correct side. Needs no opening-normal estimate; closedness picks
    the flip. PCA frames give the collapse, ICP cleans up each stage."""
    scale = min(A.extent, B.extent)
    voxel = voxel or scale * 0.01
    thr_eval = voxel * 1.5
    src, tgt = B.pcd, A.pcd

    # 1) COLLAPSE: align PCA frames (4 proper sign combos), keep the MAX-overlap one
    FA, FB = _pca_frame(A), _pca_frame(B)
    base = None
    for d in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]:
        R = FA @ np.diag(d).astype(float) @ FB.T
        res = _icp(src, tgt, _T(R), voxel * 3.0)
        f, _ = _evaluate(src, tgt, res.transformation, thr_eval)
        if base is None or f > base[1]:
            base = (res.transformation, f)
    Tc = base[0]
    log(f"  collapsed (overlap fit={base[1]:.3f}); now flipping about seam")

    # 2) FLIP 180 deg about the seam to the correct side; closedness decides
    flips = {"none": None, "flipLong": A.e_long,
             "flipMid": A.e_mid, "flipShort": A.e_short}
    scored = []
    for fn, axis in flips.items():
        T0 = Tc if axis is None else _flip_about(A.c_rim, axis) @ Tc
        res = _icp(src, tgt, T0, voxel * 3.0)
        f, e = _evaluate(src, tgt, res.transformation, thr_eval)
        c = _closedness(A, B, res.transformation)
        scored.append((f"collapse[{fn}]", res.transformation, f, e, c))
        log(f"  {f'collapse[{fn}]':24s} fit={f:.3f}  rmse={e:.4g}  closed={c:.3f}")

    name, T, f, e, c, d = _select(scored, fit_floor, A, B, voxel, log)
    log(f"  chosen {name}: closed={c:.3f} detail_fit={d:.3f} fit={f:.3f}")
    return AlignResult("collapse", T, f, e, closed=c, detail=d,
                       candidates=[(s[0], s[2], s[3], s[4]) for s in scored])


# --------------------------------------------------------------------------- #
#  Final refinement: coarse-to-fine ICP with per-iteration diagnostics
# --------------------------------------------------------------------------- #
@dataclass
class RefineParams:
    """Knobs for refine_icp(). Thresholds are multiples of `voxel`."""
    thresholds: tuple = (3.0, 1.0, 0.5, 0.25)   # correspondence dist per stage
    max_iters: int = 100          # ICP iteration cap PER STAGE
    tol: float = 1e-7             # relative fitness/rmse convergence tolerance
    points: int = 300000          # dense points sampled per side for refinement
    band: float = 0.0             # >0: only keep points within band*thr of the other side
    robust: float = 0.0           # >0: Tukey loss with this sigma (in voxels)
    chunk: int = 10               # iterations between logged trace points

    @staticmethod
    def parse_thresholds(text):
        vals = tuple(float(x) for x in str(text).replace(";", ",").split(",") if x.strip())
        if not vals or any(v <= 0 for v in vals):
            raise ValueError(f"bad refine thresholds: {text!r}")
        return vals


def _dense_cloud(md: MeshData, n):
    """Centered, denser sample of the FULL geometry (the 60k registration cloud
    is too coarse to resolve sub-voxel offsets)."""
    g = md.full
    if md.is_mesh:
        pcd = g.sample_points_uniformly(number_of_points=n)
    else:
        pcd = o3d.geometry.PointCloud(g)
        if len(pcd.points) > n:
            pcd = pcd.uniform_down_sample(max(1, len(pcd.points) // n))
    pcd.translate(-md.centroid)
    return pcd


def _residuals(src, tgt, T, thr):
    """Point-to-plane residuals |(s - t) . n_t| over correspondences within thr.
    Unsigned: target normals are not globally oriented."""
    r = reg.evaluate_registration(src, tgt, thr, T)
    corr = np.asarray(r.correspondence_set)
    if corr.shape[0] == 0:
        return r, np.zeros(0)
    s = (np.asarray(src.points)[corr[:, 0]] @ T[:3, :3].T) + T[:3, 3]
    t = np.asarray(tgt.points)[corr[:, 1]]
    n = np.asarray(tgt.normals)[corr[:, 1]]
    return r, np.abs(np.einsum("ij,ij->i", s - t, n))


def _res_summary(d, voxel):
    if d.size == 0:
        return {"n": 0}
    return {"n": int(d.size), "mean": float(d.mean()),
            "median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
            "median_vox": float(np.median(d) / voxel)}


def _histogram(d, voxel, bins=40, hi=2.0):
    """Histogram of residuals in voxel units on [0, hi]. A doubled surface shows
    as a peak away from 0 instead of a spike at 0."""
    h, edges = np.histogram(np.clip(d / voxel, 0, hi), bins=bins, range=(0, hi))
    return {"counts": h.tolist(), "hi_vox": hi}


def _pose_delta(Ta, Tb):
    """(translation, rotation-deg) between two poses."""
    D = Tb @ np.linalg.inv(Ta)
    c = np.clip((np.trace(D[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.linalg.norm(D[:3, 3])), float(np.degrees(np.arccos(c)))


def refine_icp(A: MeshData, B: MeshData, T, voxel, params: RefineParams = None, log=print):
    """Polish pose T (source->target, centered frame) with a coarse-to-fine ICP
    on dense clouds. Returns (T_refined, report_dict). Every stage is run in
    chunks of `params.chunk` iterations so the fitness/rmse trace shows whether
    ICP is still improving or has plateaued (a plateau with a still-visible
    double surface means the leftover error is non-rigid)."""
    p = params or RefineParams()
    log(f"[icp] refine: voxel={voxel:.5g} thresholds={list(p.thresholds)} "
        f"max_iters={p.max_iters} points={p.points} band={p.band} robust={p.robust}")
    src_full = _dense_cloud(B, p.points)
    tgt_full = _dense_cloud(A, p.points)
    tgt_full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=A.extent * 0.01, max_nn=30))
    log(f"[icp] dense clouds: src={len(src_full.points)} tgt={len(tgt_full.points)}")

    kernel = None
    if p.robust > 0:
        try:
            kernel = reg.TukeyLoss(k=p.robust * voxel)
        except Exception as e:                      # older Open3D
            log(f"[icp] robust kernel unavailable ({e}); using plain L2")
    est = (reg.TransformationEstimationPointToPlane(kernel) if kernel
           else reg.TransformationEstimationPointToPlane())

    eval_thr = voxel * 2.0
    r0, d0 = _residuals(src_full, tgt_full, T, eval_thr)
    before = _res_summary(d0, voxel)
    before["fitness"], before["rmse"] = float(r0.fitness), float(r0.inlier_rmse)
    log(f"[icp] before: fitness={r0.fitness:.4f} rmse={r0.inlier_rmse:.4g} "
        f"median|d|={before.get('median_vox', 0):.3f}vox p95={before.get('p95', 0):.4g}")

    T0, cur = T.copy(), T.copy()
    stages = []
    for si, mult in enumerate(p.thresholds):
        thr = voxel * mult
        src, tgt = src_full, tgt_full
        if p.band > 0:
            # keep only the seam region: points that have a partner within band*thr
            moved = copy.deepcopy(src_full).transform(cur.copy())
            sd = np.asarray(moved.compute_point_cloud_distance(tgt_full))
            td = np.asarray(tgt_full.compute_point_cloud_distance(moved))
            src = src_full.select_by_index(np.where(sd < p.band * thr)[0])
            tgt = tgt_full.select_by_index(np.where(td < p.band * thr)[0])
            if len(src.points) < 100 or len(tgt.points) < 100:
                log(f"[icp] stage {si+1}: band left too few points, using all")
                src, tgt = src_full, tgt_full
        log(f"[icp] stage {si+1}/{len(p.thresholds)} thr={thr:.5g} ({mult:g} vox) "
            f"src={len(src.points)} tgt={len(tgt.points)}")
        trace, done, prev = [], 0, None
        Tst = cur.copy()
        while done < p.max_iters:
            n = min(p.chunk, p.max_iters - done)
            res = reg.registration_icp(
                src, tgt, thr, cur, est,
                reg.ICPConvergenceCriteria(relative_fitness=p.tol,
                                           relative_rmse=p.tol, max_iteration=n))
            cur = res.transformation
            done += n
            trace.append({"iter": done, "fitness": float(res.fitness),
                          "rmse": float(res.inlier_rmse)})
            log(f"[icp]   iter {done:4d}: fitness={res.fitness:.4f} rmse={res.inlier_rmse:.5g}")
            # a chunk that stopped early on convergence keeps returning the same
            # numbers; treat an unchanged rmse as converged
            if prev is not None and abs(prev - res.inlier_rmse) <= p.tol * max(prev, 1e-12):
                break
            prev = res.inlier_rmse
        dt, dr = _pose_delta(Tst, cur)
        stages.append({"stage": si + 1, "threshold": thr, "mult": mult,
                       "n_src": len(src.points), "n_tgt": len(tgt.points),
                       "iters": done, "trace": trace,
                       "move_translation": dt, "move_rotation_deg": dr})
        log(f"[icp]   stage {si+1} moved pose by {dt:.4g} ({dt/voxel:.3f} vox), {dr:.4f} deg")

    r1, d1 = _residuals(src_full, tgt_full, cur, eval_thr)
    after = _res_summary(d1, voxel)
    after["fitness"], after["rmse"] = float(r1.fitness), float(r1.inlier_rmse)
    tdt, tdr = _pose_delta(T0, cur)
    log(f"[icp] after:  fitness={r1.fitness:.4f} rmse={r1.inlier_rmse:.4g} "
        f"median|d|={after.get('median_vox', 0):.3f}vox p95={after.get('p95', 0):.4g}")
    log(f"[icp] total pose change {tdt:.4g} ({tdt/voxel:.3f} vox), {tdr:.4f} deg")
    report = {"voxel": voxel,
              "params": {"thresholds": list(p.thresholds), "max_iters": p.max_iters,
                         "tol": p.tol, "points": p.points, "band": p.band,
                         "robust": p.robust},
              "before": before, "after": after,
              "hist_before": _histogram(d0, voxel), "hist_after": _histogram(d1, voxel),
              "stages": stages, "total_move_translation": tdt,
              "total_move_rotation_deg": tdr}
    return cur, report


# --------------------------------------------------------------------------- #
#  Visualization + output
# --------------------------------------------------------------------------- #
def _centered_full(md: MeshData):
    """Full-res geometry (mesh or cloud) moved to the centered frame."""
    g = copy.deepcopy(md.full)
    return g.translate(-md.centroid)


def view_geometries(A: MeshData, B: MeshData, T=None):
    """Colored copies for an Open3D window. Target=green, Source=red.
    T maps the (centered) source onto the (centered) target."""
    gt = _centered_full(A)
    gs = _centered_full(B)
    gt.paint_uniform_color([0.30, 0.75, 0.35])
    gs.paint_uniform_color([0.85, 0.25, 0.25])
    if T is not None:
        gs.transform(T)
    size = 0.3 * max(A.extent, B.extent)
    return [gt, gs, o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)]


def merged_mesh(A: MeshData, B: MeshData, T):
    """Single combined geometry in the TARGET's original world frame.
    Returns a TriangleMesh if both inputs were meshes, else a PointCloud."""
    gt = _centered_full(A)
    gs = _centered_full(B).transform(T)
    combined = gt + gs
    combined.translate(A.centroid)        # back to target's original world coords
    if A.is_mesh and B.is_mesh:
        combined.compute_vertex_normals()
    return combined


def save_merged(A, B, T, path):
    combined = merged_mesh(A, B, T)
    if isinstance(combined, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(path, combined)
    else:
        o3d.io.write_point_cloud(path, combined)
    return path
