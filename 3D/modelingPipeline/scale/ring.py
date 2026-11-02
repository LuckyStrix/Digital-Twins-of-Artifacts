"""Find the reference ring in the raw photos and fit one 3D circle to it.

The ring (white top face on black felt, fixed to the turntable) is in every
raw photo, and COLMAP has already solved where each camera was. So instead of
fitting an ellipse per photo, this fits a single 3D circle (center, tilt,
inner and outer radius) to the ring's edges in all photos at once:

1. Turntable axis: each physical camera (cam1, cam2, ...) traces a circle
   around the axis as the turntable steps, so the axis is the shared normal
   of those circles. Used for the starting guess and as a check.
2. Starting guess: a circle centered on the axis and perpendicular to it.
   Grid-search its height and radius for the brightest thin band (white ring
   between dark felt) in a few downscaled photos.
3. Edges: walk across the predicted ring in each full-size photo and find the
   inner edge (dark → bright going outward) and outer edge (bright → dark),
   to sub-pixel accuracy. Edges on the tablet (Stage 1's mask) are skipped.
4. Fit: cast each edge pixel's ray onto the ring plane; the residual is its
   distance from the center minus the inner or outer radius. Robust
   Gauss-Newton (Huber weights) over center, tilt and both radii, then redo
   the edges with the new circle and fit again.
5. Scale: mm per model unit = centerline radius in mm / fitted centerline
   radius, so any edge bias cancels. Uncertainty from a bootstrap over photos.

Raw photos are read with EXIF orientation ignored: Stage 1 never applies it
and exiftool strips it before COLMAP, so COLMAP sees the sensor's own layout.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .colmap_model import Camera, Image, Model, project, undistort_to_normalized

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Limits on how many photos each step reads (spread across the cameras).
INIT_VIEWS = 16
FIT_VIEWS = 48
DEBUG_VIEWS = 12

SAMPLES_PER_EDGE = 720          # points around the ring per photo and edge
MIN_VIEWS = 6                   # photos with enough edges for a fit
MIN_EDGES_PER_VIEW = 30
MIN_EDGES_PER_KIND = 300
MIN_INIT_CONTRAST = 0.12        # band brightness above the felt, 0-1 scale
EDGE_MIN_STEP = 0.10            # intensity step across an accepted edge, 0-1 scale

# Checks. Any FAIL_* failing leaves the scan unscaled; WARN_* only warn.
FAIL_SIGMA_PCT = 0.5            # bootstrap uncertainty of the scale
FAIL_INNER_OUTER_PCT = 1.0      # inner-only vs outer-only scale
FAIL_GROUP_SPREAD_PCT = 1.0     # per-camera scales, max - min
FAIL_WIDTH_PCT = 25.0           # fitted band width vs expected width
WARN_OFF_AXIS_MM = 10.0
WARN_TILT_DEG = 3.0


@dataclass
class RingSpec:
    inner_mm: float = 200.0     # inner diameter
    outer_mm: float = 216.0     # outer diameter
    width_mm: float = 0.0       # caliper-measured band width; 0 = not measured

    @property
    def mid_radius_mm(self) -> float:
        return (self.inner_mm + self.outer_mm) / 4.0

    @property
    def expected_width_mm(self) -> float:
        return self.width_mm if self.width_mm > 0 else (self.outer_mm - self.inner_mm) / 2.0

    def validate(self) -> str:
        if not (self.inner_mm > 0 and self.outer_mm > self.inner_mm):
            return f"ring diameters must satisfy 0 < inner < outer (got {self.inner_mm}, {self.outer_mm})"
        return ""


@dataclass
class RingFit:
    ok: bool
    reason: str = ""
    scale: float = 0.0              # mm per model unit
    sigma_pct: float = 0.0
    r_in: float = 0.0               # model units
    r_out: float = 0.0
    center: list = field(default_factory=list)
    normal: list = field(default_factory=list)   # points up, toward the cameras
    scale_inner: float = 0.0
    scale_outer: float = 0.0
    width_mm: float = 0.0
    per_camera: dict = field(default_factory=dict)
    off_axis_mm: float = 0.0
    tilt_deg: float = 0.0
    residual_mm: float = 0.0        # robust sigma of the edge residuals
    n_views: int = 0
    n_edges_inner: int = 0
    n_edges_outer: int = 0
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.floating,)):
                d[k] = float(v)
        return d


def failed(reason: str, **kw) -> RingFit:
    return RingFit(ok=False, reason=reason, **kw)


# ── photos ────────────────────────────────────────────────────────────────────

@dataclass
class View:
    image: Image
    camera: Camera
    raw_path: Path
    mask_path: Path | None
    group: str
    f: float = 1.0                  # COLMAP pixels per raw pixel
    raw_size: tuple = (0, 0)        # (w, h)


def _index_by_stem(root: Path | None) -> dict[str, Path]:
    """Relative path without extension (lowercase, '/') → file."""
    out: dict[str, Path] = {}
    if root is None or not Path(root).is_dir():
        return out
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            key = p.relative_to(root).with_suffix("").as_posix().lower()
            out.setdefault(key, p)
    return out


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            return im.size          # stored size; EXIF orientation not applied
    except Exception:
        return None


def pair_views(model: Model, raw_root: Path, processed_root: Path | None,
               log=print) -> list[View]:
    raw = _index_by_stem(raw_root)
    proc = _index_by_stem(processed_root)
    views, unmatched, bad_size, unsupported = [], 0, 0, 0
    from .colmap_model import camera_matrix_and_dist
    for img in sorted(model.images.values(), key=lambda i: i.name):
        key = Path(img.name).with_suffix("").as_posix().lower()
        rp = raw.get(key)
        if rp is None:
            unmatched += 1
            continue
        cam = model.cameras[img.camera_id]
        if camera_matrix_and_dist(cam) is None:
            unsupported += 1
            continue
        size = _image_size(rp)
        if size is None:
            unmatched += 1
            continue
        fx, fy = cam.width / size[0], cam.height / size[1]
        if abs(fx - fy) > 0.01 * fx:
            bad_size += 1
            continue
        group = img.name.split("/")[0] if "/" in img.name else "root"
        views.append(View(img, cam, rp, proc.get(key), group, fx, size))
    if unmatched:
        log(f"[scale] {unmatched} photo(s) in the model have no matching raw photo")
    if bad_size:
        log(f"[scale] {bad_size} raw photo(s) don't match their camera's aspect ratio; skipped")
    if unsupported:
        log(f"[scale] {unsupported} photo(s) use a camera model the scale step can't undistort")
    return views


def _spread(views: list[View], n: int) -> list[View]:
    """Up to n views, spread evenly over each camera group."""
    groups: dict[str, list[View]] = {}
    for v in views:
        groups.setdefault(v.group, []).append(v)
    per = max(1, n // max(1, len(groups)))
    out = []
    for vs in groups.values():
        idx = np.linspace(0, len(vs) - 1, min(per, len(vs))).round().astype(int)
        out += [vs[i] for i in sorted(set(idx.tolist()))]
    return out[:max(n, len(groups))]


def _read_gray(path: Path, reduce: int = 1) -> np.ndarray | None:
    import cv2
    flags = cv2.IMREAD_IGNORE_ORIENTATION | {
        1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
        4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduce]
    img = cv2.imread(str(path), flags)
    if img is None:
        return None
    img = img.astype(np.float32)
    if img.max() > 255:
        img /= 65535.0
    else:
        img /= 255.0
    lo, hi = np.percentile(img, (1, 99))
    if hi - lo < 1e-3:
        return None
    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32)


def _read_mask(view: View, shape: tuple[int, int], dilate_frac: float = 0.012) -> np.ndarray | None:
    """Tablet mask from Stage 1's output (alpha, or difference from the flat
    fill colour), dilated, at ``shape`` (h, w). None if there's no processed photo."""
    import cv2
    if view.mask_path is None:
        return None
    im = cv2.imread(str(view.mask_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if im is None:
        return None
    if im.ndim == 3 and im.shape[2] == 4:
        m = im[:, :, 3] > 127
    else:
        rgb = im if im.ndim == 3 else im[:, :, None]
        border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).reshape(-1, rgb.shape[2])
        fill = np.median(border, axis=0)
        m = np.abs(rgb.astype(np.int16) - fill.astype(np.int16)).max(axis=2) > 12
    m = m.astype(np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    r = max(1, int(dilate_frac * math.hypot(*shape)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m.astype(bool)


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear samples at index coords (x, y); returns (values, valid)."""
    h, w = img.shape
    valid = (x >= 0) & (y >= 0) & (x <= w - 1.001) & (y <= h - 1.001) & np.isfinite(x) & np.isfinite(y)
    xs = np.where(valid, x, 0.0)
    ys = np.where(valid, y, 0.0)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx, fy = xs - x0, ys - y0
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
         + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)
    return np.where(valid, v, np.nan), valid


def _lookup(mask: np.ndarray | None, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if mask is None:
        return np.zeros(x.shape, dtype=bool)
    h, w = mask.shape
    xi = np.clip(np.round(np.nan_to_num(x)).astype(np.int64), 0, w - 1)
    yi = np.clip(np.round(np.nan_to_num(y)).astype(np.int64), 0, h - 1)
    return mask[yi, xi]


# ── geometry ──────────────────────────────────────────────────────────────────

def _basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def circle_points(c, n, r, theta) -> tuple[np.ndarray, np.ndarray]:
    """Points on the circle and their outward radial unit vectors."""
    e1, e2 = _basis(n)
    u = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    return c + r * u, u


def fit_plane(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c)
    return c, vt[2]


def fit_circle_3d(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Least-squares circle through 3D points: center, normal, radius."""
    if len(P) < 3:
        return None
    c0, n = fit_plane(P)
    e1, e2 = _basis(n)
    q = np.column_stack([(P - c0) @ e1, (P - c0) @ e2])
    A = np.column_stack([2 * q, np.ones(len(q))])
    b = (q ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, k = sol
    r2 = k + cx * cx + cy * cy
    if r2 <= 0:
        return None
    return c0 + cx * e1 + cy * e2, n, float(math.sqrt(r2))


@dataclass
class Axis:
    point: np.ndarray
    direction: np.ndarray           # unit, "up" (cameras look down along -direction)
    cam_radius: float               # largest camera distance from the axis
    cam_height: float               # mean camera height along the axis


def turntable_axis(views: list[View]) -> Axis | None:
    groups: dict[str, list[np.ndarray]] = {}
    for v in views:
        groups.setdefault(v.group, []).append(v.image.center)
    circles = []
    for pts in groups.values():
        P = np.array(pts)
        if len(P) < 4:
            continue
        fit = fit_circle_3d(P)
        if fit is None:
            continue
        c, n, r = fit
        # Points must actually go around: reject near-collinear sets.
        if r <= 0 or np.ptp((P - c) @ _basis(n)[0]) < 0.5 * r:
            continue
        circles.append((c, n, r, len(P)))
    if not circles:
        return None
    ref = circles[0][1]
    ns = np.array([n if n @ ref >= 0 else -n for _, n, _, _ in circles])
    w = np.array([k for *_, k in circles], dtype=float)
    d = (ns * w[:, None]).sum(axis=0)
    d /= np.linalg.norm(d)
    cs = np.array([c for c, *_ in circles])
    point = cs.mean(axis=0)
    # "Up": cameras look down at the turntable.
    looks = np.array([v.image.R[2] for v in views])        # camera z axis in world
    if np.mean(looks @ d) > 0:
        d = -d
    centers = np.array([v.image.center for v in views])
    rel = centers - point
    h = rel @ d
    radial = np.linalg.norm(rel - np.outer(h, d), axis=1)
    return Axis(point, d, float(radial.max()), float(h.mean()))


def _axis_coords(axis: Axis, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = P - axis.point
    h = rel @ axis.direction
    return h, np.linalg.norm(rel - np.outer(h, axis.direction), axis=1)


# ── step 2: starting guess ────────────────────────────────────────────────────

LOG_STEP = 0.012                # radius grid step (log scale, ~1.2%)
SLACK = 0.10                    # radial slack per angle in the grid search (offset/tilt)


def _band_contrast(v: View, g: np.ndarray, red: float, P_band, P_in, P_out) -> np.ndarray:
    """Band brightness minus the brighter of the two felt samples, NaN where
    any sample is off the photo or behind the camera."""
    vals = []
    for P in (P_band, P_in, P_out):
        uv, z = project(v.camera, v.image, P.reshape(-1, 3))
        val, _ = _sample(g, (uv[:, 0] / v.f - 0.5) * red, (uv[:, 1] / v.f - 0.5) * red)
        val[z <= 0] = np.nan
        vals.append(val.reshape(P.shape[:-1]))
    return vals[0] - np.maximum(vals[1], vals[2])


def initial_guess(views: list[View], axis: Axis, model: Model, spec: RingSpec,
                  log=print):
    """(height along axis, centerline radius, score, coarse photos) of the
    brightest thin band, or None. Each angle may find the band up to SLACK
    away from the candidate radius, so an off-center or tilted ring is still
    found; coarse_fit() then pins it down."""
    half_band = (spec.outer_mm - spec.inner_mm) / (spec.outer_mm + spec.inner_mm)
    R = axis.cam_radius
    pts = model.points
    if len(pts) >= 20:
        h, rad = _axis_coords(axis, pts)
        keep = rad < 0.9 * R
        h, rad = h[keep], rad[keep]
    else:
        h = rad = np.zeros(0)
    if len(h) >= 20:
        z_low = float(np.percentile(h, 2))
        r_obj = float(np.percentile(rad, 95))
        h_lo, h_hi = z_low - 0.10 * R, z_low + 0.05 * R
    else:
        log("[scale] few sparse points; searching a wide range of ring heights")
        r_obj = 0.05 * R
        h_lo, h_hi = axis.cam_height - 1.5 * R, axis.cam_height
    r_lo = max(r_obj * 0.9, 0.03 * R)
    r_hi = 0.97 * R
    if r_hi <= r_lo:
        return None
    radii = np.exp(np.arange(math.log(r_lo), math.log(r_hi), LOG_STEP))
    heights = np.linspace(h_lo, h_hi, int(np.clip(round((h_hi - h_lo) / (0.004 * R)) + 1, 8, 60)))

    imgs = []
    for v in _spread(views, INIT_VIEWS):
        g = _read_gray(v.raw_path, reduce=4)
        if g is None:
            continue
        import cv2
        g = cv2.GaussianBlur(g, (0, 0), 1.5)
        imgs.append((v, g, g.shape[1] / v.raw_size[0]))
    if not imgs:
        return None

    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    e1, e2 = _basis(axis.direction)
    ring_dirs = np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    H, Rr = np.meshgrid(heights, radii, indexing="ij")             # (nh, nr)
    scores = np.zeros(H.shape)
    counts = np.zeros(H.shape)
    wr = int(math.ceil(math.log(1 + SLACK) / LOG_STEP))
    for v, g, red in imgs:
        base = axis.point + H[..., None, None] * axis.direction
        contrast = _band_contrast(
            v, g, red,
            base + Rr[..., None, None] * ring_dirs[None, None],
            base + (Rr * (1 - 3 * half_band))[..., None, None] * ring_dirs[None, None],
            base + (Rr * (1 + 3 * half_band))[..., None, None] * ring_dirs[None, None])
        # best band within +-SLACK of each candidate radius, per angle
        slack = contrast.copy()
        for k in range(1, wr + 1):
            slack[:, k:] = np.fmax(slack[:, k:], contrast[:, :-k])
            slack[:, :-k] = np.fmax(slack[:, :-k], contrast[:, k:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(slack, axis=2)
        good = np.isfinite(med) & (np.isfinite(slack).sum(axis=2) >= 12)
        scores[good] += med[good]
        counts[good] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts >= max(2, len(imgs) // 3), scores / np.maximum(counts, 1), -np.inf)
    i, j = np.unravel_index(np.argmax(mean), mean.shape)
    best = float(mean[i, j])
    if not np.isfinite(best):
        return None
    return float(heights[i]), float(radii[j]), best, imgs


def coarse_fit(imgs, c, n, r_mid, spec: RingSpec, rounds: int = 3):
    """Find the band center along each angle in the coarse photos (searching
    +-20% of the radius), turn those pixels into rays, and fit a free 3D
    circle to them. Returns (center, normal, centerline radius) or None."""
    half_band = (spec.outer_mm - spec.inner_mm) / (spec.outer_mm + spec.inner_mm)
    theta = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    for rnd in range(rounds):
        span = 0.20 if rnd == 0 else 0.08
        rho = r_mid * np.linspace(1 - span, 1 + span, 81)
        _, U = circle_points(c, n, 1.0, theta)                     # (nt, 3)
        P = c + rho[None, :, None] * U[:, None, :]                  # (nt, nr, 3)
        edges = []
        for vi, (v, g, red) in enumerate(imgs):
            con = _band_contrast(
                v, g, red, P,
                c + (rho * (1 - 3 * half_band))[None, :, None] * U[:, None, :],
                c + (rho * (1 + 3 * half_band))[None, :, None] * U[:, None, :])
            filled = np.where(np.isfinite(con), con, -np.inf)
            k = np.argmax(filled, axis=1)
            best = filled[np.arange(len(k)), k]
            ok = (best > 0.8 * MIN_INIT_CONTRAST) & (k > 0) & (k < len(rho) - 1)
            if ok.sum() < 8:
                continue
            Pk = P[np.arange(len(k)), k][ok]
            uv, _ = project(v.camera, v.image, Pk)
            xn = undistort_to_normalized(v.camera, uv)
            rays = np.column_stack([xn, np.ones(len(xn))]) @ v.image.R
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            m = len(rays)
            edges.append(Edges(np.repeat(v.image.center[None], m, axis=0), rays,
                               np.zeros(m, dtype=int), np.full(m, vi), np.zeros((m, 2))))
        if len(edges) < max(3, len(imgs) // 3):
            return None
        E = _concat(edges)
        c, n, r_mid, _, *_ = robust_fit(c, n, r_mid, r_mid * 1.001, E, iters=15, outer=2)
        if not r_mid > 0:
            return None
    return c, n, r_mid


# ── step 3: edges ─────────────────────────────────────────────────────────────

@dataclass
class Edges:
    origins: np.ndarray             # (N,3) camera centers
    dirs: np.ndarray                # (N,3) ray directions (world)
    kind: np.ndarray                # (N,) 0 = inner, 1 = outer
    view: np.ndarray                # (N,) index into the fit's view list
    px: np.ndarray                  # (N,2) raw pixel index coords (for debug)


def find_edges(vi: int, v: View, g: np.ndarray, mask: np.ndarray | None,
               c, n, r_in, r_out) -> Edges | None:
    theta = np.linspace(0, 2 * np.pi, SAMPLES_PER_EDGE, endpoint=False)
    out_px, out_kind = [], []
    P_in, U = circle_points(c, n, r_in, theta)
    P_out, _ = circle_points(c, n, r_out, theta)
    uv_in, z_in = project(v.camera, v.image, P_in)
    uv_out, z_out = project(v.camera, v.image, P_out)
    a_in = uv_in / v.f - 0.5                         # raw index coords
    a_out = uv_out / v.f - 0.5
    band = a_out - a_in                              # outward, across the band
    width = np.linalg.norm(band, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = band / width[:, None]
    ok = (z_in > 0) & (z_out > 0) & (width > 2.0) & np.isfinite(d).all(axis=1)
    for kind, a in ((0, a_in), (1, a_out)):
        win = np.clip(0.35 * width, 1.5, 60.0)
        # samples no more than ~0.75 px apart, whatever the photo size
        n_steps = int(np.clip(np.ceil(2 * np.nanmax(np.where(ok, win, 0)) / 0.75) + 1, 41, 201))
        steps = np.linspace(-1, 1, n_steps)                  # fraction of the window
        tau = win[:, None] * steps[None, :]                  # (M, S) pixels
        x = a[:, 0:1] + tau * d[:, 0:1]
        y = a[:, 1:2] + tau * d[:, 1:2]
        prof, valid = _sample(g, x, y)
        hit_mask = _lookup(mask, x, y).any(axis=1)
        good_line = ok & valid.all(axis=1) & ~hit_mask
        if not good_line.any():
            continue
        dprof = np.gradient(prof, axis=1)
        sign = 1.0 if kind == 0 else -1.0                    # inner: dark→bright outward
        s = sign * dprof
        s[~good_line] = -np.inf
        k = np.argmax(s[:, 1:-1], axis=1) + 1
        rows = np.arange(len(k))
        sk = s[rows, k]
        # Step size across the edge (same polarity), from the profile itself.
        reach = max(4, int(round(4 * (n_steps - 1) / 40)))   # same span as 4 of 41 steps
        lo = np.clip(k - reach, 0, prof.shape[1] - 1)
        hi = np.clip(k + reach, 0, prof.shape[1] - 1)
        step = sign * (prof[rows, hi] - prof[rows, lo])
        accept = good_line & np.isfinite(sk) & (sk > 0) & (step > EDGE_MIN_STEP)
        accept &= (k > 1) & (k < prof.shape[1] - 2)
        # Sub-pixel peak of the derivative.
        sm, s0, sp = s[rows, k - 1], sk, s[rows, k + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            den = sm - 2 * s0 + sp
            off = np.where(np.abs(den) > 1e-12, 0.5 * (sm - sp) / den, 0.0)
        off = np.clip(np.nan_to_num(off), -0.5, 0.5)
        t_star = (steps[k] + off * (steps[1] - steps[0])) * win
        px = a + t_star[:, None] * d
        out_px.append(px[accept])
        out_kind.append(np.full(int(accept.sum()), kind))
    if not out_px:
        return None
    px = np.concatenate(out_px)
    kinds = np.concatenate(out_kind)
    if len(px) < MIN_EDGES_PER_VIEW:
        return None
    uv = (px + 0.5) * v.f                                     # back to COLMAP pixels
    xn = undistort_to_normalized(v.camera, uv)
    if xn is None:
        return None
    R = v.image.R
    rays = np.column_stack([xn, np.ones(len(xn))]) @ R           # = R.T @ [x, y, 1]
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    C = v.image.center
    return Edges(np.repeat(C[None], len(px), axis=0), rays, kinds,
                 np.full(len(px), vi), px)


def _concat(edges: list[Edges]) -> Edges:
    return Edges(*(np.concatenate([getattr(e, f) for e in edges])
                   for f in ("origins", "dirs", "kind", "view", "px")))


def _subset(e: Edges, sel) -> Edges:
    return Edges(e.origins[sel], e.dirs[sel], e.kind[sel], e.view[sel], e.px[sel])


# ── step 4: robust fit ────────────────────────────────────────────────────────

def residuals(params: np.ndarray, n0, e1, e2, E: Edges, jac: bool = False):
    """Edge residuals (distance from the center on the ring plane minus the
    edge's radius). With jac=True also returns the exact derivatives with
    respect to the center (3) and the two radii; the tilt derivatives are
    left to finite differences."""
    c = params[0:3]
    n = n0 + params[3] * e1 + params[4] * e2
    n = n / np.linalg.norm(n)
    radii = params[5:7]
    denom = E.dirs @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = ((c - E.origins) @ n) / denom
        P = E.origins + lam[:, None] * E.dirs
        v = P - c
        rho = np.linalg.norm(v, axis=1)
        res = rho - radii[E.kind]
    bad = ~np.isfinite(res) | (lam <= 0) | (np.abs(denom) < 1e-3)
    res[bad] = np.nan
    if not jac:
        return res
    with np.errstate(divide="ignore", invalid="ignore"):
        u = v / rho[:, None]
        ud = (u * E.dirs).sum(axis=1) / denom
    J = np.zeros((len(res), 7))
    J[:, 0:3] = ud[:, None] * n[None, :] - u
    J[:, 5] = np.where(E.kind == 0, -1.0, 0.0)
    J[:, 6] = np.where(E.kind == 1, -1.0, 0.0)
    J[bad] = 0.0
    return res, J


def robust_fit(c, n, r_in, r_out, E: Edges, iters: int = 25, weights=None, outer: int = 3):
    """Huber-weighted Gauss-Newton over (center, tilt, r_in, r_out).
    Returns (c, n, r_in, r_out, sigma, residuals, weights)."""
    for _ in range(outer):       # re-linearize the normal around the current estimate
        e1, e2 = _basis(n)
        p = np.concatenate([c, [0.0, 0.0], [r_in, r_out]])
        scale = max(r_in, r_out)
        for _ in range(iters):
            res, J = residuals(p, n, e1, e2, E, jac=True)
            ok = np.isfinite(res)
            if ok.sum() < 10:
                break
            sigma = 1.4826 * np.median(np.abs(res[ok] - np.median(res[ok]))) + 1e-12
            if weights is None:
                k = 1.345 * sigma
                a = np.abs(res)
                w = np.where(a <= k, 1.0, k / np.maximum(a, 1e-300))
                w[np.abs(res) > 4 * sigma] = 0.0
            else:
                w = weights.copy()
            w[~ok] = 0.0
            for j in (3, 4):
                h = 1e-6
                dp = p.copy()
                dp[j] += h
                J[:, j] = (residuals(dp, n, e1, e2, E) - res) / h
            J[~ok] = 0.0
            r0 = np.where(ok, res, 0.0)
            sw = np.sqrt(w)
            A = J * sw[:, None]
            b = -r0 * sw
            lam = 1e-9 * np.trace(A.T @ A) + 1e-12
            delta = np.linalg.solve(A.T @ A + lam * np.eye(7), A.T @ b)
            p += delta
            if np.abs(delta[[0, 1, 2, 5, 6]]).max() < 1e-7 * scale and np.abs(delta[3:5]).max() < 1e-8:
                break
        c = p[0:3]
        n = n + p[3] * e1 + p[4] * e2
        n /= np.linalg.norm(n)
        r_in, r_out = float(p[5]), float(p[6])
    res = residuals(np.concatenate([c, [0, 0], [r_in, r_out]]), n, *_basis(n), E)
    ok = np.isfinite(res)
    sigma = 1.4826 * np.median(np.abs(res[ok] - np.median(res[ok]))) if ok.any() else np.inf
    w = np.where(ok & (np.abs(res) <= 3 * sigma), 1.0, 0.0)
    return c, n, r_in, r_out, float(sigma), res, w


# ── main entry ────────────────────────────────────────────────────────────────

def fit_ring(model: Model, raw_root: Path, processed_root: Path | None, spec: RingSpec,
             debug_dir: Path | None = None, log=print, seed: int = 0) -> RingFit:
    err = spec.validate()
    if err:
        return failed(err)
    views = pair_views(model, Path(raw_root), Path(processed_root) if processed_root else None, log)
    if len(views) < MIN_VIEWS:
        return failed(f"only {len(views)} photo(s) in the model matched a raw photo "
                      f"(need {MIN_VIEWS}); check the raw photo folder")
    axis = turntable_axis(views)
    if axis is None:
        return failed("couldn't find the turntable axis from the camera positions "
                      "(need at least 4 turntable steps per camera)")

    guess = initial_guess(views, axis, model, spec, log)
    if guess is None or guess[2] < MIN_INIT_CONTRAST:
        score = "none" if guess is None else f"{guess[2]:.2f}"
        return failed(f"no ring found in the photos (best band contrast {score}, "
                      f"need {MIN_INIT_CONTRAST})")
    h0, r_mid0, score, coarse_imgs = guess
    half_band = (spec.outer_mm - spec.inner_mm) / (spec.outer_mm + spec.inner_mm)
    coarse = coarse_fit(coarse_imgs, axis.point + h0 * axis.direction, axis.direction.copy(),
                        r_mid0, spec)
    if coarse is None:
        return failed("found a bright band but couldn't fit a circle to it in the photos")
    c, n, r_mid0 = coarse
    if n @ axis.direction < 0:
        n = -n
    r_in, r_out = r_mid0 * (1 - half_band), r_mid0 * (1 + half_band)
    log(f"[scale] ring found: band contrast {score:.2f}; refining on full-size photos")

    import cv2
    fit_views = _spread(views, FIT_VIEWS)
    grays, masks = [], []
    for v in fit_views:
        g = _read_gray(v.raw_path)
        if g is not None:
            g = cv2.GaussianBlur(g, (0, 0), 1.0)
        grays.append(g)
        masks.append(_read_mask(v, g.shape) if g is not None else None)

    notes: list[str] = []
    # If rembg kept the ring as foreground, Stage 1's mask covers the ring and
    # would hide every edge. Detect that and fall back to no mask.
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    P_mid, _ = circle_points(c, n, 0.5 * (r_in + r_out), theta)
    covered = []
    for v, g, m in zip(fit_views, grays, masks):
        if g is None or m is None:
            continue
        uv, z = project(v.camera, v.image, P_mid)
        a = uv / v.f - 0.5
        inb = (z > 0) & (a[:, 0] >= 0) & (a[:, 1] >= 0) & (a[:, 0] < g.shape[1]) & (a[:, 1] < g.shape[0])
        if inb.sum() > 20:
            covered.append(_lookup(m, a[inb, 0], a[inb, 1]).mean())
    if covered and np.median(covered) > 0.5:
        notes.append("Stage 1 kept the ring as foreground; fitting without the tablet mask")
        log(f"[scale] note: {notes[-1]}")
        masks = [None] * len(masks)

    sigma = np.inf
    E = None
    for it in range(3):
        all_e = []
        for vi, (v, g, m) in enumerate(zip(fit_views, grays, masks)):
            if g is None:
                continue
            e = find_edges(vi, v, g, m, c, n, r_in, r_out)
            if e is not None:
                all_e.append(e)
        if len(all_e) < MIN_VIEWS:
            return failed(f"ring edges found in only {len(all_e)} photo(s) (need {MIN_VIEWS})",
                          notes=notes)
        E = _concat(all_e)
        c, n, r_in, r_out, sigma, res, w = robust_fit(c, n, r_in, r_out, E)
        if not (0 < r_in < r_out):
            return failed("the fit collapsed (inner radius not below outer radius)", notes=notes)

    keep = w > 0
    E = _subset(E, keep)
    n_in, n_out = int((E.kind == 0).sum()), int((E.kind == 1).sum())
    used_views = np.unique(E.view)
    if n_in < MIN_EDGES_PER_KIND or n_out < MIN_EDGES_PER_KIND:
        return failed(f"too few ring edges after outlier removal (inner {n_in}, outer {n_out}; "
                      f"need {MIN_EDGES_PER_KIND} each)", notes=notes)

    if n @ axis.direction < 0:
        n = -n
    r_mid = 0.5 * (r_in + r_out)
    s = spec.mid_radius_mm / r_mid
    s_in = (spec.inner_mm / 2) / r_in
    s_out = (spec.outer_mm / 2) / r_out

    # Per camera: radii with the plane held fixed.
    per_cam = {}
    res = residuals(np.concatenate([c, [0, 0], [r_in, r_out]]), n, *_basis(n), E)
    for gname in sorted({fit_views[i].group for i in used_views}):
        sel = np.isin(E.view, [i for i in used_views if fit_views[i].group == gname])
        ri = r_in + np.nanmedian(res[sel & (E.kind == 0)]) if (sel & (E.kind == 0)).sum() > 20 else None
        ro = r_out + np.nanmedian(res[sel & (E.kind == 1)]) if (sel & (E.kind == 1)).sum() > 20 else None
        if ri is not None and ro is not None:
            per_cam[gname] = spec.mid_radius_mm / (0.5 * (ri + ro))

    # Bootstrap over photos.
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(30):
        pick = rng.choice(used_views, size=len(used_views), replace=True)
        idx = np.concatenate([np.where(E.view == vi)[0] for vi in pick])
        Eb = _subset(E, idx)
        cb, nb, rib, rob, *_ = robust_fit(c, n, r_in, r_out, Eb, iters=6, outer=1)
        if 0 < rib < rob:
            boots.append(spec.mid_radius_mm / (0.5 * (rib + rob)))
    sigma_pct = float(np.std(boots) / s * 100) if len(boots) >= 10 else float("inf")

    h_c = (c - axis.point) @ axis.direction
    off_axis = float(np.linalg.norm((c - axis.point) - h_c * axis.direction)) * s
    tilt = float(math.degrees(math.acos(min(1.0, abs(float(n @ axis.direction))))))
    width_mm = (r_out - r_in) * s

    fit = RingFit(
        ok=True, scale=float(s), sigma_pct=sigma_pct, r_in=float(r_in), r_out=float(r_out),
        center=c.tolist(), normal=n.tolist(), scale_inner=float(s_in), scale_outer=float(s_out),
        width_mm=float(width_mm), per_camera={k: float(v) for k, v in per_cam.items()},
        off_axis_mm=off_axis, tilt_deg=tilt, residual_mm=float(sigma * s),
        n_views=int(len(used_views)), n_edges_inner=n_in, n_edges_outer=n_out, notes=notes)

    problems = []
    if sigma_pct > FAIL_SIGMA_PCT:
        problems.append(f"scale uncertainty {sigma_pct:.2f}% is above {FAIL_SIGMA_PCT}%")
    io = abs(s_in - s_out) / s * 100
    if io > FAIL_INNER_OUTER_PCT:
        problems.append(f"inner and outer edges disagree by {io:.2f}% "
                        f"(over {FAIL_INNER_OUTER_PCT}%); check the ring diameters entered")
    if len(per_cam) >= 2:
        spread = (max(per_cam.values()) - min(per_cam.values())) / s * 100
        if spread > FAIL_GROUP_SPREAD_PCT:
            problems.append(f"cameras disagree by {spread:.2f}% (over {FAIL_GROUP_SPREAD_PCT}%)")
    exp_w = spec.expected_width_mm
    wdiff = abs(width_mm - exp_w) / exp_w * 100
    if wdiff > FAIL_WIDTH_PCT:
        problems.append(f"fitted band width {width_mm:.2f} mm vs expected {exp_w:.2f} mm "
                        f"({wdiff:.0f}% off); the edges found may not be the ring's")
    if off_axis > WARN_OFF_AXIS_MM:
        fit.warnings.append(f"ring center is {off_axis:.1f} mm from the turntable axis")
    if tilt > WARN_TILT_DEG:
        fit.warnings.append(f"ring is tilted {tilt:.1f} deg from the turntable plane")
    if problems:
        fit.ok = False
        fit.reason = "; ".join(problems)

    if debug_dir is not None:
        try:
            save_debug(Path(debug_dir), fit_views, grays, E, c, n, r_in, r_out)
        except Exception as exc:     # debug images must never break the fit
            log(f"[scale] couldn't write debug images: {exc}")
    return fit


def save_debug(out: Path, views: list[View], grays, E: Edges, c, n, r_in, r_out) -> None:
    """Downscaled photos with the fitted ring (green inner, red outer) and the
    accepted edge points (yellow)."""
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    used = [vi for vi in np.unique(E.view)]
    pick = [used[i] for i in np.linspace(0, len(used) - 1, min(DEBUG_VIEWS, len(used))).round().astype(int)]
    for vi in sorted(set(pick)):
        v, g = views[vi], grays[vi]
        if g is None:
            continue
        k = min(1.0, 1600 / max(g.shape))
        img = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        for r, col in ((r_in, (0, 200, 0)), (r_out, (0, 0, 230))):
            P, _ = circle_points(c, n, r, theta)
            uv, z = project(v.camera, v.image, P)
            a = ((uv / v.f - 0.5) * k)[z > 0]
            cv2.polylines(img, [np.round(a).astype(np.int32).reshape(-1, 1, 2)], True, col, 2,
                          cv2.LINE_AA)
        for p in E.px[E.view == vi][::3]:
            cv2.circle(img, (int(round(p[0] * k)), int(round(p[1] * k))), 2, (0, 220, 255), -1)
        name = v.image.name.replace("/", "_").rsplit(".", 1)[0]
        cv2.imwrite(str(out / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
