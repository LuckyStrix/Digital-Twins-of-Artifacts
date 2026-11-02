"""COLMAP model helpers for the scale step: read TXT models, rescale them,
undistort pixels, and rescale a finished workspace's outputs.

Rescaling a reconstruction by ``s`` multiplies every length: camera
translations (``t`` in cam_from_world, ``rig_from_world`` and
``sensor_from_rig``) and 3D point positions. Rotations and intrinsics don't
change. This is done on the TXT export directly instead of with
``colmap model_transformer``, whose transform-file format has changed
between COLMAP versions. COLMAP 4.x adds rigs.txt and frames.txt (poses live
in frames); older versions have neither, and both are handled.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ── reading ───────────────────────────────────────────────────────────────────

@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass
class Image:
    image_id: int
    qvec: np.ndarray          # w, x, y, z (cam_from_world)
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def R(self) -> np.ndarray:
        return qvec_to_rotmat(self.qvec)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.tvec


@dataclass
class Model:
    cameras: dict[int, Camera] = field(default_factory=dict)
    images: dict[int, Image] = field(default_factory=dict)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


def qvec_to_rotmat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _data_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                yield s


def read_cameras_txt(path: Path) -> dict[int, Camera]:
    cams = {}
    for line in _data_lines(path):
        p = line.split()
        cams[int(p[0])] = Camera(int(p[0]), p[1], int(p[2]), int(p[3]),
                                 [float(v) for v in p[4:]])
    return cams


def read_images_txt(path: Path) -> dict[int, Image]:
    """images.txt: two lines per image; the second (points2D) may be empty,
    so walk the raw lines instead of the non-empty ones."""
    images = {}
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        p = lines[i].split()
        if len(p) < 10:
            i += 1
            continue
        images[int(p[0])] = Image(
            int(p[0]),
            np.array([float(v) for v in p[1:5]]),
            np.array([float(v) for v in p[5:8]]),
            int(p[8]),
            " ".join(p[9:]),
        )
        i += 2
    return images


def read_points3d_txt(path: Path) -> np.ndarray:
    pts = [[float(v) for v in line.split()[1:4]] for line in _data_lines(path)]
    return np.array(pts, dtype=np.float64).reshape(-1, 3)


def read_model_txt(model_dir: Path) -> Model:
    model_dir = Path(model_dir)
    pts_path = model_dir / "points3D.txt"
    return Model(
        cameras=read_cameras_txt(model_dir / "cameras.txt"),
        images=read_images_txt(model_dir / "images.txt"),
        points=read_points3d_txt(pts_path) if pts_path.exists() else np.zeros((0, 3)),
    )


# ── rescaling a TXT model ─────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return repr(float(v))


def _scale_fields(parts: list[str], idxs, s: float) -> None:
    for k in idxs:
        parts[k] = _fmt(float(parts[k]) * s)


def rescale_model_txt(model_dir: Path, s: float) -> None:
    """Multiply every length in the TXT model in ``model_dir`` by ``s``, in place."""
    model_dir = Path(model_dir)

    # images.txt: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME / POINTS2D line.
    path = model_dir / "images.txt"
    out, header_next = [], True
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                out.append(line)
                continue
            if header_next:
                parts = line.split()
                if len(parts) >= 10:
                    _scale_fields(parts, (5, 6, 7), s)
                    line = " ".join(parts) + "\n"
                    header_next = False
                out.append(line)
            else:
                out.append(line)
                header_next = True
    path.write_text("".join(out), encoding="utf-8")

    # points3D.txt: POINT3D_ID X Y Z R G B ERROR TRACK...
    path = model_dir / "points3D.txt"
    if path.exists():
        _rewrite_data_lines(path, lambda p: _scale_fields(p, (1, 2, 3), s))

    # frames.txt (COLMAP 4.x): FRAME_ID RIG_ID QW QX QY QZ TX TY TZ ...
    path = model_dir / "frames.txt"
    if path.exists():
        _rewrite_data_lines(path, lambda p: _scale_fields(p, (6, 7, 8), s))

    # rigs.txt (COLMAP 4.x): RIG_ID NUM_SENSORS [REF_TYPE REF_ID]
    #   then per extra sensor: TYPE ID HAS_POSE [QW QX QY QZ TX TY TZ]
    path = model_dir / "rigs.txt"
    if path.exists():
        def rig(p):
            n = int(p[1])
            k = 4 if n > 0 else 2
            for _ in range(max(0, n - 1)):
                has_pose = p[k + 2] == "1"
                k += 3
                if has_pose:
                    _scale_fields(p, (k + 4, k + 5, k + 6), s)
                    k += 7
        _rewrite_data_lines(path, rig)


def _rewrite_data_lines(path: Path, fn) -> None:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                parts = s.split()
                fn(parts)
                line = " ".join(parts) + "\n"
            out.append(line)
    path.write_text("".join(out), encoding="utf-8")


# ── pixels → rays ─────────────────────────────────────────────────────────────

# COLMAP camera model → (fx, fy, cx, cy, OpenCV distortion) from params.
def camera_matrix_and_dist(cam: Camera):
    """K and OpenCV distortion coefficients for ``cam``, or None if the model
    isn't supported. K uses COLMAP's pixel convention (the top-left pixel's
    center is at (0.5, 0.5)); pass pixel coordinates in that convention."""
    p, m = cam.params, cam.model
    if m == "SIMPLE_PINHOLE":
        f, cx, cy = p
        fx = fy = f
        dist = np.zeros(4)
    elif m == "PINHOLE":
        fx, fy, cx, cy = p
        dist = np.zeros(4)
    elif m == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        fx = fy = f
        dist = np.array([k, 0.0, 0.0, 0.0])
    elif m == "RADIAL":
        f, cx, cy, k1, k2 = p
        fx = fy = f
        dist = np.array([k1, k2, 0.0, 0.0])
    elif m == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p
        dist = np.array([k1, k2, p1, p2])
    elif m == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = p
        dist = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
    else:
        return None
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K, dist.astype(np.float64)


def undistort_to_normalized(cam: Camera, uv: np.ndarray) -> np.ndarray | None:
    """Pixel coords (N,2, COLMAP convention) → normalized image coords (N,2)."""
    import cv2

    kd = camera_matrix_and_dist(cam)
    if kd is None:
        return None
    K, dist = kd
    if len(uv) == 0:
        return np.zeros((0, 2))
    pts = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    if not np.any(dist):
        out = (pts.reshape(-1, 2) - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
        return out
    und = cv2.undistortPoints(pts, K, dist,
                              criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10))
    return und.reshape(-1, 2)


def project(cam: Camera, img: Image, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N,3) → pixel coords (N,2, COLMAP convention) and depth (N,).
    Same distortion model as cv2.projectPoints, in numpy (much faster here)."""
    K, dist = camera_matrix_and_dist(cam)
    Xc = np.asarray(X, dtype=np.float64) @ img.R.T + img.tvec
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = Xc[:, 0] / z
        y = Xc[:, 1] / z
    if np.any(dist):
        d = np.zeros(8)
        d[:len(dist)] = dist
        k1, k2, p1, p2, k3, k4, k5, k6 = d
        r2 = x * x + y * y
        radial = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = xd, yd
    uv = np.column_stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]])
    return uv, z


# ── rescaling a finished workspace (dense outputs) ────────────────────────────

_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def rescale_ply(path: Path, s: float) -> None:
    """Multiply x/y/z of every vertex in a binary or ASCII PLY by ``s``, in place.
    Every other property and the vertex order are kept, so COLMAP's .vis
    files (which index points by order) stay valid."""
    path = Path(path)
    raw = path.read_bytes()
    end = raw.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: not a PLY file")
    end = raw.index(b"\n", end) + 1
    header = raw[:end].decode("ascii", errors="replace").splitlines()
    fmt = next(l.split()[1] for l in header if l.startswith("format"))
    elements, cur = [], None
    for line in header:
        p = line.split()
        if not p:
            continue
        if p[0] == "element":
            cur = [p[1], int(p[2]), []]
            elements.append(cur)
        elif p[0] == "property" and cur is not None:
            if p[1] == "list":
                if cur[0] == "vertex":
                    raise ValueError(f"{path}: list properties on vertices aren't supported")
                cur[2].append((p[-1], None))
            else:
                cur[2].append((p[-1], _PLY_TYPES[p[1]]))
    if not elements or elements[0][0] != "vertex":
        raise ValueError(f"{path}: first element isn't 'vertex'")
    _, n, props = elements[0]
    if fmt == "ascii":
        body = raw[end:].decode("ascii").splitlines()
        names = [nm for nm, _ in props]
        ix = [names.index(c) for c in ("x", "y", "z")]
        for i in range(n):
            parts = body[i].split()
            for k in ix:
                parts[k] = _fmt(float(parts[k]) * s)
            body[i] = " ".join(parts)
        path.write_bytes(raw[:end] + ("\n".join(body) + "\n").encode("ascii"))
        return
    order = "<" if fmt == "binary_little_endian" else ">"
    dtype = np.dtype([(nm, order + t) for nm, t in props])
    verts = np.frombuffer(raw, dtype=dtype, count=n, offset=end).copy()
    for c in ("x", "y", "z"):
        verts[c] = verts[c] * s
    rest = raw[end + n * dtype.itemsize:]
    path.write_bytes(raw[:end] + verts.tobytes() + rest)


def rescale_images_bin(path: Path, s: float) -> None:
    """Multiply the translation of every image in a COLMAP images.bin by ``s``."""
    buf = bytearray(Path(path).read_bytes())
    (n,) = struct.unpack_from("<Q", buf, 0)
    off = 8
    for _ in range(n):
        off += 4 + 32                                  # image_id, qvec
        t = struct.unpack_from("<3d", buf, off)
        struct.pack_into("<3d", buf, off, *(v * s for v in t))
        off += 24 + 4                                  # tvec, camera_id
        off = buf.index(b"\x00", off) + 1              # name
        (n2d,) = struct.unpack_from("<Q", buf, off)
        off += 8 + n2d * 24
    Path(path).write_bytes(bytes(buf))


def rescale_points3d_bin(path: Path, s: float) -> None:
    buf = bytearray(Path(path).read_bytes())
    (n,) = struct.unpack_from("<Q", buf, 0)
    off = 8
    for _ in range(n):
        off += 8                                       # point3D_id
        xyz = struct.unpack_from("<3d", buf, off)
        struct.pack_into("<3d", buf, off, *(v * s for v in xyz))
        off += 24 + 3 + 8                              # xyz, rgb, error
        (tl,) = struct.unpack_from("<Q", buf, off)
        off += 8 + tl * 8
    Path(path).write_bytes(bytes(buf))


def rescale_camera_centers(path: Path, s: float) -> None:
    path = Path(path)
    centers = json.loads(path.read_text())
    path.write_text(json.dumps([[v * s for v in c] for c in centers]))
