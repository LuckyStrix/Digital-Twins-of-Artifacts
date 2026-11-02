"""Synthetic turntable scans of the reference ring, for the scale tests.

Renders what the rig would photograph: a white ring (top face only) on black
felt, a clay-grey tablet in the middle, and a white backdrop beyond the
turntable. Writes raw photos, Stage 1 style processed photos (tablet alpha),
and a COLMAP TXT model whose frame is a random similarity of the true
millimeter frame, so the expected answer is known exactly:
mm per model unit = 1 / model_scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

RING_IN = 100.0            # radius, mm (200 mm inner diameter)
RING_OUT = 108.0           # radius, mm (216 mm outer diameter)
TURNTABLE_R = 150.0
FELT, RING, BACKDROP, CLAY = 0.08, 0.92, 0.95, 0.55


@dataclass
class Scene:
    raw_w: int = 640
    raw_h: int = 426
    focal: float = 560.0              # raw pixels
    image_scale: float = 1.0          # COLMAP camera size / raw size
    distortion: float = 0.0           # SIMPLE_RADIAL k (0 = SIMPLE_PINHOLE)
    elevations: tuple = (20.0, 35.0, 55.0)
    distance: float = 420.0
    steps: int = 12
    ring_offset: tuple = (3.0, -2.0)  # ring center in the turntable plane, mm
    ring_tilt_deg: float = 0.5
    ring_present: bool = True
    tablet: tuple = (40.0, 28.0, 15.0)
    model_scale: float = 0.0137       # model units per mm
    seed: int = 1
    raw_ext: str = ".JPG"
    exif_orientation: int = 0         # written to the raw JPGs; must be ignored
    stride: int = 1                   # only every Nth photo goes into the model
    noise: float = 0.01
    with_rigs_frames: bool = False    # also write COLMAP 4.x rigs.txt / frames.txt
    mask_includes_ring: bool = False  # Stage 1 kept the ring as foreground
    meta: dict = field(default_factory=dict)

    @property
    def mm_per_unit(self) -> float:
        return 1.0 / self.model_scale


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def _look_at(C, target):
    f = target - C
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.vstack([x, y, f])
    return R, -R @ C


def _rotmat_to_qvec(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    # 180-degree rotations: pick the largest diagonal
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def render(scene: Scene, root: Path) -> dict:
    """Write raw/, processed/ and model/ under ``root``; return paths + truth."""
    rng = np.random.default_rng(scene.seed)
    raw_dir, proc_dir, model_dir = root / "raw", root / "processed", root / "model"
    for d in (raw_dir, proc_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    W, H = scene.raw_w, scene.raw_h
    # Camera rays for every sub-pixel (2x2 supersampling), in camera coords.
    ss = np.array([-0.25, 0.25])
    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    sub = []
    for dy in ss:
        for dx in ss:
            sub.append(np.column_stack([(ii + 0.5 + dx).ravel(), (jj + 0.5 + dy).ravel()]))
    pix = np.concatenate(sub)                                   # raw pixel coords (corner convention)
    K = np.array([[scene.focal, 0, W / 2], [0, scene.focal, H / 2], [0, 0, 1.0]])
    if scene.distortion:
        und = cv2.undistortPoints(pix.reshape(-1, 1, 2).astype(np.float64), K,
                                  np.array([scene.distortion, 0, 0, 0.0])).reshape(-1, 2)
    else:
        und = (pix - K[:2, 2]) / scene.focal
    rays_cam = np.column_stack([und, np.ones(len(und))])

    ring_n = _rot([1, 0.3, 0], scene.ring_tilt_deg) @ np.array([0, 0, 1.0])
    ring_c = np.array([scene.ring_offset[0], scene.ring_offset[1], 0.0])
    a, b, cz = scene.tablet
    tab_c = np.array([5.0, -3.0, cz - 1.0])                     # resting on the felt (z = -1)

    # Model frame = random similarity of the mm frame.
    Rg = _rot(rng.normal(size=3), rng.uniform(20, 160))
    tg = rng.normal(size=3) * 2.0
    s0 = scene.model_scale

    images, k = [], 0
    target = np.array([0, 0, 10.0])
    for gi, elev in enumerate(scene.elevations):
        for j in range(scene.steps):
            ang = math.radians(-360.0 * j / scene.steps + 7 * gi)
            e = math.radians(elev)
            C = np.array([scene.distance * math.cos(e) * math.cos(ang),
                          scene.distance * math.cos(e) * math.sin(ang),
                          scene.distance * math.sin(e)])
            R, t = _look_at(C, target)
            rays = rays_cam @ R                                  # world directions
            O = C
            n_rays = len(rays)
            val = np.full(n_rays, BACKDROP)
            hit_r = np.zeros(n_rays, dtype=bool)
            # felt / backdrop plane z = -1
            lam_f = (-1.0 - O[2]) / rays[:, 2]
            Pf = O + lam_f[:, None] * rays
            on_tt = np.hypot(Pf[:, 0], Pf[:, 1]) < TURNTABLE_R
            val = np.where((lam_f > 0) & on_tt, FELT, val)
            # ring top face
            if scene.ring_present:
                lam_r = ((ring_c - O) @ ring_n) / (rays @ ring_n)
                Pr = O + lam_r[:, None] * rays
                rr = np.linalg.norm(Pr - ring_c, axis=1)
                hit_r = (lam_r > 0) & (rr >= RING_IN) & (rr <= RING_OUT)
                val = np.where(hit_r, RING, val)
            # tablet (ellipsoid), in front of everything
            q = (O - tab_c) / np.array([a, b, cz])
            dq = rays / np.array([a, b, cz])
            A = (dq * dq).sum(1)
            B = 2 * (q * dq).sum(1)
            Cq = (q * q).sum() - 1
            disc = B * B - 4 * A * Cq
            hit_t = disc > 0
            val = np.where(hit_t, CLAY, val)
            img = val.reshape(4, H, W).mean(axis=0)
            fg = hit_t | hit_r if scene.mask_includes_ring else hit_t
            alpha = fg.reshape(4, H, W).mean(axis=0) > 0.5
            img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.7)
            img = np.clip(img + rng.normal(scale=scene.noise, size=img.shape), 0, 1)
            g8 = (img * 255).round().astype(np.uint8)

            cam = f"cam{gi + 1}"
            name = f"IMG_{k:04d}"
            k += 1
            (raw_dir / cam).mkdir(exist_ok=True)
            (proc_dir / cam).mkdir(exist_ok=True)
            pil = PILImage.fromarray(g8).convert("RGB")
            raw_path = raw_dir / cam / f"{name}{scene.raw_ext}"
            if scene.exif_orientation and scene.raw_ext.lower() in (".jpg", ".jpeg"):
                exif = PILImage.Exif()
                exif[0x0112] = scene.exif_orientation
                pil.save(raw_path, quality=95, exif=exif)
            else:
                pil.save(raw_path, quality=95)
            rgba = np.dstack([np.repeat(g8[..., None], 3, axis=2), (alpha * 255).astype(np.uint8)])
            PILImage.fromarray(rgba, "RGBA").save(proc_dir / cam / f"{name}.png")

            # pose in the model frame
            Rm = R @ Rg.T
            tm = s0 * t - Rm @ tg
            images.append((cam, name, Rm, tm))

    # COLMAP model (only every `stride`-th photo, like Stage 1's stride)
    cw, ch = int(round(W * scene.image_scale)), int(round(H * scene.image_scale))
    sx = cw / W
    f_c, cx, cy = scene.focal * sx, W / 2 * sx, H / 2 * sx
    with (model_dir / "cameras.txt").open("w") as fh:
        fh.write("# Camera list\n")
        for gi in range(len(scene.elevations)):
            if scene.distortion:
                fh.write(f"{gi + 1} SIMPLE_RADIAL {cw} {ch} {f_c} {cx} {cy} {scene.distortion}\n")
            else:
                fh.write(f"{gi + 1} SIMPLE_PINHOLE {cw} {ch} {f_c} {cx} {cy}\n")
    with (model_dir / "images.txt").open("w") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        iid = 0
        for idx, (cam, name, Rm, tm) in enumerate(images):
            if idx % scene.stride:
                continue
            iid += 1
            q = _rotmat_to_qvec(Rm)
            fh.write(f"{iid} {' '.join(repr(float(v)) for v in q)} "
                     f"{' '.join(repr(float(v)) for v in tm)} {int(cam[3:])} {cam}/{name}.png\n")
            fh.write("1.5 2.5 1 3.5 4.5 -1\n" if iid % 2 else "\n")
    # tablet surface points (upper part), in the model frame
    u = rng.uniform(0, 2 * np.pi, 400)
    v = rng.uniform(0.15, 1.0, 400)
    pts = tab_c + np.column_stack([a * np.sqrt(1 - v * v) * np.cos(u),
                                   b * np.sqrt(1 - v * v) * np.sin(u), cz * v])
    pts_m = s0 * pts @ Rg.T + tg
    with (model_dir / "points3D.txt").open("w") as fh:
        fh.write("# 3D point list\n")
        for i, p in enumerate(pts_m):
            fh.write(f"{i + 1} {float(p[0])!r} {float(p[1])!r} {float(p[2])!r} 128 128 128 0.5 1 0 2 1\n")
    if scene.with_rigs_frames:
        with (model_dir / "rigs.txt").open("w") as fh:
            fh.write("# Rig calib list\n1 2 CAMERA 1 CAMERA 2 1 1 0 0 0 0.5 -0.25 2.0\n2 1 CAMERA 3\n")
        with (model_dir / "frames.txt").open("w") as fh:
            fh.write("# Frame list\n")
            for i, (_, _, Rm, tm) in enumerate(images[:3]):
                q = _rotmat_to_qvec(Rm)
                fh.write(f"{i + 1} 1 {' '.join(repr(float(x)) for x in q)} "
                         f"{' '.join(repr(float(x)) for x in tm)} 1 CAMERA 1 {i + 1}\n")

    return {
        "raw": raw_dir, "processed": proc_dir, "model": model_dir,
        "mm_per_unit": scene.mm_per_unit, "Rg": Rg, "tg": tg, "s0": s0,
        "ring_center_mm": ring_c, "ring_normal_mm": ring_n,
    }
