"""scale/colmap_model.py: reading and rescaling COLMAP models, projection,
and rescaling a finished workspace's files."""

import json
import shutil
import struct

import cv2
import numpy as np
import pytest

from scale.colmap_model import (
    Camera, Image, camera_matrix_and_dist, project, read_model_txt, rescale_camera_centers,
    rescale_images_bin, rescale_model_txt, rescale_ply, rescale_points3d_bin,
    undistort_to_normalized,
)
from scale_synth import Scene, render


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    info = render(Scene(raw_w=96, raw_h=64, focal=84.0, steps=4, with_rigs_frames=True), root)
    return info["model"]


def test_read_model(tiny_model):
    m = read_model_txt(tiny_model)
    assert len(m.cameras) == 3 and len(m.images) == 12 and len(m.points) == 400
    assert all(i.name.startswith("cam") and i.name.endswith(".png") for i in m.images.values())


def test_rescale_model_txt_scales_lengths_only(tiny_model, tmp_path):
    d = tmp_path / "m"
    shutil.copytree(tiny_model, d)
    before = read_model_txt(d)
    frames0 = (d / "frames.txt").read_text().splitlines()
    rescale_model_txt(d, 2.5)
    after = read_model_txt(d)
    for iid, a in before.images.items():
        b = after.images[iid]
        assert np.allclose(b.center, 2.5 * a.center)
        assert np.allclose(b.qvec, a.qvec)
        assert b.name == a.name
    assert np.allclose(after.points, 2.5 * before.points)
    # COLMAP 4.x: rig_from_world translation (fields 6-8) and sensor_from_rig translation
    f0 = [l.split() for l in frames0 if not l.startswith("#")][0]
    f1 = [l.split() for l in (d / "frames.txt").read_text().splitlines() if not l.startswith("#")][0]
    assert f1[2:6] == f0[2:6]
    assert np.allclose([float(v) for v in f1[6:9]], [2.5 * float(v) for v in f0[6:9]])
    assert f1[9:] == f0[9:]
    rig = (d / "rigs.txt").read_text().splitlines()[1].split()
    assert [float(v) for v in rig[-3:]] == [1.25, -0.625, 5.0]      # 0.5, -0.25, 2.0 x 2.5
    assert rig[:-3] == "1 2 CAMERA 1 CAMERA 2 1 1 0 0 0".split()
    # points2D lines (including empty ones) are untouched
    lines = (d / "images.txt").read_text().split("\n")
    assert "1.5 2.5 1 3.5 4.5 -1" in lines and "" in lines


@pytest.mark.parametrize("model,params", [
    ("SIMPLE_PINHOLE", [800, 320, 213]),
    ("SIMPLE_RADIAL", [800, 320, 213, -0.08]),
    ("RADIAL", [800, 320, 213, -0.08, 0.01]),
    ("OPENCV", [800, 790, 320, 213, -0.1, 0.02, 0.001, -0.002]),
    ("FULL_OPENCV", [800, 790, 320, 213, -0.1, 0.02, 0.001, -0.002, 0.01, 0.02, 0.003, 0.001]),
])
def test_project_matches_opencv_and_undistort_inverts_it(model, params):
    rng = np.random.default_rng(3)
    cam = Camera(1, model, 640, 426, params)
    q = rng.normal(size=4)
    img = Image(1, q / np.linalg.norm(q), rng.normal(size=3), 1, "a")
    X = rng.normal(size=(100, 3)) * 0.3 + img.center + img.R[2] * 3
    uv, z = project(cam, img, X)
    K, d = camera_matrix_and_dist(cam)
    ref, _ = cv2.projectPoints(X.reshape(-1, 1, 3), cv2.Rodrigues(img.R)[0], img.tvec, K, d)
    assert np.allclose(uv, ref.reshape(-1, 2), atol=1e-9) and (z > 0).all()
    xn = undistort_to_normalized(cam, uv)
    Xc = X @ img.R.T + img.tvec
    assert np.allclose(xn, Xc[:, :2] / Xc[:, 2:], atol=1e-7)


def test_unsupported_camera_model():
    assert camera_matrix_and_dist(Camera(1, "FISHEYE", 10, 10, [1, 2, 3, 4])) is None


def _confidence_module():
    pytest.importorskip("open3d")
    import importlib.util
    from conftest import SCRIPT_DIR
    spec = importlib.util.spec_from_file_location("confidence", SCRIPT_DIR / "alignment" / "confidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_binary_ply(path, xyz, extra=True):
    n = len(xyz)
    props = ["float x", "float y", "float z"] + (["float nx", "uchar red"] if extra else [])
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" + \
        "".join(f"property {p}\n" for p in props) + "end_header\n"
    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("nx", "<f4"), ("red", "u1")] if extra else [])
    arr = np.zeros(n, dtype=dt)
    arr["x"], arr["y"], arr["z"] = xyz.T
    if extra:
        arr["nx"] = 0.25
        arr["red"] = np.arange(n) % 256
    path.write_bytes(header.encode() + arr.tobytes())
    return arr


def test_rescale_ply_binary_keeps_order_and_other_properties(tmp_path):
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    p = tmp_path / "a.ply"
    arr = _write_binary_ply(p, xyz)
    rescale_ply(p, 0.5)
    raw = p.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    out = np.frombuffer(raw, dtype=arr.dtype, offset=end)
    assert np.allclose(np.column_stack([out["x"], out["y"], out["z"]]), xyz * 0.5)
    assert (out["nx"] == 0.25).all() and (out["red"] == arr["red"]).all()


def test_rescale_ply_ascii(tmp_path):
    p = tmp_path / "a.ply"
    p.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nend_header\n1 2 3 7\n4 5 6 8\n")
    rescale_ply(p, 2.0)
    body = p.read_text().split("end_header\n")[1].split("\n")
    assert [float(v) for v in body[0].split()] == [2, 4, 6, 7]


def test_rescale_images_and_points_bin(tmp_path):
    imgs = tmp_path / "images.bin"
    buf = struct.pack("<Q", 2)
    for iid, t in ((1, (1.0, 2.0, 3.0)), (2, (-4.0, 0.5, 8.0))):
        buf += struct.pack("<I", iid) + struct.pack("<4d", 1, 0, 0, 0) + struct.pack("<3d", *t)
        buf += struct.pack("<I", 1) + f"cam1/img{iid}.png".encode() + b"\x00"
        buf += struct.pack("<Q", 2) + struct.pack("<ddq", 1, 2, -1) * 2
    imgs.write_bytes(buf)
    rescale_images_bin(imgs, 3.0)
    ids, centers = _confidence_module().read_images_bin(str(imgs))   # Stage 3's reader
    assert np.allclose(centers[0], [-3.0, -6.0, -9.0])
    assert np.allclose(centers[1], [12.0, -1.5, -24.0])

    pts = tmp_path / "points3D.bin"
    buf = struct.pack("<Q", 1) + struct.pack("<Q", 5) + struct.pack("<3d", 1, 2, 3)
    buf += bytes([1, 2, 3]) + struct.pack("<d", 0.5) + struct.pack("<Q", 2) + struct.pack("<II", 1, 0) * 2
    pts.write_bytes(buf)
    rescale_points3d_bin(pts, 2.0)
    assert struct.unpack_from("<3d", pts.read_bytes(), 16) == (2.0, 4.0, 6.0)


def test_rescale_camera_centers(tmp_path):
    p = tmp_path / "camera_centers.json"
    p.write_text(json.dumps([[1, 2, 3], [4, 5, 6]]))
    rescale_camera_centers(p, 10)
    assert json.loads(p.read_text()) == [[10, 20, 30], [40, 50, 60]]
