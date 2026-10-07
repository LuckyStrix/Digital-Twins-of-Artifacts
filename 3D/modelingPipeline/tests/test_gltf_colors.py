"""glTF COLOR_0 must hold linear light; Open3D's vertex colours are sRGB photo pixels."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")

SRC = Path(__file__).resolve().parent.parent / "src" / "reconstruct_mesh.py"


@pytest.fixture(scope="module")
def rm():
    spec = importlib.util.spec_from_file_location("reconstruct_mesh", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("reconstruct_mesh", mod)
    spec.loader.exec_module(mod)
    return mod


def _triangle(srgb):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.eye(3))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([srgb] * 3, dtype=np.float64))
    return mesh


def _color0(doc, buf):
    acc = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"]]
    view = doc["bufferViews"][acc["bufferView"]]
    dt = np.uint8 if acc["componentType"] == 5121 else np.float32
    arr = np.frombuffer(bytes(buf), dtype=dt, count=acc["count"] * 4,
                        offset=view.get("byteOffset", 0) + acc.get("byteOffset", 0))
    arr = arr.reshape(-1, 4).astype(np.float64)
    return arr / 255.0 if acc.get("normalized") else arr


def test_srgb_to_linear_known_values(rm):
    got = rm.srgb_to_linear(np.array([0.0, 0.04045, 0.5, 1.0]))
    np.testing.assert_allclose(got, [0.0, 0.04045 / 12.92, 0.21404, 1.0], atol=1e-5)


@pytest.mark.parametrize("quantize", [False, True])
def test_color0_is_linear(rm, quantize):
    srgb = [79 / 255, 58 / 255, 34 / 255]          # a dark-brown photo pixel
    doc, buf = rm.build_gltf_document(_triangle(srgb), quantize_colors=quantize)
    col = _color0(doc, buf)
    tol = 1 / 255 if quantize else 1e-6
    np.testing.assert_allclose(col[:, :3], np.tile(rm.srgb_to_linear(srgb), (3, 1)), atol=tol)
    np.testing.assert_allclose(col[:, 3], 1.0)
