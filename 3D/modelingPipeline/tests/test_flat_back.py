"""flat_back closes a one-sided scan with a flat cap and keeps the scanned surface."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("pymeshlab")
pytest.importorskip("mapbox_earcut")

SRC = Path(__file__).resolve().parent.parent / "src" / "reconstruct_mesh.py"


@pytest.fixture(scope="module")
def rm():
    spec = importlib.util.spec_from_file_location("reconstruct_mesh", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("reconstruct_mesh", mod)
    spec.loader.exec_module(mod)
    return mod


def _one_sided_scan():
    """A closed ellipsoid (stand-in for the Poisson surface) whose points only cover
    the top, with a wavy rim, like a single turntable side."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=80)
    mesh.scale(1.0, center=(0, 0, 0))
    V = np.asarray(mesh.vertices) * [1.6, 1.0, 0.6]
    mesh.vertices = o3d.utility.Vector3dVector(V)
    ang = np.arctan2(V[:, 1], V[:, 0])
    seen = V[:, 2] > -0.15 + 0.08 * np.sin(5 * ang)        # ragged rim
    pts = V[seen]
    rng = np.random.default_rng(0)
    pts = np.vstack([pts, pts + rng.normal(scale=0.002, size=pts.shape)])
    return mesh, pts


def test_flat_back_is_closed_orientable_and_flat(rm):
    mesh, pts = _one_sided_scan()
    out, back = rm.flat_back(mesh, pts)
    T = np.asarray(out.triangles)
    V = np.asarray(out.vertices)
    boundary, conflicts = rm._mesh_edge_report(T)
    assert boundary == 0 and len(conflicts) == 0
    assert out.is_edge_manifold(allow_boundary_edges=False) and out.is_orientable()
    assert rm._signed_volume(V, T) > 0                      # outward winding

    # the back is one plane (fitted to the wavy rim, so it may tilt a little),
    # everything else lies in front of it, and it faces down, inside the rim's
    # shallowest point (z = -0.07) so the cut crossed real surface all round
    assert back.sum() > 20
    P = V[back]
    c = P.mean(axis=0)
    _, sv, Wt = np.linalg.svd(P - c, full_matrices=False)
    n = Wt[2] if Wt[2][2] < 0 else -Wt[2]                   # outward = down
    assert sv[2] / np.sqrt(len(P)) < 1e-6                   # coplanar
    assert n[2] < -0.99                                     # facing down (within ~8 deg)
    assert ((V - c) @ n).max() < 1e-6                       # nothing behind the back
    assert -0.15 + 0.08 < c[2] < 0.0

    # the scanned top is kept: every surviving non-back vertex is an original one
    orig = np.asarray(mesh.vertices)
    idx = rm._match_rows(orig, V[~back])                 # each kept vertex -> its original
    assert (idx >= 0).mean() > 0.99


def test_flat_back_gives_up_without_an_open_rim(rm):
    mesh, _ = _one_sided_scan()
    everywhere = np.asarray(mesh.vertices)                  # points support the whole surface
    assert rm.flat_back(mesh, everywhere) is None
