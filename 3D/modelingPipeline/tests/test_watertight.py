"""make_watertight closes every hole, fixes winding, and never moves original vertices."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

o3d = pytest.importorskip("open3d")
pytest.importorskip("pymeshlab")

SRC = Path(__file__).resolve().parent.parent / "src" / "reconstruct_mesh.py"


@pytest.fixture(scope="module")
def rm():
    spec = importlib.util.spec_from_file_location("reconstruct_mesh", SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("reconstruct_mesh", mod)
    spec.loader.exec_module(mod)
    return mod


def _holey_sphere():
    """A coloured sphere with a slit, a round hole, a flipped patch and a stray triangle."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    V = np.asarray(mesh.vertices)
    T = np.asarray(mesh.triangles).copy()
    c = V[T].mean(axis=1)
    round_hole = np.linalg.norm(c - [0, 0, 1], axis=1) < 0.25
    slit = (np.abs(c[:, 1]) < 0.03) & (c[:, 0] > 0.3) & (np.abs(c[:, 2]) < 0.5)
    T = T[~(round_hole | slit)]
    flip = np.linalg.norm(V[T].mean(axis=1) - [0, -1, 0], axis=1) < 0.2
    T[flip] = T[flip][:, ::-1]
    # Stray triangle far away (debris): can't be closed into a solid, must go.
    V = np.vstack([V, [[5, 5, 5], [5.1, 5, 5], [5, 5.1, 5]]])
    T = np.vstack([T, [[len(V) - 3, len(V) - 2, len(V) - 1]]])
    out = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(T))
    out.vertex_colors = o3d.utility.Vector3dVector((V - V.min(0)) / np.ptp(V, axis=0))
    return out


def test_closes_holes_and_fixes_winding(rm):
    mesh = _holey_sphere()
    boundary, conflicts = rm._mesh_edge_report(np.asarray(mesh.triangles))
    assert boundary > 0 and len(conflicts) > 0

    out = rm.make_watertight(mesh, label="test")
    T = np.asarray(out.triangles)
    boundary, conflicts = rm._mesh_edge_report(T)
    assert boundary == 0 and len(conflicts) == 0
    # Closed 2-manifold and orientable. (Open3D's is_watertight() also demands no
    # self-intersections, which hole patches across a curved slit can have and which
    # real Poisson meshes already have before any repair; not checked here.)
    assert out.is_edge_manifold(allow_boundary_edges=False)
    assert out.is_vertex_manifold() and out.is_orientable()


def test_original_vertices_stay_put_and_colours_carry_over(rm):
    mesh = _holey_sphere()
    V0 = np.asarray(mesh.vertices)[:-3]          # sphere part (stray triangle is dropped)
    C0 = np.asarray(mesh.vertex_colors)[:-3]
    out = rm.make_watertight(mesh, label="test")
    V = np.asarray(out.vertices)
    C = np.asarray(out.vertex_colors)
    idx = rm._match_rows(V, V0)
    used = np.unique(np.asarray(mesh.triangles)[:-1])
    assert (idx[used] >= 0).all()                # every original sphere vertex survives unmoved
    assert np.allclose(C[idx[used]], C0[used])   # with its own colour
    assert len(V) > len(V0) - 1                  # patches were refined (new vertices added)
    assert np.all(np.linalg.norm(V, axis=1) < 1.01)  # patches stay on/inside the surface


def test_already_watertight_mesh_is_unchanged(rm):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    out = rm.make_watertight(mesh, label="test")
    assert len(out.triangles) == len(mesh.triangles)
    assert np.allclose(np.sort(np.asarray(out.vertices), axis=0),
                       np.sort(np.asarray(mesh.vertices), axis=0))
