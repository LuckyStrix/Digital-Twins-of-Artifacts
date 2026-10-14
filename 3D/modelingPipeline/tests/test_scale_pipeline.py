"""The ring scale wired through the app: Stage 2 env and checks, the split
warning, Stage 4's units and ring crop, and glTF in meters."""

import importlib.util
import json
import sys

import numpy as np
import pytest

from conftest import SCRIPT_DIR, RecordingRunner
from pipeline import settings as S
from pipeline.platform import venv_python


@pytest.fixture
def s(settings, two_side_scan, tmp_path):
    settings["input_var"] = str(two_side_scan)
    settings["output_var"] = str(tmp_path / "out")
    return settings


def fipmesh_scale(env):
    return {k: v for k, v in env.items() if k.startswith("FIPMESH_SCALE")}


def write_scale(ws, status, s=0.5, **extra):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "scale.json").write_text(json.dumps({
        "status": status, "units": "mm" if status == "scaled" else "model",
        "scale_mm_per_unit": s if status == "scaled" else None, "sigma_pct": 0.01,
        "reason": "" if status == "scaled" else "no ring found", "sparse_models": 1,
        "ring_mm": {"center": [0, 0, 0], "normal": [0, 0, 1], "r_inner": 100, "r_outer": 108},
        "models": [{"fit": {"ok": status == "scaled", "scale": s, "r_in": 200.0, "r_out": 216.0}}],
        **extra}))


def test_scale_tab_settings():
    assert "Scale" in S.TABS
    d = S.Settings.defaults()
    assert d["scale_ring_var"] is False
    assert (d["scale_ring_inner_var"], d["scale_ring_outer_var"], d["scale_ring_width_var"]) == \
        (200.0, 216.0, 0.0)
    for k in ("scale_ring_var", "scale_ring_inner_var", "scale_ring_outer_var", "scale_ring_width_var"):
        assert k in S.PERSISTED_KEYS


def test_stage2_env_when_on(s, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    s.update(scale_ring_var=True, scale_ring_inner_var=200.4, scale_ring_outer_var=216.1,
             scale_ring_width_var=7.9)
    out = tmp_path / "out"
    r = RecordingRunner(s)
    assert r.execute_stage(1)
    (_, _, e1), (_, _, e2) = r.calls[:2]
    for env, side in ((e1, "side1"), (e2, "side2")):
        assert fipmesh_scale(env) == {
            "FIPMESH_SCALE_RING": "1",
            "FIPMESH_SCALE_RING_INNER_MM": "200.4",
            "FIPMESH_SCALE_RING_OUTER_MM": "216.1",
            "FIPMESH_SCALE_RING_WIDTH_MM": "7.9",
            "FIPMESH_SCALE_RAW_IMAGES": str(tmp_path / "scan" / side),
            "FIPMESH_SCALE_PROCESSED_IMAGES": str(out / "processed" / side),
        }


def test_stage2_both_sides_scaled(s, tmp_path, monkeypatch):
    s["scale_ring_var"] = True
    out = tmp_path / "out"
    write_scale(out / "colmap_side1", "scaled", 0.5)
    write_scale(out / "colmap_side2", "scaled", 0.25)
    states = []
    r = RecordingRunner(s)
    r._ui_stage_state = lambda idx, state, status: states.append((idx, state, status))
    assert r.execute_stage(1)
    assert len(r.calls) == 2                                  # no unscale
    assert "[stage 2] scaled to millimeters: side1 1 unit = 0.5 mm; side2 1 unit = 0.25 mm" in r.info()
    assert states[-1] == (1, "done", "Done · scaled (mm)")


def test_stage2_one_side_failed_reverts_the_other(s, tmp_path):
    s["scale_ring_var"] = True
    out = tmp_path / "out"
    write_scale(out / "colmap_side1", "scaled")
    write_scale(out / "colmap_side2", "unscaled")
    r = RecordingRunner(s)
    assert r.execute_stage(1)
    assert r.calls[2][0] == [venv_python(), "-m", "scale.apply", "unscale", str(out / "colmap_side1")]
    assert r.calls[2][1] == str(SCRIPT_DIR)
    out_lines = [t for t, k in r.logs if k == "output"]
    assert "[warn] side2: not scaled: no ring found" in out_lines
    assert "[warn] scan left unscaled (see the warnings above)" in out_lines


def test_stage2_split_warning(s, tmp_path):
    r = RecordingRunner(s, output={"colmap_side1": ["[set] primary sparse models found: 3"],
                                   "colmap_side2": ["[set] primary sparse models found: 1"]})
    states = []
    r._ui_stage_state = lambda idx, state, status: states.append((idx, state, status))
    assert r.execute_stage(1)
    warns = [t for t, k in r.logs if k == "output" and t.startswith("[warn]")]
    assert len(warns) == 1 and "side1: COLMAP split the photos into 3 separate models" in warns[0]
    assert states[-1] == (1, "done", "Done · split: side1 3 models")


def test_stage2_scale_off_leaves_card_default(s):
    states = []
    r = RecordingRunner(s)
    r._ui_stage_state = lambda idx, state, status: states.append((idx, state, status))
    assert r.execute_stage(1)
    assert states[-1] == (1, "done", "")


@pytest.fixture
def merged(s, tmp_path):
    out = tmp_path / "out"
    (out / "aligned_cloud").mkdir(parents=True)
    (out / "aligned_cloud" / "merged_fpfh.ply").write_bytes(b"ply")
    return out


def test_stage4_units_mm_when_both_scaled(s, merged):
    s["scale_ring_var"] = True
    write_scale(merged / "colmap_side1", "scaled", 0.5)
    write_scale(merged / "colmap_side2", "scaled", 0.25)
    r = RecordingRunner(s)
    assert r.execute_stage(3)
    cmd = r.calls[0][0]
    assert cmd[cmd.index("--units") + 1] == "mm"
    assert cmd[cmd.index("--scale-json") + 1] == str(merged / "colmap_side1" / "scale.json")
    summary = json.loads((merged / "scale.json").read_text())
    assert summary["units"] == "mm" and summary["gltf_units"] == "m"
    assert summary["sides"]["side2"]["scale_mm_per_unit"] == 0.25


def test_stage4_unscaled_when_a_side_reverted(s, merged):
    s["scale_ring_var"] = True
    write_scale(merged / "colmap_side1", "reverted")
    write_scale(merged / "colmap_side2", "unscaled")
    r = RecordingRunner(s)
    assert r.execute_stage(3)
    cmd = r.calls[0][0]
    assert "--units" not in cmd and "--scale-json" not in cmd
    assert json.loads((merged / "scale.json").read_text())["units"] == "model"


def test_stage4_scale_off_unchanged(s, merged):
    write_scale(merged / "colmap_side1", "scaled")
    write_scale(merged / "colmap_side2", "scaled")
    r = RecordingRunner(s)
    assert r.execute_stage(3)
    assert "--units" not in r.calls[0][0]
    assert not (merged / "scale.json").exists()


def test_run_record_keeps_scale(s, tmp_path):
    from pipeline.paths import SessionPaths
    from pipeline.runlog import stage_outputs
    out = tmp_path / "out"
    write_scale(out / "colmap_side1", "scaled", 0.5)
    rec = stage_outputs(1, SessionPaths(s))
    assert rec["scale_side1"]["status"] == "scaled"
    assert rec["scale_side1"]["models"][0]["r_in"] == 200.0
    assert "scale_side2" not in rec


# ── Stage 4 script ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rm():
    pytest.importorskip("open3d")
    spec = importlib.util.spec_from_file_location("reconstruct_mesh_scale",
                                                  SCRIPT_DIR / "src" / "reconstruct_mesh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gltf_positions_in_meters(rm):
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [100.0, 0, 0], [0, 50.0, 20.0]]))
    mesh.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))
    gltf, buf = rm.build_gltf_document(mesh, unit_scale=0.001)
    pos = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert np.allclose(pos["max"], [0.1, 0.05, 0.02])
    gltf1, _ = rm.build_gltf_document(mesh)
    pos1 = gltf1["accessors"][gltf1["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
    assert np.allclose(pos1["max"], [100, 50, 20])


def test_crop_ring_keeps_tablet(rm, tmp_path):
    import open3d as o3d
    rng = np.random.default_rng(0)
    tablet = rng.uniform([-40, -30, -1], [40, 30, 25], size=(500, 3))        # inside the ring
    theta = rng.uniform(0, 2 * np.pi, 300)
    rr = rng.uniform(100, 108, 300)
    ring = np.column_stack([rr * np.cos(theta), rr * np.sin(theta), rng.uniform(-0.5, 0.5, 300)])
    felt = np.column_stack([130 * np.cos(theta), 130 * np.sin(theta), np.full(300, -1.0)])
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([tablet, ring, felt])))
    report = tmp_path / "scale.json"
    report.write_text(json.dumps({"status": "scaled", "ring_mm": {
        "center": [0, 0, 0], "normal": [0, 0, 1], "r_inner": 100.0, "r_outer": 108.0}}))
    ring_mm = rm.load_ring(str(report))
    kept, removed = rm.crop_ring(pcd, ring_mm)
    assert removed == 600 and len(kept.points) == 500
    report.write_text(json.dumps({"status": "unscaled"}))
    assert rm.load_ring(str(report)) is None
