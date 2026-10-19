"""scale/apply.py: the Stage 2 hook (all-or-none, scale.json, backups) and
undoing a side's scaling. COLMAP's model_converter is replaced by a copy:
the synthetic models are already TXT, which COLMAP reads as well."""

import json
import shutil
import struct

import numpy as np
import pytest

from scale import apply as sa
from scale.colmap_model import read_model_txt
from scale_synth import Scene, render

SMALL = dict(raw_w=480, raw_h=320, focal=420.0, steps=10)


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    return render(Scene(**SMALL), tmp_path_factory.mktemp("scene"))


@pytest.fixture(autouse=True)
def fake_converter(monkeypatch):
    def convert(colmap_bin, src, dst, out_type):
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix == ".txt":
                shutil.copy(f, dst / f.name)
    monkeypatch.setattr(sa, "_convert", convert)


def workspace(tmp_path, scene, n_models=1, break_second=False):
    ws = tmp_path / "colmap_side1"
    for i in range(n_models):
        md = ws / "sparse" / str(i)
        shutil.copytree(scene["model"], md)
        if break_second and i == 1:
            # photos that don't exist in the raw folder → this model can't fit
            p = md / "images.txt"
            p.write_text(p.read_text().replace("cam1/", "camX/").replace("cam2/", "camY/")
                         .replace("cam3/", "camZ/"))
    return ws, [ws / "sparse" / str(i) for i in range(n_models)]


def set_env(monkeypatch, scene, on="1"):
    monkeypatch.setenv("FIPMESH_SCALE_RING", on)
    monkeypatch.setenv("FIPMESH_SCALE_RING_INNER_MM", "200")
    monkeypatch.setenv("FIPMESH_SCALE_RING_OUTER_MM", "216")
    monkeypatch.setenv("FIPMESH_SCALE_RAW_IMAGES", str(scene["raw"]))
    monkeypatch.setenv("FIPMESH_SCALE_PROCESSED_IMAGES", str(scene["processed"]))


def test_scales_single_model(tmp_path, scene, monkeypatch):
    set_env(monkeypatch, scene)
    ws, models = workspace(tmp_path, scene)
    before = read_model_txt(models[0])
    logs = []
    report = sa.scale_from_env("colmap", models, ws, log=logs.append)
    assert report["status"] == "scaled" and report["units"] == "mm"
    s = report["scale_mm_per_unit"]
    assert abs(s / scene["mm_per_unit"] - 1) < 5e-4
    after = read_model_txt(models[0])
    i = next(iter(before.images))
    assert np.allclose(after.images[i].center, s * before.images[i].center)
    assert (ws / "sparse_unscaled" / "0" / "images.txt").read_text() == \
        (scene["model"] / "images.txt").read_text()
    saved = json.loads((ws / "scale.json").read_text())
    assert saved["status"] == "scaled"
    ring = saved["ring_mm"]
    assert abs(ring["r_inner"] - 100) < 0.2 and abs(ring["r_outer"] - 108) < 0.2
    m = saved["models"][0]["fit"]
    assert m["r_in"] > 0 and m["r_out"] > m["r_in"]       # raw radii kept for rescaling later
    assert any("side scaled to millimeters" in l for l in logs)


def test_one_model_failing_means_none_scaled(tmp_path, scene, monkeypatch):
    set_env(monkeypatch, scene)
    ws, models = workspace(tmp_path, scene, n_models=2, break_second=True)
    original = (models[0] / "images.txt").read_text()
    logs = []
    report = sa.scale_from_env("colmap", models, ws, log=logs.append)
    assert report["status"] == "unscaled"
    assert "1 of 2 models couldn't be scaled" in report["reason"]
    assert any("split this side into 2" in w for w in report["warnings"])
    assert (models[0] / "images.txt").read_text() == original
    assert not (ws / "sparse_unscaled").exists()
    assert any("continuing unscaled" in l for l in logs)


def test_disabled(tmp_path, scene, monkeypatch):
    set_env(monkeypatch, scene, on="0")
    ws, models = workspace(tmp_path, scene)
    assert sa.scale_from_env("colmap", models, ws, log=lambda *a: None)["status"] == "disabled"
    assert json.loads((ws / "scale.json").read_text())["status"] == "disabled"


def test_missing_raw_folder_never_raises(tmp_path, scene, monkeypatch):
    set_env(monkeypatch, scene)
    monkeypatch.setenv("FIPMESH_SCALE_RAW_IMAGES", str(tmp_path / "nope"))
    ws, models = workspace(tmp_path, scene)
    report = sa.scale_from_env("colmap", models, ws, log=lambda *a: None)
    assert report["status"] == "unscaled" and "raw photo folder not found" in report["reason"]


def test_unscale_workspace(tmp_path):
    ws = tmp_path / "colmap_side1"
    (ws / "dense" / "sparse").mkdir(parents=True)
    (ws / "sparse" / "0").mkdir(parents=True)
    (ws / "sparse" / "0" / "images.txt").write_text("scaled")
    (ws / "sparse_unscaled" / "0").mkdir(parents=True)
    (ws / "sparse_unscaled" / "0" / "images.txt").write_text("original")
    hdr = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\n" \
          b"property float y\nproperty float z\nend_header\n"
    (ws / "fused.ply").write_bytes(hdr + np.array([[2, 4, 6], [8, 10, 12]], "<f4").tobytes())
    buf = struct.pack("<Q", 1) + struct.pack("<I", 1) + struct.pack("<4d", 1, 0, 0, 0)
    buf += struct.pack("<3d", 2, 4, 6) + struct.pack("<I", 1) + b"a.png\x00" + struct.pack("<Q", 0)
    (ws / "dense" / "sparse" / "images.bin").write_bytes(buf)
    (ws / "camera_centers.json").write_text("[[2, 4, 6]]")
    (ws / "scale.json").write_text(json.dumps({"status": "scaled", "scale_mm_per_unit": 2.0}))

    assert sa.unscale_workspace(ws, log=lambda *a: None)
    pts = np.frombuffer((ws / "fused.ply").read_bytes()[len(hdr):], "<f4").reshape(-1, 3)
    assert np.allclose(pts, [[1, 2, 3], [4, 5, 6]])
    assert struct.unpack_from("<3d", (ws / "dense" / "sparse" / "images.bin").read_bytes(), 44) == (1, 2, 3)
    assert json.loads((ws / "camera_centers.json").read_text()) == [[1, 2, 3]]
    assert (ws / "sparse" / "0" / "images.txt").read_text() == "original"
    rep = json.loads((ws / "scale.json").read_text())
    assert rep["status"] == "reverted" and rep["units"] == "model"
    assert not sa.unscale_workspace(ws, log=lambda *a: None)      # only once


def test_cli_fit_only(tmp_path, scene, capsys):
    ws, models = workspace(tmp_path, scene)
    rc = sa.main(["fit", "--workspace", str(ws), "--raw", str(scene["raw"]),
                  "--processed", str(scene["processed"]), "--colmap-bin", "colmap"])
    assert rc == 0
    rep = json.loads((ws / "scale" / "fit_report.json").read_text())
    assert rep["status"] == "fit_only"
    assert (models[0] / "images.txt").read_text() == (scene["model"] / "images.txt").read_text()
    assert not (ws / "scale.json").exists()


def finished_session(tmp_path, scene, sides=("side1", "side2")):
    """A session whose Stage 2 finished with the Scale tab off."""
    scan, session = tmp_path / "scan", tmp_path / "scan_recon"
    hdr = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\n" \
          b"property float y\nproperty float z\nend_header\n"
    for side in sides:
        shutil.copytree(scene["raw"], scan / side)
        shutil.copytree(scene["processed"], session / "processed" / side)
        ws = session / f"colmap_{side}"
        shutil.copytree(scene["model"], ws / "sparse" / "0")
        (ws / "fused.ply").write_bytes(hdr + np.array([[1, 2, 3], [4, 5, 6]], "<f4").tobytes())
        (ws / "camera_centers.json").write_text("[[1, 2, 3]]")
        sa.write_report(ws, {"status": "disabled", "units": "model"})
    return session, scan, hdr


def test_scale_finished_session(tmp_path, scene):
    session, scan, hdr = finished_session(tmp_path, scene)
    logs = []
    res = sa.scale_finished_session("colmap", session, scan, sa.RingSpec(), log=logs.append)
    assert res["status"] == "scaled", res["reason"]
    for side in ("side1", "side2"):
        ws = session / f"colmap_{side}"
        rep = json.loads((ws / "scale.json").read_text())
        s = rep["scale_mm_per_unit"]
        assert rep["status"] == "scaled" and rep["scaled_after_stage2"]
        assert abs(s / scene["mm_per_unit"] - 1) < 5e-4
        pts = np.frombuffer((ws / "fused.ply").read_bytes()[len(hdr):], "<f4").reshape(-1, 3)
        assert np.allclose(pts, s * np.array([[1, 2, 3], [4, 5, 6]]), rtol=1e-6)
        assert np.allclose(json.loads((ws / "camera_centers.json").read_text()), [[s, 2 * s, 3 * s]])
        assert (ws / "sparse_unscaled" / "0" / "images.txt").is_file()
    # a second run refuses rather than scaling twice
    again = sa.scale_finished_session("colmap", session, scan, sa.RingSpec(), log=logs.append)
    assert again["status"] == "unscaled" and "already scaled" in again["reason"]
    # and Stage 2's revert still undoes it
    assert sa.unscale_workspace(session / "colmap_side1", log=lambda *a: None)
    pts = np.frombuffer((session / "colmap_side1" / "fused.ply").read_bytes()[len(hdr):], "<f4")
    assert np.allclose(pts.reshape(-1, 3), [[1, 2, 3], [4, 5, 6]], rtol=1e-5)


def test_finished_session_all_or_nothing(tmp_path, scene):
    session, scan, hdr = finished_session(tmp_path, scene)
    shutil.rmtree(scan / "side2")
    before = (session / "colmap_side1" / "fused.ply").read_bytes()
    res = sa.scale_finished_session("colmap", session, scan, sa.RingSpec(), log=lambda *a: None)
    assert res["status"] == "unscaled" and "side2: raw photo folder not found" in res["reason"]
    assert (session / "colmap_side1" / "fused.ply").read_bytes() == before
    assert json.loads((session / "colmap_side1" / "scale.json").read_text())["status"] == "disabled"


def test_finished_session_fit_only_cli(tmp_path, scene, capsys):
    session, scan, hdr = finished_session(tmp_path, scene, sides=("side1",))
    before = (session / "colmap_side1" / "fused.ply").read_bytes()
    rc = sa.main(["session", "--session", str(session), "--scan", str(scan), "--fit-only",
                  "--colmap-bin", "colmap"])
    assert rc == 0 and '"fit_only"' in capsys.readouterr().out
    assert (session / "colmap_side1" / "fused.ply").read_bytes() == before
