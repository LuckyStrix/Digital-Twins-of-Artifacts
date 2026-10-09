from pathlib import Path

from pipeline import platform
from pipeline.paths import SessionPaths, default_output_for, describe_sides, detect_sides
from pipeline.settings import Settings


def test_detect_and_describe_sides(tmp_path):
    inp = tmp_path / "scan"
    (inp / "b_side" / "nested").mkdir(parents=True)
    (inp / "b_side" / "nested" / "x.PNG").write_bytes(b"x")
    (inp / "a_side").mkdir()
    (inp / "a_side" / "x.jpeg").write_bytes(b"x")
    (inp / "notes").mkdir()
    (inp / "notes" / "readme.txt").write_text("hi")
    (inp / "c_side").mkdir()
    (inp / "c_side" / "x.tif").write_bytes(b"x")
    sides = detect_sides(inp)
    assert sides == ["a_side", "b_side", "c_side"]
    assert describe_sides(sides) == ("a_side", "b_side", "Detected sides: a_side, b_side, c_side")
    assert describe_sides(["front"]) == ("front", "", "Detected: 1 side (front), no alignment stage")
    assert describe_sides([]) == ("", "", "Flat structure — images at root, no alignment stage")
    assert detect_sides(tmp_path / "missing") == []


def test_default_output():
    assert default_output_for(str(Path("/data/scan"))) == str(Path("/data/scan_recon"))


def test_session_dir_resolution(tmp_path):
    s = Settings.defaults()
    p = SessionPaths(s)
    assert p.session_dir() == Path.home() / "tablet_recon"
    s["input_var"] = str(tmp_path / "scan") + "  "
    assert p.session_dir() == tmp_path / "scan_recon"
    s["output_var"] = str(tmp_path / "out")
    assert p.session_dir() == tmp_path / "out"


def test_stage_paths_and_readiness(tmp_path):
    s = Settings.defaults()
    s["output_var"] = str(tmp_path)
    p = SessionPaths(s)
    assert p.expected_output_for_stage(0) == tmp_path / "processed"
    assert p.expected_output_for_stage(1) == tmp_path / "colmap_side1" / "fused.ply"
    assert p.expected_output_for_stage(2) == tmp_path / "aligned_cloud" / "merged_fpfh.ply"
    assert p.expected_output_for_stage(3) == tmp_path / "recon" / "recon_mesh_recon.obj"
    assert p.icp_report_path() == tmp_path / "aligned_cloud" / "icp_report.json"
    assert p.input_ply_for_recon() == tmp_path / "aligned_cloud" / "merged_fpfh.ply"
    assert not any(p.stage_output_ready(i) for i in range(4))

    (tmp_path / "processed" / "side1").mkdir(parents=True)
    assert not p.stage_output_ready(0)
    (tmp_path / "processed" / "side1" / "a.jpg").write_bytes(b"x")
    assert p.stage_output_ready(0)

    (tmp_path / "colmap_side1").mkdir()
    (tmp_path / "colmap_side1" / "fused.ply").write_bytes(b"")
    assert not p.stage_output_ready(1)      # empty file doesn't count
    (tmp_path / "colmap_side1" / "fused.ply").write_bytes(b"ply")
    assert not p.stage_output_ready(1)      # side2 (default "side2") isn't ready yet
    (tmp_path / "colmap_side2").mkdir()
    (tmp_path / "colmap_side2" / "fused.ply").write_bytes(b"ply")
    assert p.stage_output_ready(1)

    s["side1_var"], s["side2_var"] = "", ""
    assert p.expected_output_for_stage(1) == tmp_path / "colmap_out" / "fused.ply"
    assert p.input_ply_for_recon() == tmp_path / "colmap_out" / "fused.ply"
    assert p.side1_camera_centers_path() == tmp_path / "colmap_out" / "camera_centers.json"


def test_windows_path_detection():
    for p in (r"D:\scans", "D:/scans", "c:", r"\\server\share\x"):
        assert platform.looks_like_windows_path(p), p
    for p in ("/mnt/d/scans", "scans", "~/x", "D", "Dx:\\"):
        assert not platform.looks_like_windows_path(p), p


def test_to_posix_path_outside_wsl(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    assert platform.to_posix_path(r" D:\scans ") == r"D:\scans"
    assert platform.native_folder_dialog() is None


def test_has_display(monkeypatch):
    monkeypatch.setattr(platform.sys, "platform", "linux")
    for k in ("WSL_DISTRO_NAME", "WSL_INTEROP", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(k, raising=False)
    assert not platform.has_display()
    monkeypatch.setenv("DISPLAY", ":0")
    assert platform.has_display()
    monkeypatch.delenv("DISPLAY")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    assert platform.has_display()


def test_viewer_env_workarounds(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("GLFW_PLATFORM", raising=False)
    env = platform.viewer_env()
    assert "WAYLAND_DISPLAY" not in env
    assert env["GLFW_PLATFORM"] == "x11"
    assert env["LIBGL_ALWAYS_SOFTWARE"] == "1"


def test_venv_python(monkeypatch, tmp_path):
    repo_venv = tmp_path / "venv" / "bin" / "python3"
    repo_venv.parent.mkdir(parents=True)
    repo_venv.touch()
    monkeypatch.setattr(platform, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(platform.sys, "executable", "/usr/bin/python3")
    # The repo's venv wins over the current interpreter, even an activated
    # venv, unless FIPMESH_PYTHON (the Docker image) says otherwise.
    monkeypatch.delenv("FIPMESH_PYTHON", raising=False)
    assert platform.venv_python() == str(repo_venv)
    monkeypatch.setenv("FIPMESH_PYTHON", "/opt/venv/bin/python3")
    assert platform.venv_python() == "/opt/venv/bin/python3"
    monkeypatch.delenv("FIPMESH_PYTHON")
    repo_venv.unlink()
    assert platform.venv_python() == "/usr/bin/python3"
