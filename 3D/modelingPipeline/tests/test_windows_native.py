"""Native-Windows support: the Python Stage 2 driver (src/run_stage2.py) and
the Windows branches of the runner and platform helpers.

These run on any OS: Windows is simulated by patching sys.platform, and
COLMAP / exiftool / the MVS script are replaced by recorded subprocess calls.
"""

import os
import sys
from pathlib import Path

import pytest

from conftest import SCRIPT_DIR, RecordingRunner

sys.path.insert(0, str(SCRIPT_DIR / "src"))
import run_stage2  # noqa: E402
from pipeline import platform as plat  # noqa: E402
from pipeline import runner  # noqa: E402


# ── run_stage2: resource defaults (must match run.sh's tiers) ─────────────────

@pytest.mark.parametrize("mem_gb, expected", [
    (16, (6, 6, 8, 8, 12)),
    (24, (8, 8, 10, 10, 16)),
    (32, (10, 10, 12, 12, 24)),
    (64, (16, 16, 16, 16, 64)),
    (0,  (16, 16, 16, 16, 64)),    # unknown RAM: run.sh's untiered defaults
])
def test_resource_defaults_tiers(mem_gb, expected):
    d = run_stage2.resource_defaults(16, mem_gb)
    got = (d["FIPMESH_COLMAP_EXTRACT_THREADS"], d["FIPMESH_COLMAP_MATCH_THREADS"],
           d["FIPMESH_COLMAP_MAPPER_THREADS"], d["FIPMESH_COLMAP_FUSION_THREADS"],
           d["FIPMESH_COLMAP_PATCH_CACHE_SIZE"])
    assert got == expected
    assert d["FIPMESH_COLMAP_FUSION_CACHE_SIZE"] == expected[4]


def test_resource_defaults_few_cpus():
    d = run_stage2.resource_defaults(4, 16)
    assert d["FIPMESH_COLMAP_EXTRACT_THREADS"] == 4
    assert d["FIPMESH_COLMAP_MAPPER_THREADS"] == 4


def test_apply_defaults_keeps_caller_values():
    env = {"FIPMESH_COLMAP_QUALITY": "extreme", "FIPMESH_COLMAP_IMAGE_STRIDE": "1",
           "FIPMESH_COLMAP_USE_GPU": ""}          # empty counts as unset, like ${VAR:-x}
    run_stage2.apply_defaults(env, secondary=False, cpus=8, mem_gb=64)
    assert env["FIPMESH_COLMAP_QUALITY"] == "extreme"
    assert env["FIPMESH_COLMAP_IMAGE_STRIDE"] == "1"
    assert env["FIPMESH_COLMAP_USE_GPU"] == "1"
    assert env["FIPMESH_COLMAP_SIFT_MAX_NUM_FEATURES"] == "16000"
    assert "FIPMESH_COLMAP_SECONDARY_ROTATE_DEG" not in env


def test_apply_defaults_secondary():
    env = {}
    run_stage2.apply_defaults(env, secondary=True, cpus=8, mem_gb=64)
    assert env["FIPMESH_COLMAP_SECONDARY_ROTATE_DEG"] == "180"
    assert env["FIPMESH_COLMAP_SECONDARY_ROTATE_AXIS"] == "primary_frame_x"
    assert env["FIPMESH_COLMAP_SECONDARY_ALIGN_MODE"] == "auto"


def test_total_mem_gb_positive():
    assert run_stage2.total_mem_gb() > 0


# ── run_stage2: binary resolution ─────────────────────────────────────────────

def test_resolve_colmap_env_wins(monkeypatch):
    assert run_stage2.resolve_colmap({"FIPMESH_COLMAP_BIN": "X:/c.bat"}) == "X:/c.bat"


def test_resolve_colmap_windows_zip_layout(monkeypatch, tmp_path):
    (tmp_path / "colmap").mkdir()
    (tmp_path / "colmap" / "COLMAP.bat").write_text("@echo off\n")
    monkeypatch.setattr(run_stage2, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(run_stage2, "WINDOWS", True)
    assert run_stage2.resolve_colmap({}) == str(tmp_path / "colmap" / "COLMAP.bat")
    monkeypatch.setattr(run_stage2, "WINDOWS", False)
    monkeypatch.setattr(run_stage2.shutil, "which", lambda name: None)
    assert run_stage2.resolve_colmap({}) is None


def test_resolve_exiftool_windows_tools_folder(monkeypatch, tmp_path):
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "exiftool.exe").write_bytes(b"")
    monkeypatch.setattr(run_stage2, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(run_stage2, "WINDOWS", True)
    monkeypatch.setattr(run_stage2.shutil, "which", lambda name: None)
    assert run_stage2.resolve_exiftool({}) == str(tmp_path / "tools" / "exiftool.exe")


# ── run_stage2: the full run with recorded subprocesses ───────────────────────

class Recorder:
    def __init__(self, rc=0, fail_on=None):
        self.calls = []
        self.rc, self.fail_on = rc, fail_on

    def __call__(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        rc = 1 if self.fail_on and any(self.fail_on in str(c) for c in cmd) else self.rc
        return type("R", (), {"returncode": rc})()


@pytest.fixture
def stage2(monkeypatch, tmp_path):
    rec = Recorder()
    monkeypatch.setattr(run_stage2.subprocess, "run", rec)
    imgs = tmp_path / "processed" / "side1"
    imgs.mkdir(parents=True)
    masks = tmp_path / "masks" / "side1"
    masks.mkdir(parents=True)
    env = {"FIPMESH_COLMAP_BIN": "COLMAP.bat", "FIPMESH_EXIFTOOL_BIN": "exiftool.exe",
           "FIPMESH_SKIP_RECON": "1"}
    return rec, imgs, masks, tmp_path / "colmap_side1", env


def test_run_single_side(stage2):
    rec, imgs, masks, out, env = stage2
    args = run_stage2.parse_args(["-i", str(imgs), "-o", str(out), "-m", str(masks), "-v"])
    run_stage2.run(args, env)

    (exif_cmd, _), (mvs_cmd, mvs_kw) = rec.calls     # recon skipped
    assert exif_cmd == ["exiftool.exe", "-overwrite_original", "-all:all=", "-r", str(imgs)]
    assert mvs_cmd == [
        sys.executable, str(SCRIPT_DIR / "src" / "run_colmap_mvs.py"),
        "--images", str(imgs),
        "--workspace", str(out),
        "--dense-cloud-out", str(out / "fused.ply"),
        "--single-camera-per-folder", "1",
        "--clean-workspace",
    ]
    child_env = mvs_kw["env"]
    assert child_env["FIPMESH_COLMAP_BIN"] == "COLMAP.bat"
    assert child_env["FIPMESH_COLMAP_MASK_PATH"] == str(masks)
    assert child_env["FIPMESH_COLMAP_QUALITY"] == "high"
    assert (out / "fused.ply").exists()


def test_run_relative_paths_resolve_against_script_dir(stage2, monkeypatch, tmp_path):
    rec, imgs, masks, out, env = stage2
    monkeypatch.setattr(run_stage2, "SCRIPT_DIR", tmp_path)
    run_stage2.run(run_stage2.parse_args(["-i", "processed/side1", "-o", "colmap_side1"]), env)
    assert rec.calls[1][0][3] == str(imgs)
    assert out.is_dir()


def test_run_two_sides_and_recon(stage2, tmp_path):
    rec, imgs, masks, out, env = stage2
    env.pop("FIPMESH_SKIP_RECON")
    side2 = tmp_path / "processed" / "side2"
    side2.mkdir()
    run_stage2.run(run_stage2.parse_args(["-i", str(imgs), "-s", str(side2), "-o", str(out)]), env)
    cmds = [c for c, _ in rec.calls]
    assert [c[-1] for c in cmds[:2]] == [str(imgs), str(side2)]     # exiftool both sides
    assert cmds[2][cmds[2].index("--images-secondary") + 1] == str(side2)
    assert cmds[3][1].endswith("reconstruct_mesh.py")
    assert rec.calls[2][1]["env"]["FIPMESH_COLMAP_IMAGES_SECONDARY"] == str(side2)


@pytest.mark.parametrize("argv, message", [
    (["-i", "nope"], "invalid primary image directory"),
    (["-i", "{imgs}", "-n", "{masks}"], "-n given without -s"),
    (["-i", "{imgs}", "-m", "nope"], "invalid primary mask directory"),
])
def test_run_bad_args(stage2, argv, message):
    rec, imgs, masks, out, env = stage2
    argv = [a.format(imgs=imgs, masks=masks) for a in argv]
    with pytest.raises(run_stage2.Stage2Error, match=message):
        run_stage2.run(run_stage2.parse_args(argv + ["-o", str(out)]), env)
    assert rec.calls == []


def test_run_missing_tools(stage2, monkeypatch):
    rec, imgs, masks, out, env = stage2
    monkeypatch.setattr(run_stage2, "resolve_colmap", lambda env: None)
    monkeypatch.setattr(run_stage2, "resolve_exiftool", lambda env: None)
    with pytest.raises(run_stage2.Stage2Error, match="colmap, exiftool"):
        run_stage2.run(run_stage2.parse_args(["-i", str(imgs), "-o", str(out)]), env)


@pytest.mark.parametrize("fail_on, message", [
    ("exiftool", "exiftool failed"),
    ("run_colmap_mvs.py", "COLMAP MVS pipeline failed"),
])
def test_run_step_failure(stage2, fail_on, message):
    rec, imgs, masks, out, env = stage2
    rec.fail_on = fail_on
    with pytest.raises(run_stage2.Stage2Error, match=message):
        run_stage2.run(run_stage2.parse_args(["-i", str(imgs), "-o", str(out)]), env)


def test_main_returns_1_on_error(capsys):
    assert run_stage2.main(["-i", "does/not/exist"]) == 1
    assert "invalid primary image directory" in capsys.readouterr().err


# ── runner / platform on Windows ──────────────────────────────────────────────

@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")


def test_stage2_uses_python_driver_on_windows(windows, settings, two_side_scan, tmp_path):
    settings["input_var"] = str(two_side_scan)
    settings["output_var"] = str(tmp_path / "out")
    r = RecordingRunner(settings)
    assert r.execute_stage(1)
    out = tmp_path / "out"
    (cmd1, cwd, env), (cmd2, _, _) = r.calls
    assert cmd1 == [plat.venv_python(), str(SCRIPT_DIR / "src" / "run_stage2.py"),
                    "-i", str(out / "processed" / "side1"),
                    "-o", str(out / "colmap_side1"), "-v"]
    assert cmd2[3] == str(out / "processed" / "side2")
    assert env["FIPMESH_SKIP_RECON"] == "1"


def test_stage2_python_driver_passes_masks(windows, tmp_path):
    cmd = runner.stage2_cmd(tmp_path / "img", tmp_path / "out", tmp_path / "m")
    assert cmd[-2:] == ["-m", str(tmp_path / "m")]


def test_stage2_still_uses_run_sh_on_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    assert runner.stage2_cmd(tmp_path, tmp_path, None)[0] == "bash"


def test_venv_python_windows_layout(windows, monkeypatch, tmp_path):
    exe = tmp_path / "venv" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setattr(plat, "SCRIPT_DIR", tmp_path)
    assert plat.venv_python() == str(exe)


def test_viewer_on_windows_uses_venv_and_plain_env(windows, monkeypatch, tmp_path):
    monkeypatch.setenv("WAYLAND_DISPLAY", "w")
    cmd, env = plat.resolve_viewer_launch(tmp_path / "m.ply")
    assert cmd == [plat.venv_python(), str(SCRIPT_DIR / "viewer.py"), str(tmp_path / "m.ply")]
    assert env == dict(os.environ)       # none of the WSLg workarounds


def test_windows_has_native_dialogs_and_drive_locations(windows, monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    assert plat.has_native_dialogs()
    monkeypatch.setattr(plat, "_windows_drives", lambda: [Path("C:\\"), Path("D:\\")])
    labels = [label for label, _ in plat.browse_locations()]
    assert labels[0] == "Home" and len(labels) == 3


def test_to_windows_path_noop_off_wsl(windows, monkeypatch, tmp_path):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    assert plat.to_windows_path(tmp_path) == str(tmp_path)
