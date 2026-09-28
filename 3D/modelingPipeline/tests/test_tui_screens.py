"""Headless tests: pickers, view actions, ICP details."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("textual")

from pipeline import platform as plat  # noqa: E402
from test_tui import log_text, make_app, run  # noqa: E402
from tui.screens import (  # noqa: E402
    ConfirmScreen, IcpDetailsScreen, NewFolderScreen, PathPicker, convergence_rows, histogram_text,
)


@pytest.fixture
def no_wsl(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)


async def submit_path(app, pilot, value):
    box = app.screen.query_one("#picker-path")
    box.value = str(value)
    box.focus()
    await pilot.press("enter")
    await pilot.pause()


# ── pickers ───────────────────────────────────────────────────────────────────

def test_browse_input_with_path_box(tmp_path, two_side_scan, no_wsl):
    async def body(app, pilot):
        app.query_one("#tabs").active = "tab-inputs"
        await pilot.pause()
        await pilot.click("#browse-input_var")
        await pilot.pause()
        assert isinstance(app.screen, PathPicker)
        locs = app.screen.query_one("#locations")
        assert [str(locs.get_option_at_index(i).prompt) for i in range(2)] == ["Home", "/"]
        await submit_path(app, pilot, str(two_side_scan))
        await pilot.pause(0.7)
        assert not isinstance(app.screen, PathPicker)
        assert app.settings["input_var"] == str(two_side_scan)
        assert app.settings["output_var"] == str(tmp_path / "scan_recon")
        assert json.loads((tmp_path / "ui.json").read_text())["recent_dirs"] == [str(two_side_scan)]
    run(make_app(tmp_path), body)


def test_picker_shows_recents(tmp_path, no_wsl):
    (tmp_path / "ui.json").write_text(json.dumps({"recent_dirs": [str(tmp_path)]}))

    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause()
        locs = app.screen.query_one("#locations")
        prompts = [str(locs.get_option_at_index(i).prompt) for i in range(locs.option_count)]
        assert prompts[-1] == str(tmp_path)
        locs.highlighted = locs.option_count - 1
        locs.action_select()
        await pilot.pause()
        assert app.screen.query_one("#picker-path").value == str(tmp_path)
    run(make_app(tmp_path), body)


def test_picker_rejects_missing_folder_and_converts_windows_paths(tmp_path, monkeypatch, no_wsl):
    target = tmp_path / "d_drive" / "scans"
    target.mkdir(parents=True)
    win = "D:" + "\\" + "scans"
    monkeypatch.setattr(plat, "to_posix_path", lambda p: str(target) if p.strip() == win else p)

    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause()
        (tmp_path / "a_file").write_text("")
        await submit_path(app, pilot, str(tmp_path / "a_file"))
        await pilot.pause()
        assert isinstance(app.screen, PathPicker)          # still open
        await submit_path(app, pilot, str(tmp_path / "nope"))
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)       # offers to create it
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, PathPicker)
        assert not (tmp_path / "nope").exists()
        await submit_path(app, pilot, win)
        await pilot.pause()
        assert app.settings["output_var"] == str(target)
    run(make_app(tmp_path), body)


def test_picker_creates_missing_output_folder(tmp_path, no_wsl):
    target = tmp_path / "new" / "out"

    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause()
        await submit_path(app, pilot, str(target))
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await pilot.pause()
        assert target.is_dir()
        assert app.settings["output_var"] == str(target)
    run(make_app(tmp_path), body)


def test_picker_new_folder_button(tmp_path, no_wsl):
    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause()
        app.screen.query_one("#picker-path").value = str(tmp_path)
        await pilot.click("#new-folder-btn")
        await pilot.pause()
        assert isinstance(app.screen, NewFolderScreen)
        app.screen.query_one("#folder-name").value = "results"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, PathPicker)
        assert (tmp_path / "results").is_dir()
        assert app.screen.query_one("#picker-path").value == str(tmp_path / "results")
        await pilot.click("#ok")
        await pilot.pause()
        assert app.settings["output_var"] == str(tmp_path / "results")
    run(make_app(tmp_path), body)


def test_picker_tree_select_expands_but_never_collapses(tmp_path, no_wsl):
    (tmp_path / "outer" / "inner").mkdir(parents=True)

    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause()
        tree = app.screen.query_one("#tree")
        tree.path = tmp_path
        await pilot.pause(0.3)
        node = next(n for n in tree.root.children if n.data.path.name == "outer")
        for _ in range(3):                      # repeated clicks/enters keep it open
            tree.move_cursor(node)
            tree.action_select_cursor()
            await pilot.pause(0.2)
            assert node.is_expanded
        assert [c.data.path.name for c in node.children] == ["inner"]
        assert app.screen.query_one("#picker-path").value == str(tmp_path / "outer")
    run(make_app(tmp_path), body)


def test_file_picker_filters_and_selects(tmp_path, no_wsl):
    (tmp_path / "cam.json").write_text("{}")
    (tmp_path / "other.txt").write_text("")

    async def body(app, pilot):
        app.browse("intr_file_var")
        await pilot.pause(0.3)
        tree = app.screen.query_one("#tree")
        names = [p.name for p in tree.filter_paths(tmp_path.iterdir())]
        assert "cam.json" in names and "other.txt" not in names
        await submit_path(app, pilot, str(tmp_path))  # folder in file mode: navigate
        await pilot.pause()
        assert isinstance(app.screen, PathPicker)
        await submit_path(app, pilot, str(tmp_path / "cam.json"))
        await pilot.pause()
        assert app.settings["intr_file_var"] == str(tmp_path / "cam.json")
    run(make_app(tmp_path), body)


def test_native_dialog_result_is_used(tmp_path, monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(plat, "browse_locations", lambda: [("Home", tmp_path)])
    monkeypatch.setattr(plat, "native_folder_dialog", lambda title, start: str(tmp_path))

    async def body(app, pilot):
        app.browse("output_var")
        for _ in range(40):
            await pilot.pause(0.05)
            if not isinstance(app.screen, PathPicker):
                break
        assert app.settings["output_var"] == str(tmp_path)
    run(make_app(tmp_path), body)


def test_native_dialog_cancel_keeps_terminal_picker(tmp_path, monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr(plat, "browse_locations", lambda: [("Home", tmp_path)])
    monkeypatch.setattr(plat, "native_folder_dialog", lambda title, start: None)

    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause(0.5)
        assert isinstance(app.screen, PathPicker)
        app.screen.query_one("#auto-native").value = False
        await pilot.pause()
        assert json.loads((tmp_path / "ui.json").read_text())["native_dialog"] is False
    run(make_app(tmp_path), body)


def test_path_field_converts_windows_path_on_submit(tmp_path, monkeypatch):
    win = "D:" + "\\" + "scans"
    monkeypatch.setattr(plat, "to_posix_path", lambda p: "/mnt/d/scans" if p == win else p)

    async def body(app, pilot):
        app.query_one("#tabs").active = "tab-inputs"
        await pilot.pause()
        inp = app.query_one("#set-output_var")
        inp.focus()
        inp.value = win
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings["output_var"] == "/mnt/d/scans"
        assert inp.value == "/mnt/d/scans"
    run(make_app(tmp_path), body)


# ── view actions ──────────────────────────────────────────────────────────────

def _session_with_outputs(tmp_path):
    out = tmp_path / "out"
    (out / "processed" / "side1").mkdir(parents=True)
    (out / "processed" / "side1" / "a.jpg").write_bytes(b"x")
    (out / "colmap_side1").mkdir()
    (out / "colmap_side1" / "fused.ply").write_bytes(b"ply")
    return out


def test_view_photos_opens_folder(tmp_path, monkeypatch):
    out = _session_with_outputs(tmp_path)
    opened = []
    monkeypatch.setattr(plat, "open_folder", lambda d: opened.append(d) or None)

    async def body(app, pilot):
        app.form.set_value("output_var", str(out))
        app.reconcile_stage_states()
        await pilot.pause()
        assert not app.query_one("#view-0").disabled
        app.query_one("#view-0").press()
        await pilot.pause()
        assert opened == [out / "processed"]
    run(make_app(tmp_path), body)


def test_view_without_display_shows_path(tmp_path, monkeypatch):
    out = _session_with_outputs(tmp_path)
    monkeypatch.setattr(plat, "has_display", lambda: False)
    monkeypatch.setattr(plat, "launch_viewer", lambda p: pytest.fail("viewer launched"))

    async def body(app, pilot):
        app.form.set_value("output_var", str(out))
        app.view_stage(1)
        await pilot.pause()
        assert any(str(out / "colmap_side1" / "fused.ply") in n.message for n in app._notifications)
    run(make_app(tmp_path), body)


def test_view_missing_file_warns(tmp_path):
    async def body(app, pilot):
        app.form.set_value("output_var", str(tmp_path / "empty"))
        app.view_stage(3)
        await pilot.pause()
        assert any(n.title == "File not found" for n in app._notifications)
    run(make_app(tmp_path), body)


def test_viewer_failure_is_reported(tmp_path, monkeypatch):
    out = _session_with_outputs(tmp_path)
    monkeypatch.setattr(plat, "has_display", lambda: True)
    monkeypatch.setattr(plat, "launch_viewer", lambda p: subprocess.Popen(
        [sys.executable, "-c", "print('Failed creating OpenGL window')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))

    async def body(app, pilot):
        app.form.set_value("output_var", str(out))
        app.view_stage(1)
        for _ in range(100):
            await pilot.pause(0.05)
            if any(n.title == "Viewer failed" for n in app._notifications):
                break
        assert any(n.title == "Viewer failed" for n in app._notifications)
        await pilot.pause(0.2)
        assert "[viewer] Failed creating OpenGL window" in log_text(app)
    run(make_app(tmp_path), body)


# ── ICP details ───────────────────────────────────────────────────────────────

ICP_REPORT = {
    "voxel": 0.5, "method": "fpfh",
    "params": {"thresholds": [3, 1], "max_iters": 100, "tol": 1e-7, "points": 300000,
               "band": 0, "robust": 0},
    "before": {"n": 10, "mean": 0.2, "median": 0.1, "p95": 0.4, "median_vox": 0.2,
               "fitness": 0.8, "rmse": 0.3},
    "after": {"n": 10, "mean": 0.1, "median": 0.05, "p95": 0.2, "median_vox": 0.1,
              "fitness": 0.9, "rmse": 0.2},
    "hist_before": {"counts": [5, 3, 2, 0], "hi_vox": 2.0},
    "hist_after": {"counts": [8, 1, 1, 0], "hi_vox": 2.0},
    "stages": [
        {"stage": 1, "threshold": 1.5, "mult": 3, "n_src": 100, "n_tgt": 120, "iters": 20,
         "trace": [{"iter": 10, "fitness": 0.8, "rmse": 0.3},
                   {"iter": 20, "fitness": 0.85, "rmse": 0.25}],
         "move_translation": 0.05, "move_rotation_deg": 0.1},
        {"stage": 2, "threshold": 0.5, "mult": 1, "n_src": 90, "n_tgt": 110, "iters": 10,
         "trace": [{"iter": 10, "fitness": 0.9, "rmse": 0.2}],
         "move_translation": 0.01, "move_rotation_deg": 0.02},
    ],
}


def test_icp_details_from_session(tmp_path):
    rp = tmp_path / "out" / "aligned_cloud" / "icp_report.json"
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(ICP_REPORT))

    async def body(app, pilot):
        app.form.set_value("output_var", str(tmp_path / "out"))
        await pilot.press("i")
        await pilot.pause()
        assert isinstance(app.screen, IcpDetailsScreen)
        assert app.screen.query_one("#icp-summary").row_count == 5
        assert app.screen.query_one("#icp-stages").row_count == 2
        assert "s1 3 vox" in str(app.screen.query_one("#icp-conv").render())
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, IcpDetailsScreen)
    run(make_app(tmp_path), body)


def test_icp_details_without_report_opens_picker(tmp_path, no_wsl):
    other = tmp_path / "elsewhere.json"
    other.write_text(json.dumps(ICP_REPORT))
    bad = tmp_path / "bad.json"
    bad.write_text("{")

    async def body(app, pilot):
        app.form.set_value("output_var", str(tmp_path / "out"))
        app.query_one("#tabs").active = "tab-alignment"
        await pilot.pause()
        app.query_one("#icp-details").press()
        await pilot.pause()
        assert isinstance(app.screen, PathPicker)
        await submit_path(app, pilot, str(bad))
        await pilot.pause()
        assert any(n.severity == "error" for n in app._notifications)
        app.action_icp_details()
        await pilot.pause()
        await submit_path(app, pilot, str(other))
        await pilot.pause()
        assert isinstance(app.screen, IcpDetailsScreen)
    run(make_app(tmp_path), body)


def test_plot_helpers():
    rows = convergence_rows(ICP_REPORT["stages"])
    assert [r[0] for r in rows] == ["s1 3 vox", "s2 1 vox"]
    assert rows[0][1] == "█▅" and rows[1][1] == "▁"
    assert rows[0][2] == "0.3 → 0.25"
    assert convergence_rows([]) == []
    t = histogram_text(ICP_REPORT["hist_before"], ICP_REPORT["hist_after"], 4, "grey", "blue")
    lines = t.plain.split("\n")
    assert len(lines) == 6
    assert lines[0].startswith(" 80.0% │")
    # first bin: after (80%) reaches the top row, before (50%) doesn't
    assert lines[0][8:10] == " █"
    assert lines[-1].rstrip().endswith(">= 2 vox")
