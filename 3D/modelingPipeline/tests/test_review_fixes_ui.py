"""Regression tests for review findings: UI layer."""

import json
import os
import re
import subprocess
import sys
import threading

import pytest

from pipeline import platform as plat

POSIX = os.name == "posix"
ROOT_USER = POSIX and os.geteuid() == 0


# ── UI layer ──────────────────────────────────────────────────────────────────

textual = pytest.importorskip("textual")

from textual.widgets import Checkbox, Input  # noqa: E402

from test_tui import FakeRunner, log_text, make_app, run, wait_idle  # noqa: E402,F401
from test_tui_screens import ICP_REPORT, submit_path  # noqa: E402
from tui.screens import IcpDetailsScreen, PathPicker, icp_report_problem  # noqa: E402


@pytest.fixture
def fake_runner():
    FakeRunner.block = None
    FakeRunner.commands = []
    yield FakeRunner
    FakeRunner.block = None


@pytest.mark.skipif(not POSIX or ROOT_USER, reason="needs POSIX permissions as non-root")
def test_input_under_unreadable_folder_does_not_crash(tmp_path):
    locked = tmp_path / "locked"
    (locked / "inner").mkdir(parents=True)
    locked.chmod(0)
    try:
        async def body(app, pilot):
            app.query_one("#set-input_var", Input).value = str(locked / "inner" / "y")
            await pilot.pause(0.6)
            assert "Folder not found" in str(app.query_one("#struct").render())
            app.browse("input_var")                    # picker start dir under it
            await pilot.pause()
            assert isinstance(app.screen, PathPicker)
        run(make_app(tmp_path), body)
    finally:
        locked.chmod(0o755)


def test_picker_unknown_user_home_does_not_crash(tmp_path):
    async def body(app, pilot):
        app.browse("output_var")
        await pilot.pause()
        await submit_path(app, pilot, "~nosuchuser_xyz/x")
        assert isinstance(app.screen, PathPicker)
        assert any("Not a folder" in n.message for n in app._notifications)
    run(make_app(tmp_path), body)


@pytest.mark.parametrize("bad", [
    [1, 2], {"params": None, "stages": []}, {"before": {"fitness": None}},
    {"stages": [{"stage": 1}]},
    {"stages": [dict(ICP_REPORT["stages"][0], trace=[{"iter": 1}])]},
    {"stages": [], "hist_before": {"counts": ["x"]}},
    {"fx": 1},
])
def test_malformed_icp_reports_are_rejected(bad):
    assert icp_report_problem(bad)


def test_valid_icp_report_accepted():
    assert icp_report_problem(ICP_REPORT) is None
    assert icp_report_problem({"stages": []}) is None


def test_malformed_icp_report_shows_error_not_crash(tmp_path):
    rp = tmp_path / "out" / "aligned_cloud" / "icp_report.json"
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps({"stages": [{"stage": 1}]}))

    async def body(app, pilot):
        app.form.set_value("output_var", str(tmp_path / "out"))
        app.action_icp_details()
        await pilot.pause()
        assert not isinstance(app.screen, IcpDetailsScreen)
        assert any("not an ICP report" in n.message for n in app._notifications)
    run(make_app(tmp_path), body)


def test_mismatched_histograms_do_not_crash(tmp_path):
    rep = dict(ICP_REPORT, hist_after={"counts": [1, 2], "hi_vox": 2.0})
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(rep))

    async def body(app, pilot):
        app.show_icp_report(rp)
        await pilot.pause()
        assert isinstance(app.screen, IcpDetailsScreen)
    run(make_app(tmp_path), body)


def test_auto_output_follows_input_until_user_sets_it(tmp_path):
    a, b, c = (tmp_path / n for n in ("scans", "tabletA", "tabletB"))
    for d in (a, b, c):
        d.mkdir()

    async def body(app, pilot):
        field = app.query_one("#set-input_var", Input)
        field.value = str(a)                  # user pauses at a parent folder
        await pilot.pause(0.6)
        assert app.settings["output_var"] == str(tmp_path / "scans_recon")
        field.value = str(b)
        await pilot.pause(0.6)
        assert app.settings["output_var"] == str(tmp_path / "tabletA_recon")
        app.query_one("#set-output_var", Input).value = str(tmp_path / "mine")
        await pilot.pause(0.6)
        field.value = str(c)
        await pilot.pause(0.6)
        assert app.settings["output_var"] == str(tmp_path / "mine")
    run(make_app(tmp_path), body)


def test_saved_log_height_is_clamped_to_small_terminal(tmp_path):
    (tmp_path / "ui.json").write_text(json.dumps({"log_height": 40}))
    app = make_app(tmp_path)

    async def go():
        async with app.run_test(size=(100, 24)) as pilot:
            await pilot.pause(0.2)
            main_h = app.query_one("#main").size.height
            assert app.query_one("#log-wrap").outer_size.height <= main_h - 6
            assert app.query_one("#form").outer_size.height >= 5
    import asyncio
    asyncio.run(go())


def test_final_state_not_lost_when_worker_exits_between_checks(tmp_path, fake_runner):
    async def body(app, pilot):
        app._run_active = True
        app._run_results = {}
        # the worker queued its last event and exited before this drain
        app._events.put(("state", 2, "failed", ""))
        app._drain_events()
        assert not app._run_active
        await pilot.pause()
        assert any("Stage 3 failed" in n.message for n in app._notifications)
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_letter_keys_ignored_in_form_but_ctrl_r_works(tmp_path, two_side_scan, fake_runner):
    fake_runner.block = threading.Event()

    async def body(app, pilot):
        app.form.set_value("input_var", str(two_side_scan))
        app.query_one("#set-hard_mask_var", Checkbox).focus()
        await pilot.press("r")
        await pilot.pause(0.2)
        assert not app.runner.is_running
        await pilot.press("q")
        await pilot.pause(0.1)
        assert app.is_running                         # didn't quit
        await pilot.press("ctrl+r")
        await pilot.pause(0.2)
        assert app.runner.is_running
        await pilot.press("s")                        # guarded too
        await pilot.pause(0.2)
        assert app.runner.is_running
        app.query_one("#stop").focus()                # outside the form (enabled while running)
        await pilot.press("s")
        await wait_idle(app, pilot)
        assert not app.runner.is_running
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_unparseable_number_blocks_run_and_save(tmp_path, fake_runner):
    async def body(app, pilot):
        app.query_one("#set-passes_var", Input).value = ""
        await pilot.pause()
        app.action_run_all()
        app.action_save_defaults()
        await pilot.pause()
        assert not app.runner.is_running
        assert not (tmp_path / "app_defaults.json").exists()
        assert any("Passes" in n.message and n.severity == "error" for n in app._notifications)
        app.query_one("#set-sift_peak_var", Input).value = "abc"
        await pilot.pause()
        assert app.form.invalid_fields() == ["Passes", "Peak threshold"]
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_quoted_path_in_field_is_unquoted(tmp_path):
    async def body(app, pilot):
        app.query_one("#tabs").active = "tab-inputs"
        await pilot.pause()
        inp = app.query_one("#set-output_var", Input)
        inp.focus()
        inp.value = f'"{tmp_path}"'
        await pilot.press("enter")
        await pilot.pause()
        assert app.settings["output_var"] == str(tmp_path)
    run(make_app(tmp_path), body)


def test_viewer_log_lines_are_timestamped(tmp_path, monkeypatch):
    out = tmp_path / "out" / "colmap_side1"
    out.mkdir(parents=True)
    (out / "fused.ply").write_bytes(b"ply")
    monkeypatch.setattr(plat, "has_display", lambda: True)
    monkeypatch.setattr(plat, "launch_viewer", lambda p: subprocess.Popen(
        [sys.executable, "-c", "print('Failed creating OpenGL window')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))

    async def body(app, pilot):
        app.form.set_value("output_var", str(tmp_path / "out"))
        app.view_stage(1)
        for _ in range(100):
            await pilot.pause(0.05)
            if "[viewer]" in log_text(app):
                break
        line = next(ln for ln in log_text(app).splitlines() if "[viewer]" in ln)
        assert re.match(r"\[\d\d:\d\d:\d\d\] \[viewer\] Failed creating OpenGL window", line)
    run(make_app(tmp_path), body)
