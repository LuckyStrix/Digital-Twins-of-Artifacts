"""Headless tests of the Textual app (App.run_test)."""

import asyncio
import json

import pytest

pytest.importorskip("textual")

from textual.widgets import Checkbox, Input, Select, TextArea  # noqa: E402

from pipeline import settings as S  # noqa: E402
from tui.app import ReconApp  # noqa: E402
from tui.widgets import Slider  # noqa: E402

SIZE = (160, 50)


def make_app(tmp_path, **kw):
    kw.setdefault("defaults_path", tmp_path / "app_defaults.json")
    kw.setdefault("ui_state_path", tmp_path / "ui.json")
    return ReconApp(**kw)


def run(app, body):
    async def go():
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await body(app, pilot)
    asyncio.run(go())


def test_form_has_a_field_for_every_setting(tmp_path):
    async def body(app, pilot):
        assert set(app.form.fields) == {s.key for s in S.REGISTRY}
        sel = app.query_one("#set-model_var", Select)
        assert len(sel._options) == 19
        assert isinstance(app.form.fields["meta_desc_text"], TextArea)
        assert isinstance(app.form.fields["seg_scale_var"], Slider)
        assert isinstance(app.form.fields["meta_type_var"], Input)   # editable combobox
    run(make_app(tmp_path), body)


def test_loads_saved_defaults(tmp_path):
    (tmp_path / "app_defaults.json").write_text(json.dumps({"bg_var": "black", "erode_px_var": 20}))

    async def body(app, pilot):
        assert app.settings["bg_var"] == "black"
        assert app.query_one("#set-bg_var", Select).value == "black"
        assert app.query_one("#set-erode_px_var", Input).value == "20"
        assert app.settings["img_stride_var"] == 3        # code default
    run(make_app(tmp_path), body)


def test_edits_update_settings(tmp_path):
    async def body(app, pilot):
        inp = app.query_one("#set-erode_px_var", Input)
        inp.value = "15"
        await pilot.pause()
        assert app.settings["erode_px_var"] == 15
        inp.value = "500"            # above the usual 0-100: accepted, as in Tk
        await pilot.pause()
        assert app.settings["erode_px_var"] == 500
        assert app.form.invalid_fields() == []
        app.query_one("#set-quality_var", Select).value = "low"
        app.query_one("#set-use_gpu_var", Checkbox).value = False
        app.query_one("#set-sift_peak_var", Input).value = "0.01"
        app.query_one("#set-meta_type_var", Input).value = "cylinder seal"
        app.query_one("#set-meta_desc_text", TextArea).text = "Two lines\nof text"
        await pilot.pause()
        assert app.settings["quality_var"] == "low"
        assert app.settings["use_gpu_var"] is False
        assert app.settings["sift_peak_var"] == "0.01"
        assert app.settings["meta_type_var"] == "cylinder seal"
        assert app.settings["meta_desc_text"] == "Two lines\nof text"
    run(make_app(tmp_path), body)


def test_slider_keys(tmp_path):
    async def body(app, pilot):
        app.query_one("#tabs").active = "tab-inputs"
        sl = app.query_one("#set-seg_scale_var", Slider)
        sl.focus()
        await pilot.press("left", "shift+left")
        await pilot.pause()
        assert app.settings["seg_scale_var"] == 89
        await pilot.press("home")
        await pilot.pause()
        assert app.settings["seg_scale_var"] == 10
    run(make_app(tmp_path), body)


def test_hole_reduction_applies_preset(tmp_path):
    async def body(app, pilot):
        app.query_one("#set-r_hole_reduction", Checkbox).value = True
        await pilot.pause()
        assert (app.settings["r_density_trim"], app.settings["r_poisson_crop_scale"],
                app.settings["r_normal_max_nn"]) == (0.003, 1.08, 80)
        assert app.query_one("#set-r_density_trim", Input).value == "0.003"
        app.query_one("#set-r_hole_reduction", Checkbox).value = False
        await pilot.pause()
        assert (app.settings["r_density_trim"], app.settings["r_poisson_crop_scale"],
                app.settings["r_normal_max_nn"]) == (0.02, 1.05, 96)
    run(make_app(tmp_path), body)


def test_hole_reduction_not_applied_on_load(tmp_path):
    (tmp_path / "app_defaults.json").write_text(json.dumps(
        {"r_hole_reduction": True, "r_density_trim": 0.5}))

    async def body(app, pilot):
        await pilot.pause()
        assert app.settings["r_density_trim"] == 0.5
    run(make_app(tmp_path), body)


def test_input_change_sets_output_sides_and_reconciles(tmp_path, two_side_scan):
    session = tmp_path / "scan_recon"
    (session / "processed" / "side1").mkdir(parents=True)
    (session / "processed" / "side1" / "a.jpg").write_bytes(b"x")

    async def body(app, pilot):
        app.form.set_value("side1_var", "")
        app.form.set_value("side2_var", "")
        app.query_one("#set-input_var", Input).value = str(two_side_scan)
        await pilot.pause(0.6)
        assert app.settings["output_var"] == str(session)
        assert (app.settings["side1_var"], app.settings["side2_var"]) == ("side1", "side2")
        assert "Detected sides: side1, side2" in str(app.query_one("#struct").render())
        assert [c.state for c in app.cards] == ["done", "waiting", "waiting", "waiting"]
        assert not app.query_one("#view-0").disabled
    run(make_app(tmp_path), body)


def test_input_change_keeps_existing_output_and_handles_flat(tmp_path):
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "a.png").write_bytes(b"x")

    async def body(app, pilot):
        app.form.set_value("output_var", str(tmp_path / "mine"))
        app.query_one("#set-input_var", Input).value = str(flat)
        await pilot.pause(0.6)
        assert app.settings["output_var"] == str(tmp_path / "mine")
        assert (app.settings["side1_var"], app.settings["side2_var"]) == ("", "")
        assert "Flat structure" in str(app.query_one("#struct").render())
    run(make_app(tmp_path), body)


def test_missing_input_folder_does_not_touch_sides(tmp_path):
    async def body(app, pilot):
        app.query_one("#set-input_var", Input).value = str(tmp_path / "nope")
        await pilot.pause(0.6)
        assert app.settings["output_var"] == ""
        assert app.settings["side1_var"] == "side1"
        assert "Folder not found" in str(app.query_one("#struct").render())
    run(make_app(tmp_path), body)


def test_save_defaults(tmp_path):
    async def body(app, pilot):
        app.query_one("#set-passes_var", Input).value = "4"
        app.query_one("#set-input_var", Input).value = "/somewhere"
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()
        saved = json.loads((tmp_path / "app_defaults.json").read_text())
        assert list(saved) == S.PERSISTED_KEYS
        assert saved["passes_var"] == 4
        assert "input_var" not in saved
    run(make_app(tmp_path), body)


def test_save_defaults_error_is_notified(tmp_path):
    async def body(app, pilot):
        await pilot.press("ctrl+s")
        await pilot.pause()
        assert any(n.severity == "error" for n in app._notifications)
    run(make_app(tmp_path, defaults_path=tmp_path / "missing_dir" / "d.json"), body)


def test_log_resize_and_maximise_persist(tmp_path):
    async def body(app, pilot):
        wrap = app.query_one("#log-wrap")
        h0 = wrap.outer_size.height
        await pilot.press("ctrl+up")
        await pilot.pause()
        assert wrap.outer_size.height == h0 + 2
        assert json.loads((tmp_path / "ui.json").read_text())["log_height"] == h0 + 2
        await pilot.press("l")
        await pilot.pause()
        assert not app.query_one("#form").display
        assert wrap.outer_size.height > h0 + 2
        await pilot.press("l")
        await pilot.pause()
        assert wrap.outer_size.height == h0 + 2
    run(make_app(tmp_path), body)


def test_log_drag_handle(tmp_path):
    async def body(app, pilot):
        handle = app.query_one("#log-handle")
        wrap = app.query_one("#log-wrap")
        y = handle.region.y
        bottom = wrap.region.bottom
        x = handle.region.x + 5
        await pilot.mouse_down("#log-handle")
        await pilot.hover(None, offset=(x, y - 5))      # screen coordinates
        await pilot.mouse_up(None, offset=(x, y - 5))
        await pilot.pause()
        assert wrap.region.bottom == bottom
        assert wrap.outer_size.height == bottom - (y - 5) - 1
        assert handle.region.y == y - 5
        assert json.loads((tmp_path / "ui.json").read_text())["log_height"] == wrap.outer_size.height
    run(make_app(tmp_path), body)


def test_theme_cycles_and_persists(tmp_path):
    async def body(app, pilot):
        assert app.theme == "campbell"
        await pilot.press("t")
        await pilot.pause()
        assert app.theme == "one-half-dark"
        assert json.loads((tmp_path / "ui.json").read_text())["theme"] == "one-half-dark"
    run(make_app(tmp_path), body)


# ── running (fake subprocesses, real stage logic) ─────────────────────────────

import os  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

from pipeline import SCRIPT_DIR  # noqa: E402
from pipeline.runner import PipelineRunner  # noqa: E402

STAGE_OUTPUT = {
    "process_photos.py": ["Processing 2 images", "[1/2] a.jpg", "[2/2] b.jpg", "[done]"],
    "run.sh": ["[step] colmap feature_extractor", "[step] colmap mapper", "dense cloud output: fused.ply"],
    "run.py": ["=== fpfh ===", "RANSAC fitness=0.8", "Saved merged cloud"],
    "reconstruct_mesh.py": ["[step] input", "[step] poisson reconstruction", "gltf output: x.gltf"],
}


class FakeRunner(PipelineRunner):
    """Runs the real stage methods; subprocesses print canned output and
    create the files the next stage needs."""

    block: threading.Event | None = None
    commands: list

    def _run_proc(self, cmd, cwd, env=None, on_line=None):
        cmd = [str(c) for c in cmd]
        self._on_log(f"$ {' '.join(cmd)}", "header")
        FakeRunner.commands.append(cmd)
        if FakeRunner.block is not None:
            while not self._stop_req:
                FakeRunner.block.wait(0.05)
            return -15
        script = next(k for k in STAGE_OUTPUT if any(c.endswith(k) for c in cmd))
        for line in STAGE_OUTPUT[script]:
            self._on_log(line, "output")
            if on_line:
                on_line(line)
        arg = lambda flag: cmd[cmd.index(flag) + 1]
        if script == "process_photos.py":
            Path(arg("--output"), "side1").mkdir(parents=True, exist_ok=True)
            Path(arg("--output"), "side1", "a.jpg").write_bytes(b"x")
        elif script == "run.sh":
            out = Path(SCRIPT_DIR, arg("-o"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "fused.ply").write_bytes(b"ply")
        elif script == "run.py":
            Path(arg("-o")).write_bytes(b"ply")
        else:
            Path(arg("--output")).write_bytes(b"obj")
        return 0


@pytest.fixture
def fake_runner():
    FakeRunner.block = None
    FakeRunner.commands = []
    yield FakeRunner
    FakeRunner.block = None


async def wait_idle(app, pilot, timeout=10.0):
    for _ in range(int(timeout / 0.05)):
        await pilot.pause(0.05)
        if not app.runner.is_running and not app._run_active:
            return
    raise AssertionError("run did not finish")


def log_text(app) -> str:
    return "\n".join(line.text for line in app.query_one("#log").lines)


def test_run_all_through_four_stages(tmp_path, two_side_scan, fake_runner):
    async def body(app, pilot):
        app.form.set_value("input_var", str(two_side_scan))
        app.form.set_value("output_var", str(tmp_path / "out"))
        await pilot.press("r")
        await wait_idle(app, pilot)
        assert [c.state for c in app.cards] == ["done"] * 4
        assert [c.status for c in app.cards] == ["Done"] * 4
        assert not any(app.query_one(f"#view-{i}").disabled for i in range(4))
        assert app.query_one("#stop").disabled and not app.query_one("#run-all").disabled
        text = log_text(app)
        assert "── $ " in text and "process_photos.py" in text
        assert "[stage 3] Aligning fused.ply + fused.ply" in text
        assert "dense cloud output: fused.ply" in text
        assert len(fake_runner.commands) == 5           # stage 2 runs once per side
        assert any("finished" in n.message for n in app._notifications)
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_progress_updates_card(tmp_path, two_side_scan, fake_runner):
    fake_runner.block = threading.Event()

    async def body(app, pilot):
        app.form.set_value("input_var", str(two_side_scan))
        app.form.set_value("output_var", str(tmp_path / "out"))
        app.query_one("#run-1").press()
        await pilot.pause(0.3)
        card = app.cards[1]
        assert card.state == "running" and card.status == "Running…"
        assert app.query_one("#run-all").disabled and not app.query_one("#stop").disabled
        assert app.query_one("#run-1").disabled and not app.query_one("#run-0").disabled
        assert app.query_one("#bar-1").total is None        # indeterminate
        app.runner._on_progress(1, 42, "Feature matching")
        await pilot.pause(0.2)
        assert app.query_one("#bar-1").progress == 42 and card.status == "Feature matching"
        await pilot.press("s")
        await wait_idle(app, pilot)
        assert card.state == "failed"
        assert "[pipeline] stopped by user" in log_text(app)
        assert any(n.severity == "error" for n in app._notifications)
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_busy_message(tmp_path, two_side_scan, fake_runner):
    fake_runner.block = threading.Event()

    async def body(app, pilot):
        app.form.set_value("input_var", str(two_side_scan))
        app.run_stage(0)
        await pilot.pause(0.2)
        app.run_stage(2)
        await pilot.pause(0.1)
        assert any("Another stage is already running" in n.message for n in app._notifications)
        app.action_stop()
        await wait_idle(app, pilot)
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_stage_failure_stops_run_all(tmp_path, two_side_scan, fake_runner):
    async def body(app, pilot):
        app.form.set_value("input_var", "")       # stage 1 fails: no input
        await pilot.press("r")
        await wait_idle(app, pilot)
        assert [c.state for c in app.cards][:2] == ["failed", "waiting"]
        assert "[error] No input directory specified" in log_text(app)
        assert "[pipeline] stopped after stage 1 failed" in log_text(app)
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_quit_while_running_asks_first(tmp_path, two_side_scan, fake_runner):
    fake_runner.block = threading.Event()

    async def body(app, pilot):
        app.form.set_value("input_var", str(two_side_scan))
        app.run_stage(0)
        await pilot.pause(0.2)
        await pilot.press("q")
        await pilot.pause(0.1)
        assert app.screen.__class__.__name__ == "ConfirmScreen"
        await pilot.press("n")
        await pilot.pause(0.1)
        assert app.runner.is_running
        await pilot.press("q")
        await pilot.pause(0.1)
        await pilot.press("y")
        await pilot.pause(0.3)
        assert app.runner._stop_req
    run(make_app(tmp_path, runner_factory=fake_runner), body)


def test_entry_point_is_the_textual_app():
    import importlib.util

    spec = importlib.util.spec_from_file_location("recon_app_entry", SCRIPT_DIR / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # __name__ != "__main__": doesn't start the UI
    from tui.app import main
    assert mod.main is main
    assert "tkinter" not in (SCRIPT_DIR / "app.py").read_text()
