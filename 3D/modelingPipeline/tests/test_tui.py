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


def test_edits_update_settings_and_invalid_values_are_ignored(tmp_path):
    async def body(app, pilot):
        inp = app.query_one("#set-erode_px_var", Input)
        inp.value = "15"
        await pilot.pause()
        assert app.settings["erode_px_var"] == 15
        inp.value = "500"            # above max 100
        await pilot.pause()
        assert app.settings["erode_px_var"] == 15
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
