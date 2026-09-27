import json

from pipeline import DEFAULTS_PATH
from pipeline.options import REMBG_MODELS
from pipeline.settings import (
    BY_KEY, PERSISTED_KEYS, REGISTRY, TABS, Settings, coerce, hole_reduction_preset,
)


def test_registry_shape():
    assert {s.tab for s in REGISTRY} == set(TABS)
    for s in REGISTRY:
        if s.type == "choice":
            assert s.choices and (s.editable or s.default in s.choices), s.key
        if s.type == "path":
            assert s.path_kind in ("dir", "file"), s.key
            assert not s.persisted, s.key
        assert coerce(s, s.default) == s.default, s.key


def test_all_rembg_models_listed():
    assert len(REMBG_MODELS) == 19
    assert BY_KEY["model_var"].choices == tuple(REMBG_MODELS)


def test_code_defaults_not_saved_defaults():
    d = Settings.defaults()
    assert d["bg_var"] == "white"
    assert d["img_stride_var"] == 3
    assert d["camera_model_var"] == "(COLMAP default)"
    assert d["meta_type_var"] == "tablet"


def test_repo_defaults_file_keys_all_known():
    data = json.loads(DEFAULTS_PATH.read_text())
    assert set(data) <= set(PERSISTED_KEYS)


def test_load_repo_defaults_applies_every_value():
    data = json.loads(DEFAULTS_PATH.read_text())
    s = Settings.load(DEFAULTS_PATH)
    for k, v in data.items():
        assert s[k] == v, k
        assert type(s[k]) is type(v), k


def test_round_trip_repo_defaults(tmp_path):
    s = Settings.load(DEFAULTS_PATH)
    out = tmp_path / "app_defaults.json"
    s.save(out)
    saved = json.loads(out.read_text())
    assert list(saved) == PERSISTED_KEYS
    original = json.loads(DEFAULTS_PATH.read_text())
    assert {k: saved[k] for k in original} == original
    assert Settings.load(out) == s


def test_save_excludes_per_run_fields(tmp_path):
    s = Settings.defaults()
    s["input_var"] = "/data/scan"
    s["meta_name_var"] = "Tablet"
    out = tmp_path / "d.json"
    s.save(out)
    saved = json.loads(out.read_text())
    assert "input_var" not in saved and "meta_name_var" not in saved


def test_load_skips_unknown_bad_and_non_persisted(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({
        "no_such_key": 1,
        "input_var": "/should/not/load",
        "erode_px_var": "not a number",
        "passes_var": 4,
        "bg_var": "purple",
        "model_var": "u2net",
        "use_gpu_var": False,
        "r_density_trim": 1,
        "gpu_index_var": 0,
        "hard_mask_var": [],
    }))
    s = Settings.load(p)
    d = Settings.defaults()
    assert s["input_var"] == ""
    assert s["erode_px_var"] == d["erode_px_var"]
    assert s["passes_var"] == 4
    assert s["bg_var"] == d["bg_var"]
    assert s["model_var"] == "u2net"
    assert s["use_gpu_var"] is False
    assert s["r_density_trim"] == 1.0 and isinstance(s["r_density_trim"], float)
    assert s["gpu_index_var"] == "0"
    assert s["hard_mask_var"] is True
    assert "no_such_key" not in s


def test_load_missing_or_corrupt_file(tmp_path):
    assert Settings.load(tmp_path / "missing.json") == Settings.defaults()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert Settings.load(bad) == Settings.defaults()
    bad.write_text("[1, 2]")
    assert Settings.load(bad) == Settings.defaults()


def test_hole_reduction_presets():
    assert hole_reduction_preset(True) == {
        "r_density_trim": 0.003, "r_poisson_crop_scale": 1.08, "r_normal_max_nn": 80}
    assert hole_reduction_preset(False) == {
        "r_density_trim": 0.02, "r_poisson_crop_scale": 1.05, "r_normal_max_nn": 96}
    # the OFF preset is the plain code defaults
    d = Settings.defaults()
    assert all(d[k] == v for k, v in hole_reduction_preset(False).items())
