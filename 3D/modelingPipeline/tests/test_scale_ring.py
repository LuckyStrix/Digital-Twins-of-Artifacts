"""scale/ring.py against rendered turntable scans with a known answer."""

import pytest

from scale.colmap_model import read_model_txt
from scale.ring import RingSpec, fit_ring
from scale_synth import Scene, render

SMALL = dict(raw_w=480, raw_h=320, focal=420.0, steps=10)
TOL_PCT = 0.05


def err_pct(fit, info):
    return abs(fit.scale / info["mm_per_unit"] - 1) * 100


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return render(Scene(**SMALL), tmp_path_factory.mktemp("base"))


@pytest.fixture(scope="module")
def base_fit(base, tmp_path_factory):
    debug = tmp_path_factory.mktemp("debug")
    fit = fit_ring(read_model_txt(base["model"]), base["raw"], base["processed"], RingSpec(),
                   debug_dir=debug, log=lambda *a: None)
    return fit, debug


def test_recovers_scale(base, base_fit):
    fit, debug = base_fit
    assert fit.ok, fit.reason
    assert err_pct(fit, base) < TOL_PCT
    assert fit.sigma_pct < 0.1
    assert abs(fit.width_mm - 8.0) < 0.2
    assert set(fit.per_camera) == {"cam1", "cam2", "cam3"}
    assert abs(fit.off_axis_mm - 3.6) < 0.5          # rendered offset (3, -2)
    assert abs(fit.tilt_deg - 0.5) < 0.2
    assert fit.warnings == [] and fit.n_views >= 20
    assert len(list(debug.glob("*.jpg"))) > 0


def test_hard_case(tmp_path):
    """Distortion, half-size COLMAP images, an EXIF rotation tag on the raw
    photos (must be ignored), stride 2, an off-center tilted ring, and a
    Stage 1 mask that kept the ring."""
    info = render(Scene(**SMALL, distortion=-0.08, image_scale=0.5, exif_orientation=6,
                        stride=2, ring_offset=(12.0, 4.0), ring_tilt_deg=3.5,
                        mask_includes_ring=True), tmp_path)
    fit = fit_ring(read_model_txt(info["model"]), info["raw"], info["processed"], RingSpec(),
                   log=lambda *a: None)
    assert fit.ok, fit.reason
    assert err_pct(fit, info) < TOL_PCT
    assert abs(fit.off_axis_mm - 12.6) < 1.0 and abs(fit.tilt_deg - 3.5) < 0.3
    assert any("from the turntable axis" in w for w in fit.warnings)
    assert any("tilted" in w for w in fit.warnings)
    assert any("kept the ring" in n for n in fit.notes)


def test_no_ring(tmp_path):
    info = render(Scene(**SMALL, ring_present=False), tmp_path)
    fit = fit_ring(read_model_txt(info["model"]), info["raw"], info["processed"], RingSpec(),
                   log=lambda *a: None)
    assert not fit.ok and "no ring found" in fit.reason


def test_wrong_ring_size_is_caught(base):
    fit = fit_ring(read_model_txt(base["model"]), base["raw"], base["processed"],
                   RingSpec(inner_mm=180.0, outer_mm=216.0), log=lambda *a: None)
    assert not fit.ok
    assert "band width" in fit.reason


def test_raw_photos_missing(base, tmp_path):
    fit = fit_ring(read_model_txt(base["model"]), tmp_path, None, RingSpec(), log=lambda *a: None)
    assert not fit.ok and "matched a raw photo" in fit.reason


def test_bad_spec(base):
    fit = fit_ring(read_model_txt(base["model"]), base["raw"], None,
                   RingSpec(inner_mm=216.0, outer_mm=200.0))
    assert not fit.ok and "inner < outer" in fit.reason
