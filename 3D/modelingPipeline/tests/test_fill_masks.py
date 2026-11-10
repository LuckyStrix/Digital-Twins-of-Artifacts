"""Stage 2 fusion masks made from Stage 1's background fill."""

import importlib.util

import numpy as np
import pytest

from pipeline import SCRIPT_DIR
from pipeline.runner import colmap_env

cv2 = pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def mvs():
    spec = importlib.util.spec_from_file_location("run_colmap_mvs_fill",
                                                  SCRIPT_DIR / "src" / "run_colmap_mvs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _image(fill: int) -> np.ndarray:
    """100x100 fill with a 40x40 tablet (gray 120) at 30..69, and the
    undistorter's pure-black border on the left."""
    im = np.full((100, 100, 3), fill, np.uint8)
    im[30:70, 30:70] = 120
    im[:, :3] = 0
    return im


@pytest.mark.parametrize("fill, value", [("black", 0), ("white", 255)])
def test_fill_masks_drop_fill_and_shrink(tmp_path, mvs, fill, value):
    ws = tmp_path / "dense"
    (ws / "images" / "cam1").mkdir(parents=True)
    cv2.imwrite(str(ws / "images" / "cam1" / "0001.jpg"), _image(value))
    out = mvs._prepare_fill_masks(ws, fill, 4)
    m = cv2.imread(str(out / "cam1" / "0001.jpg.png"), cv2.IMREAD_GRAYSCALE) > 0
    assert not m[:, :3].any()                       # undistortion border
    assert not m[10, 50] and not m[50, 90]          # fill
    assert m[50, 50]                                # tablet center
    assert not m[31, 50] and not m[50, 68]          # within 4 px of the outline
    assert m[36, 50] and m[50, 63]


def test_fill_masks_combine_with_stage1_masks(tmp_path, mvs):
    ws = tmp_path / "dense"
    (ws / "images").mkdir(parents=True)
    cv2.imwrite(str(ws / "images" / "a.png"), _image(0))
    base = tmp_path / "base"
    base.mkdir()
    b = np.zeros((100, 100), np.uint8)
    b[:, :50] = 255                                 # Stage 1 keeps only the left half
    cv2.imwrite(str(base / "a.png.png"), b)
    m = cv2.imread(str(mvs._prepare_fill_masks(ws, "black", 2, base) / "a.png.png"), 0) > 0
    assert m[50, 40] and not m[50, 60]


def test_fill_mask_env(settings):
    s = dict(settings, bg_var="black", fusion_mask_erode_var=6)
    env = colmap_env(s, base_env={})
    assert (env["FIPMESH_COLMAP_FUSION_MASK_FILL"], env["FIPMESH_COLMAP_FUSION_MASK_ERODE"]) == ("black", "6")
    assert colmap_env(dict(s, fusion_mask_erode_var=0), base_env={})["FIPMESH_COLMAP_FUSION_MASK_FILL"] == "none"
    assert colmap_env(dict(s, bg_var="transparent"), base_env={})["FIPMESH_COLMAP_FUSION_MASK_FILL"] == "none"


def test_fill_mask_config_defaults_off(mvs, monkeypatch):
    monkeypatch.delenv("FIPMESH_COLMAP_FUSION_MASK_FILL", raising=False)
    monkeypatch.setenv("FIPMESH_COLMAP_FUSION_MASK_ERODE", "3")
    assert mvs._fusion_fill_mask_config() == ("none", 3)
    monkeypatch.setenv("FIPMESH_COLMAP_FUSION_MASK_FILL", "White")
    assert mvs._fusion_fill_mask_config() == ("white", 3)
