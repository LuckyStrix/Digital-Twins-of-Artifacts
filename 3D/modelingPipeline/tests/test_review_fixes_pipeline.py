"""Regression tests for review findings: pipeline layer."""

import os
import subprocess
import sys
import threading
import time

import pytest

from pipeline import platform as plat
from pipeline.runner import PipelineRunner
from pipeline.settings import Settings

POSIX = os.name == "posix"
ROOT_USER = POSIX and os.geteuid() == 0


# ── pipeline layer (no Textual needed) ────────────────────────────────────────

def test_quoted_paths_are_unquoted(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    win = '"C:' + "\\" + 'Users' + "\\" + 'x"'
    assert plat.looks_like_windows_path(win)
    assert plat.looks_like_windows_path("'D:/scans'")
    assert plat.to_posix_path("'/mnt/d/my scans'") == "/mnt/d/my scans"
    assert plat.needs_normalising(' "/a b" ') and not plat.needs_normalising("/a b")
    assert plat.unquote('"') == '"'          # a lone quote isn't a pair


def test_non_finite_floats_rejected_on_load(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('{"r_outlier_std": NaN, "r_dbscan_eps": Infinity, "r_radius_factor": 3.5}')
    s = Settings.load(p)
    d = Settings.defaults()
    assert s["r_outlier_std"] == d["r_outlier_std"]
    assert s["r_dbscan_eps"] == d["r_dbscan_eps"]
    assert s["r_radius_factor"] == 3.5


@pytest.mark.skipif(not POSIX or ROOT_USER, reason="needs POSIX permissions as non-root")
def test_detect_sides_survives_unreadable_parent(tmp_path):
    from pipeline.paths import detect_sides, has_images, is_dir
    locked = tmp_path / "locked"
    (locked / "inner").mkdir(parents=True)
    locked.chmod(0)
    try:
        assert not is_dir(locked / "inner" / "x")
        assert detect_sides(locked / "inner") == []
        assert has_images(locked) is False
    finally:
        locked.chmod(0o755)


def test_runner_snapshots_settings_per_run(tmp_path):
    release = threading.Event()
    live = Settings.defaults()
    live["output_var"] = str(tmp_path / "A")
    r = PipelineRunner(live)
    seen = []

    def fake_stage(idx):
        release.wait(5)
        seen.append(r.paths.session_dir())
        return True

    r.execute_stage = fake_stage
    assert r.run_stage(0)
    live["output_var"] = str(tmp_path / "B")        # edited in the UI mid-run
    release.set()
    r.wait(5)
    assert seen == [tmp_path / "A"]
    assert r.run_stage(0)                            # the next run sees the edit
    r.wait(5)
    assert seen[-1] == tmp_path / "B"


def test_stop_during_process_start_is_not_delayed(tmp_path, monkeypatch):
    """stop() landing between the flag check and self._proc being set used to
    wait until the child printed something."""
    import pipeline.runner as R
    r = PipelineRunner(Settings.defaults())
    real_popen = R.subprocess.Popen

    def popen_then_stop(*a, **kw):
        p = real_popen(*a, **kw)
        r._stop_req = True                           # stop() arrives right now
        return p

    monkeypatch.setattr(R.subprocess, "Popen", popen_then_stop)
    t0 = time.time()
    rc = r._run_proc([sys.executable, "-c", "import time; time.sleep(5); print('late')"],
                     cwd=tmp_path)
    assert time.time() - t0 < 4
    assert rc != 0


@pytest.mark.skipif(not POSIX, reason="process groups are POSIX-only")
def test_shutdown_kills_term_ignoring_children(tmp_path):
    r = PipelineRunner(Settings.defaults())
    pidfile = tmp_path / "pid"
    script = f"trap '' TERM; (trap '' TERM; exec sleep 60) & echo $! > {pidfile}; echo up; wait"
    th = threading.Thread(target=lambda: r._run_proc(["bash", "-c", script], cwd=tmp_path))
    th.start()
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    pid = int(pidfile.read_text())
    r.shutdown(grace=0.5)
    th.join(5)
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("TERM-ignoring grandchild survived shutdown()")
