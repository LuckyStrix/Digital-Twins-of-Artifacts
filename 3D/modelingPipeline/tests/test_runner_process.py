"""PipelineRunner against real subprocesses: streaming, stop, single-run lock."""

import os
import sys
import threading
import time

import pytest

from pipeline.runner import STAGE_NICENESS, PipelineRunner
from pipeline.settings import Settings

POSIX = os.name == "posix"


def make_runner():
    logs = []
    r = PipelineRunner(Settings.defaults(), on_log=lambda t, k: logs.append((t, k)))
    return r, logs


def test_run_proc_streams_output(tmp_path):
    r, logs = make_runner()
    seen = []
    code = "import sys; print('one'); print('two', file=sys.stderr); print(''); sys.exit(3)"
    rc = r._run_proc([sys.executable, "-c", code], cwd=tmp_path, on_line=seen.append)
    assert rc == 3
    assert logs[0] == (f"$ {sys.executable} -c {code}", "header")
    assert [t for t, k in logs if k == "output"] == ["one", "two", ""]
    assert seen == ["one", "two"]          # blank lines aren't fed to parsers
    assert r._proc is None


def test_run_proc_sets_unbuffered(tmp_path):
    r, logs = make_runner()
    code = "import os; print(os.environ['PYTHONUNBUFFERED'], os.environ.get('X'))"
    r._run_proc([sys.executable, "-c", code], cwd=tmp_path, env={**os.environ, "X": "y"})
    assert ("1 y", "output") in logs


@pytest.mark.skipif(not POSIX, reason="nice is POSIX-only")
def test_run_proc_lowers_cpu_priority(tmp_path):
    r, logs = make_runner()
    code = "import os; print('nice', os.nice(0) - PARENT)".replace("PARENT", str(os.nice(0)))
    r._run_proc([sys.executable, "-c", code], cwd=tmp_path)
    assert logs[0] == (f"$ {sys.executable} -c {code}", "header")   # log shows the real command
    expected = min(STAGE_NICENESS, 19 - os.nice(0))                 # capped at 19
    assert (f"nice {expected}", "output") in logs


def _stop_soon(r, delay=0.5):
    def go():
        deadline = time.time() + 10
        while r._proc is None and time.time() < deadline:
            time.sleep(0.02)
        time.sleep(delay)
        r.stop()
    threading.Thread(target=go, daemon=True).start()


def test_stop_kills_running_process(tmp_path):
    r, logs = make_runner()
    code = "import time\nprint('started', flush=True)\ntime.sleep(60)"
    _stop_soon(r)
    t0 = time.time()
    rc = r._run_proc([sys.executable, "-c", code], cwd=tmp_path)
    assert time.time() - t0 < 10
    assert rc != 0
    assert any(t.endswith("[pipeline] stopped by user") for t, k in logs if k == "info")


def test_stopped_runner_does_not_start_next_process(tmp_path):
    r, logs = make_runner()
    r._stop_req = True
    assert r._run_proc([sys.executable, "-c", "print('x')"], cwd=tmp_path) == -1
    assert not [t for t, k in logs if k == "output"]


@pytest.mark.skipif(not POSIX, reason="process groups are POSIX-only")
def test_stop_kills_grandchildren(tmp_path):
    """Stage 2 runs `bash run.sh`; stopping must also end COLMAP under it."""
    r, logs = make_runner()
    pidfile = tmp_path / "grandchild.pid"
    script = (
        f"sleep 60 & echo $! > {pidfile}; echo started; wait"
    )
    _stop_soon(r)
    t0 = time.time()
    rc = r._run_proc(["bash", "-c", script], cwd=tmp_path)
    # Killing only bash leaves the grandchild holding stdout open, so the
    # run would hang until it exits on its own.
    assert time.time() - t0 < 10
    assert rc != 0
    pid = int(pidfile.read_text())
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {pid} still alive after stop")


def test_single_run_lock(tmp_path):
    release = threading.Event()
    r, _ = make_runner()
    r.execute_stage = lambda idx: release.wait(10)
    assert r.run_stage(0)
    assert r.is_running
    assert not r.run_stage(1)
    assert not r.run_all()
    release.set()
    r.wait(10)
    assert not r.is_running
    assert r.run_all() is True
    r.wait(10)


def test_run_all_threaded_reports_states(tmp_path):
    states = []
    r = PipelineRunner(Settings.defaults(),
                       on_stage_state=lambda i, st, msg: states.append((i, st)))
    assert r.run_all()
    r.wait(10)
    # no input folder: stage 1 fails immediately and the run stops there
    assert states == [(0, "running"), (0, "failed")]
