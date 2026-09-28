"""Run records: pipeline.log and pipeline_runs.json in the session folder."""

import json
import time
from pathlib import Path

import pytest

from conftest import RecordingRunner
from pipeline import runlog
from pipeline.runlog import LOG_NAME, RUNS_NAME, ply_vertex_count

OUTPUT = {
    "process_photos.py": ["Processing 2 images", "[1/2] a.jpg", "[2/2] b.jpg", "[done]"],
    "run.sh": ["\x1b[1;31m[step] colmap feature_extractor\x1b[m", "[step] colmap mapper"],
}


def write_ply(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"ply\nformat binary_little_endian 1.0\n"
                     b"element vertex %d\nproperty float x\nend_header\n" % n)


@pytest.fixture
def fast_env(monkeypatch):
    monkeypatch.setattr(runlog, "environment", lambda: {"host": "test"})


@pytest.fixture
def s(settings, two_side_scan, tmp_path):
    settings["input_var"] = str(two_side_scan)
    settings["output_var"] = str(tmp_path / "out")
    settings["side1_var"], settings["side2_var"] = "side1", "side2"
    return settings


def outputs_for_all_stages(out: Path) -> None:
    write_ply(out / "colmap_side1" / "fused.ply", 1000)
    write_ply(out / "colmap_side2" / "fused.ply", 800)
    write_ply(out / "aligned_cloud" / "merged_fpfh.ply", 1700)


def runs(out: Path) -> list[dict]:
    return json.loads((out / RUNS_NAME).read_text())["runs"]


def wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_ply_vertex_count(tmp_path):
    write_ply(tmp_path / "a.ply", 1234)
    assert ply_vertex_count(tmp_path / "a.ply") == 1234
    (tmp_path / "b.ply").write_bytes(b"not a ply")
    assert ply_vertex_count(tmp_path / "b.ply") is None
    assert ply_vertex_count(tmp_path / "missing.ply") is None


def test_run_all_writes_log_and_record(s, tmp_path, fast_env):
    out = tmp_path / "out"
    outputs_for_all_stages(out)
    r = RecordingRunner(s, output=OUTPUT)
    assert r.run_all()
    r.wait(10)
    wait_for(lambda: "environment" in runs(out)[0])

    run, = runs(out)
    assert run["mode"] == "all" and run["result"] == "done"
    assert run["duration_s"] >= 0 and run["finished"]
    assert run["settings"]["input_var"] == s["input_var"]
    assert run["settings"].keys() == s.keys()                  # every setting, persisted or not
    assert run["environment"] == {"host": "test"}
    assert run["inputs"] == {"sides": {"side1": {"images": 1}, "side2": {"images": 1}}}
    assert [st["stage"] for st in run["stages"]] == [1, 2, 3, 4]
    assert all(st["result"] == "done" and st["duration_s"] >= 0 for st in run["stages"])
    assert run["stages"][1]["outputs"] == {"fused_points_side1": 1000, "fused_points_side2": 800}
    assert run["stages"][2]["outputs"] == {"merged_points": 1700}
    assert run["stages"][3]["outputs"] == {"input_points": 1700}
    texts = [step["text"] for step in run["stages"][0]["steps"]]
    assert texts == sorted(set(texts), key=texts.index)       # one entry per change of step
    assert texts[-1] == "Complete"

    log = (out / LOG_NAME).read_text()
    assert "===== run started" in log and "(all) =====" in log
    assert "[stage 1] Background removal" in log
    assert "===== run finished: done" in log
    assert "[[" not in log                                     # runner's [HH:MM:SS] stripped


def test_later_runs_append(s, tmp_path, fast_env):
    out = tmp_path / "out"
    r = RecordingRunner(s, output=OUTPUT)
    for _ in range(2):
        assert r.run_stage(0)
        r.wait(10)
    wait_for(lambda: all("environment" in x for x in runs(out)))
    assert [x["mode"] for x in runs(out)] == ["stage 1", "stage 1"]
    assert len({x["id"] for x in runs(out)}) == 2
    assert (out / LOG_NAME).read_text().count("===== run started") == 2


def test_failed_stage_is_recorded(s, tmp_path, fast_env):
    out = tmp_path / "out"
    r = RecordingRunner(s, rc=1)
    assert r.run_all()
    r.wait(10)
    run, = runs(out)
    assert run["result"] == "failed"
    assert [(st["stage"], st["result"]) for st in run["stages"]] == [(1, "failed")]


def test_stopped_run_is_recorded(s, tmp_path, fast_env):
    out = tmp_path / "out"
    r = RecordingRunner(s, rc=-15)
    orig = r._run_proc

    def stop_then_fail(*a, **kw):
        r.stop()
        return orig(*a, **kw)

    r._run_proc = stop_then_fail
    assert r.run_stage(0)
    r.wait(10)
    run, = runs(out)
    assert run["result"] == "stopped"
    assert run["stages"][0]["result"] == "stopped"
    assert "stopped by user" in (out / LOG_NAME).read_text()


def test_nothing_written_without_folders(settings, tmp_path, monkeypatch, fast_env):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    r = RecordingRunner(settings)
    assert r.run_all()
    r.wait(10)
    assert not (tmp_path / "tablet_recon").exists()


def test_unwritable_session_warns_and_runs(s, tmp_path, fast_env):
    (tmp_path / "blocker").write_text("")
    s["output_var"] = str(tmp_path / "blocker" / "out")   # parent is a file
    r = RecordingRunner(s)
    assert r.run_stage(1)
    r.wait(10)
    assert any("can't write the run log" in m for m in r.info())
    assert r.states == [(1, "running"), (1, "failed")]    # the stage still ran
