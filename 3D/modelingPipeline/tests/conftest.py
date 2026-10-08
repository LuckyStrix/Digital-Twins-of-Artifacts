import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline.runner import PipelineRunner  # noqa: E402
from pipeline.settings import Settings  # noqa: E402


class RecordingRunner(PipelineRunner):
    """PipelineRunner that records subprocess calls instead of running them.

    ``output`` maps a stage-script name fragment to lines fed to on_line, and
    ``rc`` is the return code every call reports.
    """

    def __init__(self, settings, rc=0, output=None):
        self.calls: list[tuple[list[str], str, dict | None]] = []
        self.logs: list[tuple[str, str]] = []
        self.states: list[tuple[int, str]] = []
        self.progress: list[tuple[int, int, str]] = []
        self.rc = rc
        self.output = output or {}
        super().__init__(
            settings,
            on_log=lambda text, kind: self.logs.append((text, kind)),
            on_stage_state=lambda idx, state, status: self.states.append((idx, state)),
            on_progress=lambda idx, pct, text: self.progress.append((idx, pct, text)),
        )

    def _run_proc(self, cmd, cwd, env=None, on_line=None):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, str(cwd), env))
        for frag, lines in self.output.items():
            if any(frag in c for c in cmd):
                for line in lines:
                    if on_line:
                        on_line(line)
        return self.rc

    def info(self) -> list[str]:
        """App log messages without their timestamps."""
        return [t[11:] for t, k in self.logs if k == "info"]


@pytest.fixture(autouse=True)
def no_native_dialogs(monkeypatch):
    """Never pop up real Windows pickers from tests run under WSL."""
    from pipeline import platform as plat
    monkeypatch.setattr(plat, "native_folder_dialog", lambda *a, **k: None)
    monkeypatch.setattr(plat, "native_file_dialog", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def no_inherited_fipmesh_env(monkeypatch):
    """Drop FIPMESH_* variables from the calling shell (e.g. FIPMESH_COLMAP_BIN,
    which BUILDING_COLMAP.md suggests exporting and the Docker image sets), so
    the golden env checks see only what the runner itself adds."""
    import os
    for key in [k for k in os.environ if k.startswith("FIPMESH_")]:
        monkeypatch.delenv(key)


@pytest.fixture
def settings():
    return Settings.defaults()


@pytest.fixture
def two_side_scan(tmp_path):
    """input/side1, input/side2 with one image each; output = tmp/out."""
    inp = tmp_path / "scan"
    for side in ("side1", "side2"):
        (inp / side).mkdir(parents=True)
        (inp / side / "img_001.JPG").write_bytes(b"x")
    return inp
