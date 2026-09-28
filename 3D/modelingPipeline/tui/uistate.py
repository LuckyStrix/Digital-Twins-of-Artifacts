"""Per-user UI preferences (log height, theme, recent folders, ...).

Kept apart from app_defaults.json, which holds pipeline settings only and is
shared with anything else that reads it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from pipeline import SCRIPT_DIR

UI_STATE_PATH = SCRIPT_DIR / ".tui_state.json"
MAX_RECENTS = 10


@dataclass
class UiState:
    log_height: int = 12
    theme: str = "campbell"
    recent_dirs: list[str] = field(default_factory=list)
    native_dialog: bool = True      # under WSL, open the Windows picker first

    @classmethod
    def load(cls, path: Path = UI_STATE_PATH) -> "UiState":
        st = cls()
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return st
        if not isinstance(data, dict):
            return st
        for f in fields(cls):
            if f.name in data and isinstance(data[f.name], type(getattr(st, f.name))):
                setattr(st, f.name, data[f.name])
        return st

    def save(self, path: Path = UI_STATE_PATH) -> None:
        try:
            Path(path).write_text(json.dumps(asdict(self), indent=2))
        except OSError:
            pass

    def add_recent(self, d: str) -> None:
        d = str(d)
        self.recent_dirs = [d] + [r for r in self.recent_dirs if r != d]
        del self.recent_dirs[MAX_RECENTS:]
