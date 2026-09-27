"""UI-independent core of the reconstruction app.

Nothing in this package may import a UI toolkit (Tk, Textual, ...): the
settings registry, path helpers, progress parsers and stage runner here are
shared by whatever front end drives the pipeline, and are unit-tested
headless.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
ALIGN_DIR = SCRIPT_DIR / "alignment"
DEFAULTS_PATH = SCRIPT_DIR / "app_defaults.json"

# The stage runner imports helpers from the sibling src/ package.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
