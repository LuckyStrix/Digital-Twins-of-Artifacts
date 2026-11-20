#!/usr/bin/env python3
"""
run_linux.py
============
Linux build of the Papyrus pipeline launcher.

    python3 run_linux.py

The launcher itself (every step, the UI, the dependency check) lives in
pipeline_app.py and is shared with the Windows build, run.py. This file only
fills in what differs on Linux: the capture scripts it runs
(arduinoIntegration_linux.py / focusViewer_linux.py, which call gphoto2/dcraw
natively and default to /dev/ttyACM0, overridable with PAPYRUS_SERIAL_PORT),
the virtualenv layout, and the capture-rig checks and install hints.
"""

import glob
import grp
import os
import shutil
import sys
from pathlib import Path

import pipeline_app as app
from pipeline_app import CAPTURE_DIR

DEFAULT_SERIAL_PORT = "/dev/ttyACM0"   # keep in sync with the *_linux.py capture scripts


class LinuxPipelineApp(app.PipelineApp):
    PLATFORM_NAME  = "Linux"
    CAPTURE_SCRIPT = CAPTURE_DIR / "arduinoIntegration_linux.py"
    FOCUS_SCRIPT   = CAPTURE_DIR / "focusViewer_linux.py"
    NPM            = "npm"
    PYTHON_HINT    = "python3.12 run_linux.py"

    def _venv_python(self, venv: Path) -> Path:
        return venv / "bin" / "python"

    def _venv_create_hint(self) -> str:
        v = sys.version_info
        return ("On Debian/Ubuntu this usually means the venv module is not "
                f"installed: sudo apt install python{v.major}.{v.minor}-venv")

    def _check_capture_tools(self, ok: str, no: str) -> list:
        problems = []
        for tool, why in (("gphoto2", "talks to the camera"),
                          ("dcraw", "converts the camera's RAW files to TIFF")):
            path = shutil.which(tool)
            if path:
                self.log(f"  {ok} {tool} ({path})")
            else:
                self.log(f"  {no} {tool} not found on PATH ({why}) — "
                         f"sudo apt install {tool}")
                problems.append(f"{tool} not installed")

        # Same port resolution as the capture scripts.
        port = os.environ.get("PAPYRUS_SERIAL_PORT")
        if not port:
            found = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
            port = (DEFAULT_SERIAL_PORT if os.path.exists(DEFAULT_SERIAL_PORT)
                    else (found[0] if found else None))
        if not port or not os.path.exists(port):
            self.log(f"  {no} Arduino not found (no {port or '/dev/ttyACM*'}) — "
                     "plug it in, flashed with backend/capture/IrisArduinoCode. "
                     "Set PAPYRUS_SERIAL_PORT if it appears under another name.")
            problems.append("Arduino not connected")
        elif os.access(port, os.R_OK | os.W_OK):
            self.log(f"  {ok} Arduino serial port {port}")
        else:
            try:
                group = grp.getgrgid(os.stat(port).st_gid).gr_name
            except (KeyError, OSError):
                group = "dialout"
            self.log(f"  {no} no permission to open {port}. Add yourself to the "
                     f"'{group}' group: sudo usermod -a -G {group} $USER, "
                     "then log out and back in.")
            problems.append(f"no permission for {port}")
        return problems


if __name__ == "__main__":
    app.main(LinuxPipelineApp)
