#!/usr/bin/env python3
"""
run.py
======
Windows build of the Papyrus pipeline launcher.

    python run.py          (or: py -3.12 run.py)

The launcher itself (every step, the UI, the dependency check) lives in
pipeline_app.py and is shared with the Linux build, run_linux.py. This file
only fills in what differs on Windows: the capture scripts it runs
(arduinoIntegration.py / focusViewer.py, which reach gphoto2 through msys2 and
use COM3 unless PAPYRUS_SERIAL_PORT is set), npm's .cmd name, the virtualenv
layout, and the capture-rig checks and install hints.
"""

import os
import shutil
from pathlib import Path

import pipeline_app as app
from pipeline_app import CAPTURE_DIR

# Keep in sync with arduinoIntegration.py / focusViewer.py.
MSYS2_ROOT   = Path(r"C:\msys64")
MSYS2_SHELL  = MSYS2_ROOT / "msys2_shell.cmd"
MSYS2_GPHOTO = MSYS2_ROOT / "mingw64" / "bin" / "gphoto2.exe"
DEFAULT_SERIAL_PORT = "COM3"


class WindowsPipelineApp(app.PipelineApp):
    PLATFORM_NAME  = "Windows"
    CAPTURE_SCRIPT = CAPTURE_DIR / "arduinoIntegration.py"
    FOCUS_SCRIPT   = CAPTURE_DIR / "focusViewer.py"
    # npm ships as npm.cmd (a batch script); Popen without shell=True cannot
    # launch a bare "npm" (CreateProcess doesn't resolve .cmd) -> [WinError 2].
    NPM            = "npm.cmd"
    PYTHON_HINT    = "py -3.12 run.py"

    def _venv_python(self, venv: Path) -> Path:
        return venv / "Scripts" / "python.exe"

    def _venv_create_hint(self) -> str:
        return ("Make sure this Python was installed from python.org with the "
                "standard library intact, then try again.")

    def _check_capture_tools(self, ok: str, no: str) -> list:
        problems = []
        if MSYS2_SHELL.exists():
            self.log(f"  {ok} msys2 ({MSYS2_SHELL})")
            if MSYS2_GPHOTO.exists():
                self.log(f"  {ok} gphoto2 inside msys2 ({MSYS2_GPHOTO})")
            else:
                self.log(f"  {no} gphoto2 not installed in msys2 — in the MSYS2 "
                         "MINGW64 shell run: pacman -S mingw-w64-x86_64-gphoto2")
                problems.append("gphoto2 not installed in msys2")
        else:
            self.log(f"  {no} msys2 not found at {MSYS2_SHELL} — install it from "
                     "https://www.msys2.org/ , then in the MSYS2 MINGW64 shell: "
                     "pacman -S mingw-w64-x86_64-gphoto2")
            problems.append("msys2 + gphoto2 not installed")

        dcraw = shutil.which("dcraw")
        if dcraw:
            self.log(f"  {ok} dcraw ({dcraw})")
        else:
            self.log(f"  {no} dcraw not found on PATH (converts the camera's RAW "
                     "files to TIFF) — download it and add its folder to PATH")
            problems.append("dcraw not on PATH")

        self.log("       The camera also needs the WinUSB driver (Zadig) — see the "
                 "2D README; this check cannot see that.")

        port = os.environ.get("PAPYRUS_SERIAL_PORT") or DEFAULT_SERIAL_PORT
        ports = self._serial_ports()
        if ports is None:
            self.log(f"  {no} cannot list serial ports (pyserial missing)")
            problems.append("cannot check the Arduino port (pyserial missing)")
        elif port.upper() in (p.upper() for p in ports):
            self.log(f"  {ok} Arduino serial port {port}")
        else:
            seen = ", ".join(ports) if ports else "none"
            self.log(f"  {no} Arduino not found on {port} (ports present: {seen}). "
                     "Plug it in, flashed with backend/capture/IrisArduinoCode; if "
                     "it shows up on another COM port, set PAPYRUS_SERIAL_PORT to it.")
            problems.append(f"Arduino not found on {port}")
        return problems


if __name__ == "__main__":
    app.main(WindowsPipelineApp)
