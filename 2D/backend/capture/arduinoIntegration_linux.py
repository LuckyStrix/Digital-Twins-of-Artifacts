#!/usr/bin/env python3
"""
arduinoIntegration_linux.py
===========================
Linux port of arduinoIntegration.py — the main capture loop. It talks to the
Arduino over serial, triggers the camera for each lighting condition, and
converts the RAW .tmp files to TIFF with dcraw.

Differences from the Windows version:
  - The serial port defaults to a Linux device (/dev/ttyACM0) instead of COM3,
    and can be overridden with the PAPYRUS_SERIAL_PORT environment variable.
    If the default is absent the script auto-detects the first ttyACM*/ttyUSB*.
  - gphoto2 is called directly (it is a native Linux tool) instead of being
    tunnelled through msys2_shell.cmd.
  - The RAW files are archived with Python's shutil instead of the Windows
    `mkdir` / `move` shell commands, so it works with POSIX paths.

Requirements (capture workstation only):
  - Camera connected over USB
  - Arduino on a serial port (/dev/ttyACM0 or /dev/ttyUSB0)
  - pyserial       (pip install pyserial)
  - gphoto2, dcraw available on PATH (sudo apt install gphoto2 dcraw)
"""

import glob
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

import serial

now = datetime.now()

# ── RAW -> TIFF conversion ────────────────────────────────────────────────────
# ONE source of truth: the sidecar written at the end records this exact list,
# so the modeling pipeline always knows which transfer curve the TIFFs carry.
#
#   -T -6     16-bit TIFF out
#   -W        fixed white level (no auto-brighten, so exposure is comparable
#             across frames -- the flat-field and colour fit both depend on this)
#   -g 1 1    LINEAR output. dcraw's default is a BT.709 curve; photometric
#             stereo solves a Lambertian model and the flat-field divide is only
#             a reflectance ratio in linear light, so the curve has to go.
#             (-6 -W -g 1 1 is exactly what dcraw's -4 shorthand means.)
#   -o 0      camera-native primaries, no colour matrix, no gamut clipping.
#             The fitted CCM absorbs the whole camera->sRGB transform, which is
#             the textbook formulation. -o 1 would clip out-of-gamut colour
#             inside dcraw before we ever saw it.
#   -q 0      bilinear demosaic
#   -t 0      no orientation flip
#
# DO NOT ADD -w.  Without it dcraw uses fixed daylight-table multipliers, which
# are deterministic for a given camera model, so one fitted matrix stays valid
# across sessions. -w uses the as-shot camera white balance, which varies per
# frame and would silently invalidate the matrix.
DCRAW_ARGS = ["-T", "-6", "-W", "-g", "1", "1", "-o", "0", "-q", "0", "-t", "0"]

CAPTURE_INFO_NAME = "capture_info.json"

# Serial port: override with PAPYRUS_SERIAL_PORT, otherwise default to the usual
# Arduino Uno device on Linux and fall back to auto-detecting the first
# ttyACM*/ttyUSB* if that default is not present.
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"


def _resolve_serial_port() -> str:
    port = os.environ.get("PAPYRUS_SERIAL_PORT")
    if port:
        return port
    if os.path.exists(DEFAULT_SERIAL_PORT):
        return DEFAULT_SERIAL_PORT
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if candidates:
        print(f"{DEFAULT_SERIAL_PORT} not found; using detected serial port "
              f"{candidates[0]}.")
        return candidates[0]
    return DEFAULT_SERIAL_PORT


SERIAL_PORT = _resolve_serial_port()
ser = serial.Serial(SERIAL_PORT, baudrate=115200, timeout=2.5)
time.sleep(2)


def message_arduino(n, e, s, w, g, b, step, dir):
    msg = ""
    for i in (n, e, s, w, g, b, step, dir):
        msg += str(int(i))
    ser.write(msg.encode())

    while 1:
        response = ser.read(1)
        if response == b'e':
            break


def capture_image(filename):
    tmp = f"{filename}.tmp"
    # On Linux gphoto2 is a native tool, so call it directly (no msys2 shim).
    # We run it synchronously and let its output stream through so its errors are
    # visible in the log, then hand the downloaded .tmp to dcraw.
    subprocess.run(
        ["gphoto2", "--capture-image-and-download", "--filename", tmp],
        cwd=img_dir,
    )

    if not os.path.exists(os.path.join(img_dir, tmp)):
        print(f"WARNING: {tmp} was not created by gphoto2 — skipping dcraw. "
              "Check that the camera is connected and gphoto2 can reach it.")
        print(" ")
        return

    #subprocess.run(["exiftool", "-Orientation=1", "-n", tmp], cwd=img_dir)
    subprocess.run(["dcraw", *DCRAW_ARGS, tmp], cwd=img_dir)
    print(filename + " captured!")
    print(" ")


def _dcraw_version():
    """dcraw's version from its banner, or None. Bare dcraw prints usage to stderr."""
    try:
        out = subprocess.run(["dcraw"], capture_output=True, text=True, timeout=10)
        for line in (out.stderr or out.stdout).splitlines():
            if "Raw photo decoder" in line or line.lower().startswith("dcraw"):
                for tok in line.split():
                    if tok.startswith("v") and any(c.isdigit() for c in tok):
                        return tok.lstrip("v")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _camera_info(cr2_path):
    """(make, model) via `dcraw -i`, or (None, None)."""
    try:
        out = subprocess.run(["dcraw", "-i", "-v", cr2_path],
                             capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            if line.startswith("Camera:"):
                parts = line.split(":", 1)[1].strip().split(None, 1)
                return (parts[0], parts[1] if len(parts) > 1 else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return (None, None)


def write_capture_info(dest_dir, frames, sample_cr2=None):
    """Record how these TIFFs were produced, beside them.

    The pipeline reads `encoding.kind` to decide whether to linearise. A folder
    without this file is assumed BT.709 (dcraw's default), which is exactly
    what every capture made before this file existed actually is -- so old
    scans keep processing correctly.

    Never allowed to fail a capture: the images are the irreplaceable part.
    """
    try:
        linear = "-g" in DCRAW_ARGS and DCRAW_ARGS[DCRAW_ARGS.index("-g") + 1:
                                                   DCRAW_ARGS.index("-g") + 3] == ["1", "1"]
        make, model = _camera_info(sample_cr2) if sample_cr2 else (None, None)
        info = {
            "schema_version": 1,
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "script": os.path.basename(__file__),
            "converter": {"tool": "dcraw", "version": _dcraw_version(),
                          "argv": list(DCRAW_ARGS)},
            "encoding": {
                # `kind` is the only field the pipeline requires.
                "kind": "linear" if linear else "bt709",
                "dcraw_g0": 1.0 if linear else 0.45,
                "dcraw_g1": 1.0 if linear else 4.5,
                "bit_depth": 16,
                "auto_bright": False,               # -W
                "output_colorspace": "raw-camera",  # -o 0
                "white_balance": "dcraw-daylight-table",   # no -w / -a / -A / -r
            },
            "camera": {"make": make, "model": model},
            "frames": list(frames),
        }
        path = os.path.join(dest_dir, CAPTURE_INFO_NAME)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        print(f"Wrote {CAPTURE_INFO_NAME} (encoding: {info['encoding']['kind']})")
    except Exception as exc:                      # never fail a capture over metadata
        print(f"WARNING: could not write {CAPTURE_INFO_NAME}: {exc}")
    print(" ")


if __name__ == "__main__":
    # run.py can direct this capture straight into a specific folder (e.g. a
    # per-side side1/ or side2/ subfolder) via PAPYRUS_CAPTURE_DIR. When unset,
    # fall back to the historic behaviour: make a fresh timestamped folder in
    # the app's top-level data/ folder so the launcher (run.py) can find it.
    capture_dir = os.environ.get("PAPYRUS_CAPTURE_DIR")
    if capture_dir:
        img_dir = capture_dir
        print("Capture folder = " + img_dir)
        print(" ")
    else:
        folder_name = now.strftime("%d-%m-%y_%H-%M-%S")
        print("Folder name = " + folder_name)
        print(" ")
        # This script lives in <app>/backend/capture/, so the app root is three
        # levels up, and the repo root (where data/ lives) is one more.
        app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        img_dir = os.path.join(app_root, "data", folder_name)
    os.makedirs(img_dir, exist_ok=True)

    #MAIN PHOTOGRAPHING LOOP
    #CROSS IMAGES
    message_arduino(1, 1, 1, 1, 0, 1, 0, 1)
    capture_image("allLight")

    message_arduino(1, 0, 0, 0, 0, 1, 0, 1)
    capture_image("ncross")

    message_arduino(0, 1, 0, 0, 0, 1, 0, 1)
    capture_image("ecross")

    message_arduino(0, 0, 1, 0, 0, 1, 0, 1)
    capture_image("scross")

    message_arduino(0, 0, 0, 1, 0, 1, 0, 1)
    capture_image("wcross")

    #ROTATE
    message_arduino(0, 0, 0, 0, 0, 1, 1, 1)

    #CO IMAGES
    message_arduino(1, 0, 0, 0, 0, 1, 0, 0)
    capture_image("nco")

    message_arduino(0, 1, 0, 0, 0, 1, 0, 0)
    capture_image("eco")

    message_arduino(0, 0, 1, 0, 0, 1, 0, 0)
    capture_image("sco")

    message_arduino(0, 0, 0, 1, 0, 1, 0, 0)
    capture_image("wco")

    #ROTATE
    message_arduino(0, 0, 0, 0, 0, 1, 1, 0)
    print("Scanning Complete!")
    print(" ")

    #SORT IMAGES — move the RAW .tmp files into a tmpArchive/ subfolder, leaving
    #the converted .tiff files in the capture folder. Uses shutil so it works
    #with POSIX paths (the Windows version shelled out to mkdir / move).
    archive_dir = os.path.join(img_dir, "tmpArchive")
    os.makedirs(archive_dir, exist_ok=True)
    archived = []
    for tmp_path in glob.glob(os.path.join(img_dir, "*.tmp")):
        dest = os.path.join(archive_dir, os.path.basename(tmp_path))
        shutil.move(tmp_path, dest)
        archived.append(dest)

    #RECORD HOW THESE TIFFS WERE MADE — the pipeline reads this to decide
    #whether to linearise. Written after archiving so a RAW (.tmp) file is available for
    #the camera make/model lookup.
    tiffs = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(img_dir, "*.tiff")))
    write_capture_info(img_dir, tiffs, archived[0] if archived else None)

    print("NOTE: these TIFFs are LINEAR, so they look very dark in an image "
          "viewer (roughly half the brightness you may be used to). That is "
          "correct and expected — the pipeline encodes for display at the end.")
    print(" ")

    #FINISH
    message_arduino(0, 0, 0, 0, 0, 0, 0, 1)
