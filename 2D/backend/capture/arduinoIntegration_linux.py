#!/usr/bin/env python3
"""
arduinoIntegration_linux.py
===========================
Linux port of arduinoIntegration.py — the main capture loop. It talks to the
Arduino over serial, triggers the camera for each lighting condition, and
converts the RAW .cr2 files to TIFF with dcraw.

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
import os
import shutil
import subprocess
import time
from datetime import datetime

import serial

now = datetime.now()

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
    cr2 = f"{filename}.cr2"
    # On Linux gphoto2 is a native tool, so call it directly (no msys2 shim).
    # We run it synchronously and let its output stream through so its errors are
    # visible in the log, then hand the downloaded .cr2 to dcraw.
    subprocess.run(
        ["gphoto2", "--capture-image-and-download", "--filename", cr2],
        cwd=img_dir,
    )

    if not os.path.exists(os.path.join(img_dir, cr2)):
        print(f"WARNING: {cr2} was not created by gphoto2 — skipping dcraw. "
              "Check that the camera is connected and gphoto2 can reach it.")
        print(" ")
        return

    #subprocess.run(["exiftool", "-Orientation=1", "-n", cr2], cwd=img_dir)
    subprocess.run(
        ["dcraw", "-T", "-6", "-W", "-o", "0", "-q", "0", "-t", "0", cr2],
        cwd=img_dir,
    )
    print(filename + " captured!")
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

    #SORT IMAGES — move the RAW .cr2 files into a cr2Archive/ subfolder, leaving
    #the converted .tiff files in the capture folder. Uses shutil so it works
    #with POSIX paths (the Windows version shelled out to mkdir / move).
    archive_dir = os.path.join(img_dir, "cr2Archive")
    os.makedirs(archive_dir, exist_ok=True)
    for cr2_path in glob.glob(os.path.join(img_dir, "*.cr2")):
        shutil.move(cr2_path, os.path.join(archive_dir, os.path.basename(cr2_path)))

    #FINISH
    message_arduino(0, 0, 0, 0, 0, 0, 0, 1)
