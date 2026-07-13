# Capture stage

This is the first stage of the full Papyrus pipeline. It drives the physical
capture rig — a DSLR (via `gphoto2`) and an Arduino-controlled lighting/rotation
stage — to shoot the directional-light TIFFs the modeling pipeline needs.

## Contents

- `arduinoIntegration.py` — main capture loop (**Windows**). Talks to the Arduino
  over serial, triggers the camera for each lighting condition (gphoto2 via
  msys2), and converts the RAW `.cr2` files to TIFF with `dcraw`.
- `arduinoIntegration_linux.py` — **Linux** port of the same capture loop. Calls
  gphoto2/dcraw natively and defaults to the `/dev/ttyACM0` serial port.
- `focusViewer.py` / `focusViewer_linux.py` — Windows / Linux focus-check helpers
  (live-view preview so the operator can set focus before a capture).
- `serialTesting.py` — tiny helper to sanity-check the serial connection.
- `IrisArduinoCode/IrisArduinoCode.ino` — firmware for the Arduino controlling
  the lights, aperture and rotation stage.

## Requirements (capture workstation only)

Capture only runs on the machine wired to the rig:

- Camera connected over USB
- Arduino on a serial port:
  - **Windows:** `arduinoIntegration.py` opens `COM3`.
  - **Linux:** `arduinoIntegration_linux.py` defaults to `/dev/ttyACM0`
    (or `/dev/ttyUSB0` for CH340-based boards). Override either with the
    `PAPYRUS_SERIAL_PORT` environment variable.
- `pyserial`  → `pip install pyserial`
- `gphoto2` and `dcraw` available on PATH (the scripts shell out to them). On
  Windows gphoto2 runs through msys2; on Linux install them natively
  (`sudo apt install gphoto2 dcraw`).

## Output

Captures are written to the app's top-level `data/<timestamp>/` folder (the RAW
`.cr2` files are moved into a `cr2Archive/` subfolder; the converted `.tiff`
files stay in the timestamp folder). Each such folder is a self-contained scan
set: the modeling and rendering stages add `maps/` and `model/` subfolders to it.
Use the launcher's **Select working image set** button to pick which scan folder
to process.

> Note: the only change from the standalone `P - Capture/arduinoIntegration.py`
> is that the output directory is now the app's top-level `data/` folder
> (computed relative to this script) instead of a hard-coded absolute path, so
> the launcher can find the captures.

Calibration images (flat copy paper, same filenames) are shot separately and go
in `backend/calibration/`.
