# Capture stage

This is the first stage of the full Papyrus pipeline. It drives the physical
capture rig — a DSLR (via `gphoto2`) and an Arduino-controlled lighting/rotation
stage — to shoot the directional-light TIFFs the modeling pipeline needs.

## Contents

- `arduinoIntegration.py` — main capture loop. Talks to the Arduino over serial,
  triggers the camera for each lighting condition, and converts the RAW `.cr2`
  files to TIFF with `dcraw`.
- `serialTesting.py` — tiny helper to sanity-check the serial connection.
- `IrisArduinoCode/IrisArduinoCode.ino` — firmware for the Arduino controlling
  the lights, aperture and rotation stage.

## Requirements (capture workstation only)

Capture only runs on the machine wired to the rig:

- Camera connected over USB
- Arduino on a serial port (the script opens `COM3` — change this for your OS;
  e.g. `/dev/ttyUSB0` or `/dev/ttyACM0` on Linux)
- `pyserial`  → `pip install pyserial`
- `gphoto2` and `dcraw` available on PATH (the script shells out to them)

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
