# Capture stage

This is the first stage of the full Papyrus pipeline. It drives the physical
capture rig — a DSLR (via `gphoto2`) and an Arduino-controlled lighting/rotation
stage — to shoot the directional-light TIFFs the modeling pipeline needs.

## Contents

- `arduinoIntegration.py` / `arduinoIntegration_linux.py` — main capture loop
  (Windows / Linux). Talks to the Arduino over serial and triggers the camera
  for each lighting condition. Once the whole sequence has been shot and the
  polariser is back in place, it converts the RAW files to TIFF with `dcraw`,
  one at a time. Conversion is deliberately kept out of the shooting sequence,
  so the rig only does one thing at a time.
- `focusViewer.py` / `focusViewer_linux.py` — live preview for setting focus.
- `IrisArduinoCode/IrisArduinoCode.ino` — firmware for the Arduino controlling
  the lights, aperture and rotation stage.

## Requirements (capture workstation only)

Capture only runs on the machine wired to the rig:

- Camera connected over USB
- Arduino on a serial port: `COM3` on Windows, `/dev/ttyACM0` on Linux (or the
  first `ttyACM*`/`ttyUSB*` found). Set `PAPYRUS_SERIAL_PORT` to use another
  port on either.
- `pyserial`  → `pip install pyserial`
- `gphoto2` and `dcraw` available on PATH (the script shells out to them).
  gphoto2 on Windows comes from [msys2](https://www.msys2.org/)
  (`pacman -S mingw-w64-x86_64-gphoto2`); the dcraw Windows binary is at
  https://sourceforge.net/app/dcraw/. Not sure how to put a tool on PATH? See
  [this guide](https://www.howtogeek.com/787217/how-to-edit-environment-variables-on-windows-10-or-11/).

> **Windows camera driver (Zadig):** on Windows, `gphoto2`/libusb cannot see the
> DSLR until you replace its USB driver with **WinUSB** using
> [Zadig](https://zadig.akeo.ie/) — Options → List All Devices → select the
> camera → set **WinUSB** → **Replace Driver**. Full walkthrough in the
> [2D README](../../README.md#camera-driver-setup-zadig-windows-only). (Not
> needed on Linux, where gphoto2 uses the kernel USB driver directly.)

## Colour handling

Frames are converted to **linear** 16-bit TIFFs (`dcraw -g 1 1 -o 0 ...`), so
they look dark in an image viewer; that is expected. Each capture folder also
gets a `capture_info.json` recording the dcraw flags, which the modeling
pipeline reads to decide whether to linearise. The Windows and Linux capture
scripts share the same flags and sidecar. Setting `PAPYRUS_CAPTURE_DIR` makes
the script write straight into that folder (no timestamped subfolder).

## Output

Captures are written to the app's top-level `data/<timestamp>/` folder (the RAW
files are moved into a `tmpArchive/` subfolder; the converted `.tiff`
files stay in the timestamp folder). Each such folder is a self-contained scan
set: modeling adds a `maps/` subfolder and the build writes `<folder>.glb` next
to it.
Use the launcher's **Select working image set** button to pick which scan folder
to process.

> Note: the only change from the standalone `P - Capture/arduinoIntegration.py`
> is that the output directory is now the app's top-level `data/` folder
> (computed relative to this script) instead of a hard-coded absolute path, so
> the launcher can find the captures.

Calibration images (flat copy paper, same filenames) are shot separately by the
launcher's Step 0a into `<working folder>/cal/` (and copied to the global
fallback `backend/calibration/`).
