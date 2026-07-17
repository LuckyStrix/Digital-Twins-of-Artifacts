# 2D — Papyrus Pipeline

Photometric-stereo pipeline for flat manuscripts (papyrus, and any object that
needs fine surface detail but no real depth). A single Tkinter launcher ties the
sub-stages together. See the [repo setup guide](../SETUP.md) for first-time
install.

There are two builds of the launcher that share the same modeling and rendering
code and differ only in how they drive the capture rig:

- **`run.py`** — Windows build (gphoto2 through msys2, serial port `COM3`).
- **`run_linux.py`** — Linux build (native gphoto2/dcraw on PATH, serial port
  `/dev/ttyACM0`). See [Linux](#linux) below.

## Prerequisites

### Windows

- **OS: Windows** (the capture rig and launcher assume Windows paths/tools).
- **Python 3.9–3.12** with tkinter (install Python with the Tcl/Tk option, and
  tick "Add python.exe to PATH" in the installer). Get it from
  https://www.python.org/downloads/. Do **not** use 3.13/3.14 — the pinned
  `numpy<2.0` (and scipy/opencv/rembg) have no prebuilt wheels there and pip will
  try to compile numpy and fail.
- **Node.js LTS** on PATH — needed for the "Build 3D Model" (rendering) step.
  Get it from https://nodejs.org/ (tick "Add to PATH").
- **Capture-rig tools** (only if you run the hardware capture / focus viewer):
  - **msys2** with gphoto2 — https://www.msys2.org/ , then in the MSYS2 MINGW64
    shell: `pacman -S mingw-w64-x86_64-gphoto2`. Add the MSYS2 `mingw64\bin`
    folder to your PATH so `gphoto2` is callable from the launcher.
  - **dcraw** on PATH (converts `.cr2` → `.tiff`). Windows binary:
    https://sourceforge.net/app/dcraw/ (download `DCRaw_V9.28.exe` and put it on
    your PATH).
  - **Zadig** — camera USB driver swap, **required on Windows for gphoto2 to see
    the DSLR**. See [Camera driver setup (Zadig)](#camera-driver-setup-zadig-windows-only)
    below.

> **Putting tools on PATH:** several tools above must be on your PATH so the
> launcher (and MSYS2) can find them. If you're not sure how, follow this guide:
> https://www.howtogeek.com/787217/how-to-edit-environment-variables-on-windows-10-or-11/

#### Camera driver setup (Zadig, Windows only)

On Windows, `gphoto2`/libusb can't talk to the DSLR until the camera's USB
driver is replaced with **WinUSB** using [Zadig](https://zadig.akeo.ie/)
(download the standalone `.exe` — no install needed):

1. Connect the camera over USB and switch it **on** (put it in PC/PTP mode if it
   has one). Close any Canon/vendor software (e.g. EOS Utility) that may grab it.
2. Run Zadig, then **Options → List All Devices**.
3. In the dropdown, select your camera (e.g. "Canon Digital Camera").
4. Set the target driver to **WinUSB** and click **Replace Driver**
   (or **Install Driver**).
5. Confirm with `gphoto2 --auto-detect` in the MSYS2 MINGW64 shell — the camera
   should now be listed.

> To go back to using the camera with its normal vendor software, uninstall the
> WinUSB driver from Windows Device Manager (or reinstall the vendor driver).

### Linux

- **Python 3.9–3.12** with tkinter — on Debian/Ubuntu:
  `sudo apt install python3-tk`. Do **not** use 3.13/3.14 (no prebuilt wheels
  for the pinned `numpy<2.0` and friends).
- **Node.js LTS** on PATH — needed for the "Build 3D Model" (rendering) step.
- **Capture-rig tools** (only if you run the hardware capture / focus viewer):
  - **gphoto2** and **dcraw** on PATH — `sudo apt install gphoto2 dcraw`
    (they are called directly, no msys2 needed).
  - Serial access to the Arduino: the scripts default to `/dev/ttyACM0`
    (override with the `PAPYRUS_SERIAL_PORT` env var). Add yourself to the
    `dialout` group for permission: `sudo usermod -a -G dialout $USER`, then
    log out and back in.

## Setup

`run.py` self-installs most dependencies. Launch it and click
**Install Python Dependencies**, which runs `pip install -r
backend/modeling/requirements.txt pyserial` and `npm install` in the renderer
for you:

```bash
python run.py
```

The only things it can't install are Node.js, gphoto2 (msys2 on Windows), and
dcraw — see Prerequisites above. To install the Python deps manually instead, see
[`backend/modeling/README.md`](backend/modeling/README.md).

## Usage

Run `python run.py` (Windows) or `python3 run_linux.py` (Linux) and click the
buttons top to bottom (select working image set → capture → run modeling → build
model → open viewer). The active working folder holds one scan set and receives
all pipeline output, so each set is self-contained. Full flow is documented in
the module docstring at the top of the launcher.

### Linux

The Linux build is `run_linux.py`. It is the same launcher as `run.py` and runs
the identical modeling and rendering stages; the only difference is the capture
scripts it invokes (`backend/capture/arduinoIntegration_linux.py` and
`focusViewer_linux.py`), which call gphoto2/dcraw natively and default to the
`/dev/ttyACM0` serial port:

```bash
python3 run_linux.py
```

## Structure

```
run.py                                Windows launcher + dependency installer
run_linux.py                          Linux launcher (native gphoto2/dcraw)
backend/capture/                      DSLR + Arduino capture rig — see backend/capture/README.md
  arduinoIntegration.py               Windows capture loop (gphoto2 via msys2)
  arduinoIntegration_linux.py         Linux capture loop (native gphoto2)
  focusViewer.py / focusViewer_linux.py   Windows / Linux focus viewers
backend/modeling/                     Photometric-stereo pipeline — see backend/modeling/README.md
backend/rendering/                    Three.js / Vite GLB baker  — see backend/rendering/README.md
```
