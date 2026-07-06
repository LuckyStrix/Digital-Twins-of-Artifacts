# 2D — Papyrus Pipeline

Photometric-stereo pipeline for flat manuscripts (papyrus, and any object that
needs fine surface detail but no real depth). A single Tkinter launcher,
`run.py`, ties the sub-stages together. See the
[repo setup guide](../SETUP.md) for first-time install.

## Prerequisites

- **OS: Windows** (the capture rig and launcher assume Windows paths/tools).
- **Python 3.9+** with tkinter (install Python with the Tcl/Tk option).
- **Node.js LTS** on PATH — needed for the "Build 3D Model" (rendering) step.
  Get it from https://nodejs.org/ (tick "Add to PATH").
- **Capture-rig tools** (only if you run the hardware capture / focus viewer):
  - **msys2** with gphoto2 — https://www.msys2.org/ , then in the MSYS2 MINGW64
    shell: `pacman -S mingw-w64-x86_64-gphoto2`
  - **dcraw** on PATH (converts `.cr2` → `.tiff`).

## Setup

`run.py` self-installs most dependencies. Launch it and click
**Install Python Dependencies**, which runs `pip install -r
backend/modeling/requirements.txt pyserial` and `npm install` in the renderer
for you:

```bash
python run.py
```

The only things it can't install are Node.js, msys2/gphoto2, and dcraw — see
Prerequisites above. To install the Python deps manually instead, see
[`backend/modeling/README.md`](backend/modeling/README.md).

## Usage

Run `python run.py` and click the buttons top to bottom (select working image
set → capture → run modeling → build model → open viewer). The active working
folder holds one scan set and receives all pipeline output, so each set is
self-contained. Full flow is documented in the module docstring at the top of
`run.py`.

## Structure

```
run.py                      Tkinter launcher + dependency installer
backend/capture/            DSLR + Arduino capture rig — see backend/capture/README.md
backend/modeling/           Photometric-stereo pipeline — see backend/modeling/README.md
backend/rendering/          Three.js / Vite GLB baker  — see backend/rendering/README.md
```
