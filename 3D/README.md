# 3D — Tablet Photogrammetry Pipeline

COLMAP-based photogrammetry for cuneiform tablets (and any 3D object placed on
the rig): multi-camera capture → COLMAP reconstruction → meshed model. See the
[repo setup guide](../SETUP.md) for first-time install.

## Prerequisites

- **OS: WSL2 (Ubuntu)** — the reconstruction runs under WSL with WSLg for the
  GUI. See [`../SETUP.md`](../SETUP.md) for the WSL2 install steps.
- **NVIDIA GPU + Windows-host driver**, with `nvidia-smi` working inside WSL.
- **CUDA-enabled COLMAP** — the stock `apt` package is CPU-only; the pipeline
  prefers a local CUDA build. See
  [`Modeling/BUILDING_COLMAP.md`](Modeling/BUILDING_COLMAP.md).
- **exiftool** (`sudo apt install exiftool`) — COLMAP is fed metadata-stripped
  images.
- Python 3.9–3.12 and the deps in `Modeling/requirements.txt` (open3d has no
  prebuilt wheels for 3.13/3.14 yet).

## Setup

```bash
sudo apt install exiftool
pip install -r Modeling/requirements.txt
```

Then build CUDA COLMAP (one-time) — see
[`Modeling/BUILDING_COLMAP.md`](Modeling/BUILDING_COLMAP.md).

## Usage

- **Capture** (needs the physical rig): `capture_app.py` or the parallel-camera
  variant `capture_app_parallel.py`; rig firmware in `arduinoCode/`.
- **Reconstruction**: `python Modeling/app.py` — see
  [`Modeling/README.md`](Modeling/README.md) for the four-stage pipeline.

## Structure

```
capture_app.py              Multi-camera capture GUI
capture_app_parallel.py     Parallel-camera capture variant
arduinoCode/                Rig firmware (arduinoCode.ino)
Modeling/                   Reconstruction pipeline — see Modeling/README.md
```
