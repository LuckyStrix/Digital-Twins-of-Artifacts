# 3D — Tablet Photogrammetry Pipeline

COLMAP-based photogrammetry for cuneiform tablets (and any 3D object placed on
the rig): multi-camera capture → COLMAP reconstruction → meshed model. See the
[repo setup guide](../SETUP.md) for first-time install.

## Prerequisites

- **OS: WSL2 (Ubuntu 24.04 LTS)** — the reconstruction runs under WSL with WSLg
  for the GUI. Use **24.04**, not the latest Ubuntu — newer releases ship a GCC
  too new for CUDA and COLMAP won't build. See [`../SETUP.md`](../SETUP.md) for the
  WSL2 install steps.
- **NVIDIA GPU + Windows-host driver**, with `nvidia-smi` working inside WSL.
- **CUDA-enabled COLMAP** — the stock `apt` package is CPU-only; the pipeline
  prefers a local CUDA build. See
  [`Modeling/BUILDING_COLMAP.md`](Modeling/BUILDING_COLMAP.md).
- **exiftool** (`sudo apt install exiftool`) — COLMAP is fed metadata-stripped
  images.
- **python3-tk** (`sudo apt install python3-tk`) — the capture and reconstruction
  GUIs use tkinter, which Ubuntu's stock `python3` does not bundle.
- **cuDNN 9 for CUDA 12** (`cudnn9-cuda-12`) — `rembg[gpu]` pulls
  `onnxruntime-gpu`, which needs the cuDNN runtime alongside CUDA. Without it
  background removal fails at model load (`libcudnn.so.9: cannot open shared
  object file`). It is **not** in Ubuntu's default repos — you must add NVIDIA's
  CUDA repo + GPG key first (see [Setup](#setup) below). On a CPU-only machine,
  use `rembg` (no `[gpu]`) instead and skip this.
- Python 3.9–3.12 and the deps in `Modeling/requirements.txt` (open3d has no
  prebuilt wheels for 3.13/3.14 yet). Pinned there: `numpy<2.5`,
  `onnxruntime-gpu<1.27`, and `torch` (required by rembg's birefnet model).

## Setup

Base tools from Ubuntu's repos:

```bash
sudo apt install exiftool python3-tk
```

Install **cuDNN 9 for CUDA 12**. It lives in NVIDIA's CUDA repo, so add the repo
and its GPG signing key first — the `cuda-keyring` package installs the key into
`/usr/share/keyrings`:

```bash
# Add NVIDIA's CUDA repo + GPG key (WSL-Ubuntu network repo)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Install cuDNN 9 for CUDA 12
sudo apt-get update
sudo apt-get -y install cudnn9-cuda-12

# Verify the runtime is present
ldconfig -p | grep libcudnn
```

Then install the Python deps (use `python3`, not `python`):

```bash
python3 -m pip install -r Modeling/requirements.txt
```

Then build CUDA COLMAP (one-time) — see
[`Modeling/BUILDING_COLMAP.md`](Modeling/BUILDING_COLMAP.md).

> **Out of memory / `Killed` mid-run?** COLMAP dense reconstruction and rembg are
> memory-hungry. WSL2 caps its VM at half the host RAM by default — raise the RAM
> and swap (pagefile) limits via `.wslconfig`; see
> [`../SETUP.md`](../SETUP.md#giving-wsl-more-ram--a-bigger-swap-pagefile).

## Usage

- **Capture** (needs the physical rig): `capture_app.py` or the parallel-camera
  variant `capture_app_parallel.py`; rig firmware in `arduinoCode/`.
- **Reconstruction**: `python3 Modeling/app.py` — see
  [`Modeling/README.md`](Modeling/README.md) for the four-stage pipeline.

## Structure

```
capture_app.py              Multi-camera capture GUI
capture_app_parallel.py     Parallel-camera capture variant
arduinoCode/                Rig firmware (arduinoCode.ino)
Modeling/                   Reconstruction pipeline — see Modeling/README.md
```
