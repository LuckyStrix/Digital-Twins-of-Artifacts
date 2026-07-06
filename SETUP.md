# Setup Guide

First-time install for the three pillars of this repo. Each pillar targets a
specific operating system — read the matrix below before you start, then follow
the linked per-pillar README for the detailed steps.

## Prerequisites matrix

| Pillar       | OS target        | Key system tools                          | Python |
|--------------|------------------|-------------------------------------------|--------|
| `2D/`        | **Windows**      | Node.js (LTS), msys2 + gphoto2, dcraw     | 3.9+   |
| `3D/`        | **WSL2 (Ubuntu)**| CUDA-enabled COLMAP, exiftool, NVIDIA GPU | 3.9+   |
| `Website/`   | any              | none (static site)                        | 3.x    |

The `2D/` and `3D/` pipelines run on **different operating systems** and keep
their Python dependencies in separate `requirements.txt` files — do not try to
install them into one shared environment.

## Windows (for `2D/`)

The `2D/` papyrus pipeline is a Windows desktop application driven by
`2D/run.py`. Its launcher self-installs most dependencies; you only need to
provide the tools it can't install for you (Python, Node.js, msys2/gphoto2,
dcraw). See [`2D/README.md`](2D/README.md) for the full list and the one
command to start it.

## WSL2 (for `3D/`)

The `3D/` tablet pipeline runs under **WSL2 with an Ubuntu distro** (it uses
WSLg for the GUI and CUDA-on-WSL for GPU-accelerated COLMAP).

1. Install WSL2 + Ubuntu from an **elevated PowerShell** on Windows:

   ```powershell
   wsl --install -d Ubuntu
   ```

   Official guide: https://learn.microsoft.com/windows/wsl/install

2. Install the **NVIDIA driver on Windows** (the *host*), not inside WSL. WSL
   picks up the GPU through it. Verify from inside Ubuntu:

   ```bash
   nvidia-smi
   ```

   CUDA-on-WSL setup:
   https://learn.microsoft.com/windows/wsl/tutorials/gpu-compute and
   https://docs.nvidia.com/cuda/wsl-user-guide/

3. Continue with [`3D/README.md`](3D/README.md), which covers `exiftool`, the
   Python deps, and building CUDA COLMAP.

## Website

No install step. From `Website/EFIP_Rewritten/`:

```bash
python3 -m http.server 8000
```

See [`Website/EFIP_Rewritten/README.md`](Website/EFIP_Rewritten/README.md).
