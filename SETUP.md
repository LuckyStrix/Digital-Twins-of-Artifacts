# Setup Guide

First-time install for the three pillars of this repo. Each pillar targets a
specific operating system — read the matrix below before you start, then follow
the linked per-pillar README for the detailed steps.

## Prerequisites matrix

| Pillar       | OS target          | Key system tools                          | Python |
|--------------|--------------------|-------------------------------------------|--------|
| `2D/`        | **Windows / Linux**| Node.js (LTS), gphoto2, dcraw             | 3.9–3.12 |
| `3D/`        | **WSL2 (Ubuntu)**  | CUDA-enabled COLMAP, exiftool, NVIDIA GPU | 3.9–3.12 |
| `Website/`   | any                | none (static site)                        | 3.x      |

> **Use Python 3.9–3.12, not 3.13/3.14.** The pipelines pin `numpy<2.0`, and
> numpy 1.26.x (plus scipy, opencv, rembg) only ship prebuilt wheels up to
> Python 3.12. On 3.13+ pip falls back to building numpy from C source and fails
> with `ERROR: Unknown compiler(s)` unless you have a full MSVC/GCC toolchain.
> If you hit that, install Python 3.12 and launch with it (e.g. `py -3.12 run.py`
> on Windows). The `Website/` static server works on any Python 3.

The `2D/` and `3D/` pipelines run on **different operating systems** and keep
their Python dependencies in separate `requirements.txt` files — do not try to
install them into one shared environment.

## Windows or Linux (for `2D/`)

The `2D/` papyrus pipeline is a desktop application with two builds of the same
launcher:

- **Windows** — run `2D/run.py` (gphoto2 through msys2, serial port `COM3`).
- **Linux** — run `2D/run_linux.py` (native `gphoto2`/`dcraw` on PATH, serial
  port `/dev/ttyACM0`). On Debian/Ubuntu:
  `sudo apt install python3-tk gphoto2 dcraw`, and add yourself to the
  `dialout` group for serial access (`sudo usermod -a -G dialout $USER`).

Both builds share the same OS-independent modeling and rendering stages. The
launcher self-installs the Python and Node packages; you only need to provide the
tools it can't install for you:

- **Python 3.9–3.12** — https://www.python.org/downloads/ (tick "Add python.exe
  to PATH" on Windows).
- **Node.js LTS** — https://nodejs.org/ (tick "Add to PATH").
- **gphoto2** — on Windows via [msys2](https://www.msys2.org/)
  (`pacman -S mingw-w64-x86_64-gphoto2`); on Linux `sudo apt install gphoto2`.
- **dcraw** — Windows binary at https://sourceforge.net/app/dcraw/
  (`DCRaw_V9.28.exe`); on Linux `sudo apt install dcraw`.

> **Windows capture only — camera driver (Zadig):** on Windows, gphoto2 cannot
> see the DSLR until its USB driver is replaced with **WinUSB** using
> [Zadig](https://zadig.akeo.ie/). Step-by-step:
> [2D README → Camera driver setup](2D/README.md#camera-driver-setup-zadig-windows-only).
> (Not needed on Linux.)

> **Adding tools to PATH on Windows:** several of the tools above must be on your
> PATH. If you're unsure how, follow
> [this guide](https://www.howtogeek.com/787217/how-to-edit-environment-variables-on-windows-10-or-11/).

See [`2D/README.md`](2D/README.md) for the full list and the one command to start
it.

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
