# Building CUDA-enabled COLMAP

The `3D/` pipeline needs GPU-accelerated dense reconstruction, so it wants a
**CUDA-enabled COLMAP**. The distro package (`sudo apt install colmap`) is
**CPU-only** and won't do — you must build COLMAP from source with CUDA and put
the result where `run.sh` expects it: an executable named **`colmap_local`** in
`3D/Modeling/` (this file's directory).

`colmap_local` is intentionally **not committed** (it's git-ignored) — everyone
builds it for their own GPU.

## Supported WSL environment — use Ubuntu 24.04 LTS

Build inside an **Ubuntu 24.04 LTS** WSL instance. This is not arbitrary: CUDA's
`nvcc` only supports a bounded range of host GCC versions (CUDA 12.x tops out at
**GCC 13**), and a Linux distro's prebuilt system libraries (`libceres`, Qt, …)
must be linked with the **same** GCC that built them.

Ubuntu 24.04 ships **GCC 13** as its default compiler, so CUDA, COLMAP, and every
system library agree — it just works. Newer distros are a trap here: **Ubuntu
26.04 ships GCC 15**, which `nvcc` refuses, *and* whose system libraries require
GCC 15's runtime — so you can neither use the new compiler (CUDA rejects it) nor
fall back to GCC 13 (the system libs won't link against it). The build dies at
the final link step with:

```
undefined reference to `__cxa_call_terminate@CXXABI_1.3.15'
```

There is **no build-flag workaround** for that — the fix is to build on 24.04.
`build_colmap.sh` refuses to run on GCC > 13 and points you here.

Install the LTS instance from **Windows PowerShell** (it lives alongside any
existing distro; your project on the Windows drive stays reachable at the same
`/mnt/c/...` path):

```powershell
wsl --install Ubuntu-24.04
```

Then open Ubuntu 24.04, install the CUDA toolkit (below), and run the build there.
Confirm the toolchain before building:

```bash
c++ --version | head -1     # expect g++ 13.x
nvcc --version | tail -2     # CUDA 12.x
```

## Prerequisites (inside WSL2 Ubuntu)

- The **NVIDIA driver is installed on the Windows host** (not inside WSL), and
  `nvidia-smi` works inside Ubuntu. See [`../../SETUP.md`](../../SETUP.md).
- The **CUDA toolkit** installed in WSL (provides `nvcc`). Follow NVIDIA's
  CUDA-on-WSL guide: https://docs.nvidia.com/cuda/wsl-user-guide/
- Build tools and COLMAP's library dependencies (installed by the helper script
  below, or manually):

  ```bash
  sudo apt update && sudo apt install -y \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libopenimageio-dev openimageio-tools libopencv-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 qt6-svg-dev \
    libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev
  ```

  This list tracks COLMAP `main` (version `4.1.0.dev0`), which is what
  `build_colmap.sh` clones by default. Note `main` moved off FreeImage to
  **OpenImageIO** and now defaults to **Qt6** — the older FreeImage/Qt5 lists
  will not build current COLMAP. Because `main` is a moving target, its
  dependencies can drift; if a build breaks on a missing package, cross-check
  the official guide below for the commit you're building.

  `libopencv-dev` is required even though COLMAP itself doesn't use OpenCV:
  Ubuntu's `libopenimageio-dev` is built with OpenCV support, so its exported
  CMake target references `/usr/include/opencv4`. Without the OpenCV headers,
  cmake's generate step fails with *"Imported target OpenImageIO::OpenImageIO
  includes non-existent path /usr/include/opencv4"*.

  Optional: `libmkl-full-dev` (Intel MKL) accelerates BLAS/LAPACK but is a
  multi-GB install and not required.

Official COLMAP build guide: https://colmap.github.io/install.html

## Quick path — the helper script

From this directory:

```bash
./build_colmap.sh
```

It installs the apt dependencies, clones COLMAP, configures a CUDA build, and
symlinks the resulting binary to `./colmap_local`. Override the install/clone
location or COLMAP version with the environment variables documented at the top
of the script.

## Manual build

```bash
git clone https://github.com/colmap/colmap.git
cd colmap
cmake -S . -B build -GNinja -DCMAKE_CUDA_ARCHITECTURES=native
ninja -C build
```

`-DCMAKE_CUDA_ARCHITECTURES=native` targets your installed GPU. If cmake can't
detect it, set an explicit compute capability (e.g. `86` for RTX 30-series):
`-DCMAKE_CUDA_ARCHITECTURES=86`. During configure, confirm cmake reports CUDA
**enabled** — if it reports CUDA disabled, fix the toolkit/driver before
building.

Then make the fresh binary discoverable, either by symlinking it to the
expected path:

```bash
ln -sf "$(pwd)/build/src/colmap/exe/colmap" /path/to/3D/Modeling/colmap_local
```

or by pointing the pipeline at it via environment variable:

```bash
export FIPMESH_COLMAP_BIN="$(pwd)/build/src/colmap/exe/colmap"
```

## Verify

```bash
./colmap_local -h        # prints COLMAP help
```

`run.sh` auto-detects `./colmap_local` (see `run.sh`, the COLMAP-binary
resolution block); no further configuration is needed once the binary is in
place.
