# Building CUDA-enabled COLMAP

The `3D/` pipeline needs GPU-accelerated dense reconstruction, so it wants a
**CUDA-enabled COLMAP**. The distro package (`sudo apt install colmap`) is
**CPU-only** and won't do — you must build COLMAP from source with CUDA and put
the result where `run.sh` expects it: an executable named **`colmap_local`** in
`3D/Modeling/` (this file's directory).

`colmap_local` is intentionally **not committed** (it's git-ignored) — everyone
builds it for their own GPU.

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
    libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev \
    libgoogle-glog-dev libgtest-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev
  ```

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
