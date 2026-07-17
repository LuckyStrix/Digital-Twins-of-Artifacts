#!/bin/bash
#
# build_colmap.sh — build CUDA-enabled COLMAP and drop the binary at
# ./colmap_local, where run.sh expects it. See BUILDING_COLMAP.md for details.
#
# Run inside WSL2 Ubuntu with the NVIDIA driver on the Windows host and the
# CUDA toolkit installed (nvcc on PATH). Override defaults via env vars:
#
#   COLMAP_SRC_DIR   where to clone/build COLMAP   (default: ./colmap-src)
#   COLMAP_VERSION   git tag/branch to check out   (default: main)
#   CUDA_ARCH        target CUDA architecture       (default: native)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLMAP_SRC_DIR="${COLMAP_SRC_DIR:-$SCRIPT_DIR/colmap-src}"
COLMAP_VERSION="${COLMAP_VERSION:-main}"
CUDA_ARCH="${CUDA_ARCH:-native}"
TARGET_BIN="$SCRIPT_DIR/colmap_local"

print_green() { printf "\e[32m%s\e[0m\n" "$1"; }
print_red()   { printf "\e[31m%s\e[0m\n" "$1"; }

# -------------------------------
# sanity checks
# -------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
    print_red "nvcc not found — install the CUDA toolkit in WSL first."
    print_red "See BUILDING_COLMAP.md."
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    print_red "nvidia-smi not working — install the NVIDIA driver on the Windows host."
    exit 1
fi

# -------------------------------
# 1. build dependencies
# -------------------------------
print_green "Installing COLMAP build dependencies via apt..."
sudo apt-get update
# Tracks COLMAP main (4.1.0.dev0): OpenImageIO (not FreeImage) + Qt6.
# libmkl-full-dev (Intel MKL) is an optional multi-GB BLAS accelerator, omitted.
sudo apt-get install -y \
    git cmake ninja-build build-essential \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libopenimageio-dev openimageio-tools libmetis-dev \
    libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qt6-base-dev libqt6opengl6-dev libqt6openglwidgets6 qt6-svg-dev \
    libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev

# -------------------------------
# 2. clone
# -------------------------------
if [ -d "$COLMAP_SRC_DIR/.git" ]; then
    print_green "Reusing existing clone at $COLMAP_SRC_DIR"
    git -C "$COLMAP_SRC_DIR" fetch --tags origin
else
    print_green "Cloning COLMAP into $COLMAP_SRC_DIR"
    git clone https://github.com/colmap/colmap.git "$COLMAP_SRC_DIR"
fi
git -C "$COLMAP_SRC_DIR" checkout "$COLMAP_VERSION"

# -------------------------------
# 3. configure + build (CUDA)
# -------------------------------
print_green "Configuring CUDA build (CMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH)..."
cmake -S "$COLMAP_SRC_DIR" -B "$COLMAP_SRC_DIR/build" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"

print_green "Building (this takes a while)..."
ninja -C "$COLMAP_SRC_DIR/build"

# -------------------------------
# 4. link binary to ./colmap_local
# -------------------------------
BUILT_BIN="$(find "$COLMAP_SRC_DIR/build" -type f -name colmap -perm -u+x | head -n1)"
if [ -z "$BUILT_BIN" ]; then
    print_red "Build finished but could not locate the colmap binary."
    exit 1
fi

ln -sf "$BUILT_BIN" "$TARGET_BIN"
print_green "Linked $TARGET_BIN -> $BUILT_BIN"
print_green "Verifying..."
"$TARGET_BIN" -h >/dev/null && print_green "CUDA COLMAP ready at ./colmap_local"
