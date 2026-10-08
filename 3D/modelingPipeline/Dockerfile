# Docker image for the 3D reconstruction pipeline: CUDA COLMAP, exiftool, the
# Python deps and the Textual app, so none of BUILDING_COLMAP.md / the WSL CUDA
# setup is needed. See DOCKER.md for usage.
#
#   docker compose build
#   docker compose run --rm pipeline
#
# Build args:
#   COLMAP_VERSION        COLMAP git tag to build                     (default 4.2.1)
#   CUDA_ARCHITECTURES    GPU generations to compile COLMAP for       (default Pascal..Blackwell + PTX)
#   BUILD_JOBS            parallel compile jobs for COLMAP             (default 4)

# CUDA 12.8 is the oldest toolkit that knows Blackwell (sm_120), and Ubuntu
# 24.04's GCC 13 is the newest compiler CUDA 12.x accepts (BUILDING_COLMAP.md).
ARG CUDA_TAG=12.8.1
ARG UBUNTU_TAG=ubuntu24.04

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: build COLMAP with CUDA
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_TAG}-devel-${UBUNTU_TAG} AS colmap-build

# pipefail, so `cmake ... | tee` still fails the build when cmake fails.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
# build_colmap.sh's list, minus Qt (no GUI here) plus OpenBLAS for the bundled
# faiss. COLMAP's configure requires the OpenGL/GLEW headers even with
# OPENGL_ENABLED=OFF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libeigen3-dev libopenimageio-dev openimageio-tools libopencv-dev libmetis-dev \
        libgoogle-glog-dev libgtest-dev libgmock-dev libsqlite3-dev \
        libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev \
        libopenblas-dev libgl-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/*

ARG COLMAP_VERSION=4.2.1
# There is no GPU during `docker build`, so "native" can't be used. "-real"
# entries embed compiled code for that generation; the trailing "-virtual"
# embeds PTX that the driver JIT-compiles on GPUs newer than this list.
#   60 Pascal (P100)       61 Pascal (GTX 10xx)
#   70 Volta (V100)        75 Turing (RTX 20xx, GTX 16xx)   80 Ampere (A100)
#   86 Ampere (RTX 30xx)   89 Ada (RTX 40xx)                90 Hopper (H100)
#   100 Blackwell (B200)   120 Blackwell (RTX 50xx)
ARG CUDA_ARCHITECTURES="60-real;61-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"

RUN git clone --depth 1 --branch "${COLMAP_VERSION}" https://github.com/colmap/colmap.git /src/colmap

# The pipeline only uses SIFT, so ONNX (learned features; would download
# onnxruntime at build time), the GUI and OpenGL are switched off. GPU SIFT and
# PatchMatch stereo run on CUDA.
RUN cmake -S /src/colmap -B /src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/colmap \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DONNX_ENABLED=OFF \
        -DCCACHE_ENABLED=OFF \
        2>&1 | tee /tmp/cmake.log \
    && if grep -q "Disabling CUDA support" /tmp/cmake.log; then \
        echo "COLMAP configure did not enable CUDA" >&2; exit 1; \
    fi

# Parallel compile jobs. Each multi-architecture nvcc job can take several GB,
# and on Docker Desktop the build shares WSL2's memory with every other running
# container, so this stays well below the core count. Raise it on a big machine.
ARG BUILD_JOBS=4
RUN ninja -C /src/colmap/build -j "${BUILD_JOBS}" install \
    && echo "${CUDA_ARCHITECTURES}" > /opt/colmap/CUDA_ARCHITECTURES

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: runtime
# ─────────────────────────────────────────────────────────────────────────────
# The cudnn variant carries cuDNN 9, which onnxruntime-gpu (rembg) needs.
FROM nvidia/cuda:${CUDA_TAG}-cudnn-runtime-${UBUNTU_TAG} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
# COLMAP's shared-library dependencies (the runtime halves of the -dev packages
# above; checked with ldd below), exiftool for run.sh, Python, and the X/GL
# libraries open3d needs just to import.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-program-options1.83.0 libboost-graph1.83.0 libboost-system1.83.0 \
        libgoogle-glog0v6t64 libceres4t64 libopenimageio2.4t64 libmetis5 \
        libsqlite3-0 libcurl4t64 libssl3t64 libgmp10 libmpfr6 libgomp1 libopenblas0 libglew2.2 libopengl0 \
        exiftool python3 python3-venv git \
        libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=colmap-build /opt/colmap /opt/colmap
RUN if ldd /opt/colmap/bin/colmap | grep "not found"; then \
        echo "COLMAP is missing runtime libraries (above)" >&2; exit 1; \
    fi \
    && /opt/colmap/bin/colmap help | head -n 3

# Ubuntu 24.04's system pip refuses installs (PEP 668), so use a venv and put
# it first on PATH: src/main.sh calls a bare `python3`.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/colmap/bin:$PATH
RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

# torch is left out: nothing in the pipeline imports it, and rembg 2.x runs its
# models through onnxruntime. Its CUDA wheels would add ~6 GB and limit the
# image to the GPUs PyTorch's build supports.
COPY requirements.txt /tmp/requirements.txt
RUN grep -v -E '^torch([^a-z_-]|$)' /tmp/requirements.txt > /tmp/requirements-docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && python3 -c "import open3d, rembg, onnxruntime, textual, cv2"

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# run.sh uses FIPMESH_COLMAP_BIN ahead of ./colmap_local or PATH. rembg keeps
# its downloaded models in U2NET_HOME, a named volume in docker-compose.yml.
#
# CUDA's JIT cache goes in the same volume. COLMAP builds its PatchMatch
# kernels for Blackwell (sm_100+) as sm_90 PTX only, to dodge an NVCC
# miscompile (colmap/colmap#3514), so those GPUs JIT-compile them. In the
# default per-container cache that would repeat on every `run --rm`.
ENV FIPMESH_COLMAP_BIN=/opt/colmap/bin/colmap \
    U2NET_HOME=/cache/u2net \
    CUDA_CACHE_PATH=/cache/nv-compute \
    CUDA_CACHE_MAXSIZE=4294967296 \
    PYTHONDONTWRITEBYTECODE=1

# The run records (pipeline/runlog.py) note the git commit. The repo is a bind
# mount owned by another user, which git otherwise refuses to read.
RUN git config --system --add safe.directory '*'

# docker-compose.yml bind-mounts the whole repo at /repo, so .git is there too.
WORKDIR /repo/3D/modelingPipeline
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python3", "app.py"]
