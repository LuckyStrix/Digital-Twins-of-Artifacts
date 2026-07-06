# Modeling — Tablet Reconstruction

The reconstruction half of the `3D/` pipeline: from captured photos to a meshed
model, driven by a Tkinter GUI (`app.py`, run under WSL with WSLg). See the
[repo setup guide](../../SETUP.md) for first-time install.

## Prerequisites

- **OS: WSL2 (Ubuntu)**, Python 3.9+.
- **CUDA-enabled COLMAP** — see the dependency notes below and
  [`BUILDING_COLMAP.md`](BUILDING_COLMAP.md).
- **exiftool**: `sudo apt install exiftool`.
- Python deps from `requirements.txt`.

## Setup

```bash
sudo apt install exiftool
pip install -r requirements.txt
```

`requirements.txt` pulls `open3d`, `numpy`, `rembg[gpu]` (use plain `rembg` on a
CPU-only machine), and `pillow`. rembg downloads its `birefnet-general` model on
first use.

## Dependencies

### COLMAP must have CUDA

`sudo apt install colmap` installs a **CPU-only** build with **no CUDA
support** — too slow for this pipeline. Build a CUDA-enabled COLMAP instead and
place it at `colmap_local` in this folder (or point `FIPMESH_COLMAP_BIN` at it).
Full instructions and a helper script: [`BUILDING_COLMAP.md`](BUILDING_COLMAP.md).

`run.sh` resolves the COLMAP binary in this order:

1. `$FIPMESH_COLMAP_BIN` if set (overrides everything).
2. `./colmap_local` if present and executable (auto-detected).
3. `colmap` on `PATH` otherwise.

### Checking what's installed

`run.sh` calls `src/check_config.sh` on startup, which reports which system and
Python deps are missing and (on Debian/Ubuntu) prints an `apt-get install`
suggestion. Run it to check your environment:

```bash
bash src/check_config.sh python3 colmap -- numpy open3d
```

(Arguments before `--` are system tools; after `--` are Python modules.)

## Usage

```bash
python app.py
```

## Pipeline stages

Mirrors the `app.py` module docstring:

1. **Background removal** — `process_photos.py`
2. **COLMAP MVS** — `run.sh` (once per side; `FIPMESH_SKIP_RECON=1` to reuse an
   existing reconstruction), which calls `src/main.sh` → `src/run_colmap_mvs.py`
3. **FPFH alignment** — `alignment/run.py` → `output/aligned_cloud/merged_fpfh.ply`
4. **Mesh reconstruction** — `src/reconstruct_mesh.py` → `output/recon/` and
   `output/model.gltf`

## Structure

```
app.py                  Tkinter reconstruction-pipeline GUI (4 stages)
process_photos.py       Stage 1: background removal
run.sh                  Stage 2 orchestrator (COLMAP; resolves colmap binary)
src/check_config.sh     Dependency checker (invoked by run.sh)
src/main.sh             COLMAP MVS + meshing driver
src/run_colmap_mvs.py   COLMAP MVS stage
src/reconstruct_mesh.py Open3D Poisson meshing stage
alignment/              FPFH point-cloud alignment (align.py, run.py)
viewer.py               Local model viewer
requirements.txt        Python dependencies
build_colmap.sh         Helper to build CUDA COLMAP into ./colmap_local
BUILDING_COLMAP.md      How to build CUDA-enabled COLMAP
```
