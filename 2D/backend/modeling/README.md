# Modeling — Photometric Stereo

Turns a folder of directional-light scroll scans into render-ready texture maps
(normals, albedo, etc.) using photometric stereo. Part of the `2D/` pipeline;
see the [repo setup guide](../../../SETUP.md) for first-time install. Normally
run from `2D/run.py`, but can be installed and run standalone.

## Prerequisites

- **OS: Windows**, Python 3.9–3.12 (not 3.13/3.14 — the pinned `numpy<2.0`
  and the other pins only ship prebuilt wheels through 3.12).
- The Python packages in `requirements.txt`.

## Setup

```bash
pip install -r requirements.txt
```

Notes on the pins (see comments in `requirements.txt`):

- **`imagecodecs` is required**, not optional — the input scroll/calibration
  TIFFs are LZW-compressed and `tifffile` needs `imagecodecs` to decode them, or
  Stage 4 crashes.
- **`rembg==2.0.65` is pinned** because `rembg>=2.0.66` requires `numpy>=2.3.0`,
  which conflicts with the `numpy<2.0` pin. rembg downloads its segmentation
  model on first use.
- `opencv-python-headless` is used deliberately to avoid clashing with the
  headless OpenCV that rembg pulls in.

### GPU acceleration

`requirements.txt` pins the **CPU** backend (`onnxruntime`, `rembg==2.0.65`), so
Stage 0 (background removal) works out of the box with no GPU. To run it on an
NVIDIA GPU instead — which needs the **CUDA toolkit (`nvcc`)**, a driver, and
cuDNN installed — follow the
[GPU acceleration steps in the 2D README](../../README.md#gpu-acceleration-optional--cuda--nvcc),
then swap in the GPU wheels (`rembg[gpu]`, `onnxruntime-gpu`, a CUDA `torch`).
`_run_rembg()` passes `CUDAExecutionProvider` automatically when onnxruntime
reports it, so no code change is needed — the run log prints `Device : CUDA` once
the GPU stack is in place, or `Device : CPU` otherwise.

## Usage

```bash
python modeling_pipeline.py
```

Reads the active folder's scans and writes maps into `<active folder>/maps/`.

## Structure

```
modeling_pipeline.py    Main photometric-stereo pipeline
rotate_images.py        Image-rotation helper
requirements.txt        Python dependencies (see notes above)
```
