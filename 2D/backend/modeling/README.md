# Modeling — Photometric Stereo

Turns a folder of directional-light scroll scans into render-ready texture maps
(normals, albedo, etc.) using photometric stereo. Part of the `2D/` pipeline;
see the [repo setup guide](../../../SETUP.md) for first-time install. Normally
run from `2D/run.py`, but can be installed and run standalone.

## Prerequisites

- **OS: Windows**, Python 3.9+.
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
