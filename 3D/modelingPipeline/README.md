# Modeling — Tablet Reconstruction

The reconstruction half of the `3D/` pipeline: from captured photos to a meshed
model, driven by a Tkinter GUI (`app.py`, run under WSL with WSLg). See the
[repo setup guide](../../SETUP.md) for first-time install.

## Prerequisites

- **OS: WSL2 (Ubuntu)**, Python 3.9–3.12 (not 3.13/3.14 — open3d has no
  prebuilt wheels there yet).
- **CUDA-enabled COLMAP** — see the dependency notes below and
  [`BUILDING_COLMAP.md`](BUILDING_COLMAP.md).
- **exiftool**: `sudo apt install exiftool`.
- **cuDNN 9 for CUDA 12** (`cudnn9-cuda-12`) for `rembg[gpu]` — needs NVIDIA's
  CUDA repo + GPG key; see [`../README.md`](../README.md#setup) or
  [`../../SETUP.md`](../../SETUP.md).
- Python deps from `requirements.txt`.

## Setup

```bash
sudo apt install exiftool
python3 -m pip install --break-system-packages -r requirements.txt
```

`requirements.txt` pulls `open3d`, `numpy<2.5`, `rembg[gpu]` (use plain `rembg`
on a CPU-only machine) with `onnxruntime-gpu<1.27` and `torch`, and `pillow`.
rembg downloads its `birefnet-general` model on first use. For the cuDNN runtime
that `onnxruntime-gpu` needs, and for raising WSL's RAM/swap limits if runs get
OOM-killed, see [`../README.md`](../README.md#setup).

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
python3 app.py
```

## Pipeline stages

Mirrors the `app.py` module docstring:

1. **Background removal** — `process_photos.py`
2. **COLMAP MVS** — `run.sh` (once per side; `FIPMESH_SKIP_RECON=1` to reuse an
   existing reconstruction), which calls `src/main.sh` → `src/run_colmap_mvs.py`
3. **FPFH alignment** — `alignment/run.py` → `output/aligned_cloud/merged_fpfh.ply`
4. **Mesh reconstruction** — `src/reconstruct_mesh.py` → `output/recon/` and
   `output/model.gltf` (plus `output/model_simplified.glb` and, if the Inputs
   tab's Artifact info Name is set, `output/info.txt`)

## GUI reference

`app.py` is one window with four tabs (Inputs, COLMAP, Reconstruct, Alignment)
plus a run panel that walks the four stages in order and streams live output to
the log pane. Settings persist between runs in `app_defaults.json` (delete it
to reset to the values below). Where a field has a longer explanation, the app
shows it inline under the control — this section is a quicker reference, not a
replacement for those.

### Inputs tab

**Paths**
- **Input dir / Output dir** — source photo folder and where the pipeline
  writes everything. Output defaults to `<input>_recon` next to the input once
  you pick one.
- **Primary / Secondary** side folder names (default `side1`/`side2`) — the
  subfolder names under the input dir holding each side's photos. Leave
  Secondary blank to skip Stage 3 (alignment) and reconstruct a single side.
  Auto-detected from the input dir's structure when you browse to it.

**Artifact info** (optional, not persisted in `app_defaults.json` — specific
to one artifact, unlike the rest of this tab) — **Name**, **Type**,
**Description**, **Link**, **Link Label**. If Name is set, Stage 4 writes an
`info.txt` next to `model.gltf`/`model_simplified.glb` in the website's
artifact-metadata format (see
[`../../website/artifacts/README.md`](../../website/artifacts/README.md)),
ready to drop straight into `website/artifacts/<id>/`. Link and Link Label are
each independently optional. Leave Name blank to skip writing it. Can also be
generated standalone, without the GUI: `python3 -m src.artifact_info
--output-dir ... --name ... --type ... --description ... [--link ...]
[--link-label ...]`.

**Background removal** (`process_photos.py`, Stage 1)
- **Background** — `white` / `black` / `transparent` composite color for the
  masked output. Transparent keeps a PNG alpha channel; white/black composite
  the tablet onto a solid backdrop.
- **Hard mask cutoff** (default **on**) — thresholds rembg's soft alpha matte
  to a binary mask instead of leaving a blended gradient at the silhouette.
  Recommended on; see the in-app note for why (CLI: `--hard-mask` /
  `--no-hard-mask`).
- **Black / White / Value threshold** (0 = off) — force near-black, near-white,
  or near-black-*or*-white background pixels transparent by raw channel/value
  cutoff, on top of rembg's mask.
- **Chroma threshold** (0 = off, try 15–30) — removes low-saturation colour
  bleed left around the mask edge; only safe against a neutral (black/white/
  grey) backdrop.
- **Edge band (px)** (0 = whole mask) — restricts the threshold cleanups above
  to a band this many pixels wide around the mask boundary, instead of the
  entire image.
- **Erode mask (px)** (0 = off) — shrinks the alpha mask inward by this many
  pixels after cleanup, matching what `erode_masks.py <folder> <px>` does to
  an already-processed folder.
- **Grow by colour** (0 = off, try 40–60) — grows the mask onto any
  colour-bearing region touching the tablet, regardless of hue. Fixes rembg
  dropping an attached, differently-coloured piece (e.g. a mounting board) as
  background. Only works against a neutral backdrop.
- **Convex-hull fill grown region** (default off) — bridges low-colour detail
  (e.g. a printed label) sitting on the grown region, even where it touches the
  region's own edge. Assumes the attached piece is roughly convex; off by
  default since most tablets aren't board-backed.
- **Passes** (1 = off) — re-runs background removal on its own output this many
  times; can clean up residual background at the cost of speed.
- **Seg. quality** (10–100%) — downscales the image fed to the segmentation
  model to save CPU/RAM time. Saved output is always full resolution; rembg
  resizes every model's input to a fixed size regardless, so this doesn't
  affect GPU VRAM.
- **rembg model** — which rembg/BiRefNet model performs the segmentation
  (default `birefnet-general`). Swap to a specialized model (e.g.
  `isnet-general-use`) if the default mis-segments a particular material.

**Experimental: COLMAP-side masking** — `process_photos.py --save-masks`
writes each photo's final binary mask as its own PNG (default `--mask-dir
<output>_masks`). If Stage 2 finds a matching `<processed-dir>_masks` folder
it passes it to `run.sh -m`/`-n` automatically, which feeds
`run_colmap_mvs.py --mask-path`/`--mask-path-secondary` so `feature_extractor`
and `stereo_fusion` ignore background pixels directly, instead of COLMAP
guessing from RGB alone. No GUI toggle for this — it's CLI/env-only
(`--save-masks`) and off by default, because it was found to be able to
destabilize camera pose estimation on low-texture images. **Prune fill
colour** below is the preferred, lower-risk fix for background bleed; masking
is left in for further experimentation.

### COLMAP tab

Mirrors `run_colmap_mvs.py`'s CLI/env options; see that script's `--help` for
full detail on any field.

- **Quality** — `extreme`/`high`/`medium`/`low` preset controlling COLMAP's
  internal image-size and matching trade-offs (higher = slower, more detail).
- **Use GPU** / **GPU index** — enable CUDA dense matching/fusion and pick
  which GPU (`-1` = auto-select).
- **Image scale (0–1)** — downscale images before feature extraction.
- **Image stride** — despite living on this tab, this drives Stage 1's own
  frame-skipping (`process_photos.py --stride N`, keep every Nth photo) so
  that COLMAP only ever receives the already-filtered images; COLMAP itself
  always runs with stride 1 against whatever Stage 1 produced.
- **SIFT**: **Max features**, **Peak threshold**, **Edge threshold** control
  how many/how strong the keypoints extracted per image are. **Domain size
  pooling** and **Estimate affine shape** are SIFT variants that trade extra
  compute for features more robust to scale/viewpoint change — useful for
  textureless or highly curved surfaces.
- **Matching**: **Guided matching** re-matches using estimated geometry for
  higher precision; **Max matches** caps matches kept per image pair.
- **Threads / Cache** (0 = auto) — per-stage CPU thread counts and PatchMatch/
  fusion GPU cache sizes in GB; raise the caches if you have GPU memory to
  spare and are fusing large scenes, lower them if COLMAP runs out of VRAM.
- **Secondary camera** — how side 2's reconstruction is re-posed relative to
  side 1 before alignment: **Rotate degrees/axis** plus **Extra rotate X/Y/Z**
  and **Translate X/Y/Z** for fine adjustment, and **Align mode**
  (`auto`/`manual`) for whether the pipeline estimates this automatically or
  uses only the values you entered.

### Reconstruct tab

Mirrors `src/reconstruct_mesh.py`'s CLI options (`python3 src/reconstruct_mesh.py
--help` for the authoritative descriptions and defaults).

- **Prune fill colour** — runs first, before any other cleanup. Drops fused
  points whose colour sits within a threshold of a flat background fill
  (`white`/`black`, matching Stage 1's **Background** setting; `none` for
  `transparent`, which has no single fill colour to prune by). Not a GUI
  field itself — the app derives it automatically from the Inputs tab's
  **Background** choice on every Stage 4 run (`reconstruct_mesh.py
  --prune-fill-color`/`--prune-fill-threshold`, default threshold 16). Targets
  background bleed that triangulated as fake surface; since it only ever
  deletes points from the already-fused cloud, it can't affect pose
  estimation the way COLMAP-side masking (above) can.
- **Point cloud filtering** — **Max input points** randomly downsamples before
  reconstruction (0 disables). **Outlier neighbors/std ratio** and **Radius
  outlier NB pts/factor** are Open3D's statistical and radius outlier removal
  passes (≤0 disables each).
- **DBSCAN clustering** — clusters the cleaned cloud and keeps only the
  largest, connected cluster(s), to drop disconnected background debris that
  outlier removal alone doesn't catch. **Eps factor** sets the neighbor-distance
  radius (as a multiple of median NN spacing) that defines a cluster; **min
  points** is the DBSCAN density threshold; **keep largest** caps how many
  clusters survive; **min cluster ratio** drops clusters smaller than this
  fraction of the largest one. **Max points** skips DBSCAN entirely above that
  cloud size (it's the slowest cleanup step).
- **Normals** — **max NN**/**orient K** control how many neighbors are used to
  estimate and consistently orient surface normals before Poisson.
- **Poisson reconstruction** — **depth** is the octree resolution (higher =
  more detail, more memory); **linear fit** trades a bit of smoothness for
  sharper detail; **density trim quantile** strips the lowest-confidence
  Poisson surface (raise to trim more); **crop scale** bounds the output mesh
  to the point cloud's bounding box scaled by this factor, to cut Poisson's
  characteristic ghost geometry outside the actual scan.
- **Hole reduction** (checkbox) — a preset that loosens density trim, widens
  crop scale, and raises normal max NN together, so sparsely sampled areas
  (turntable rotation-axis poles, smooth/textureless surfaces) survive trimming
  instead of becoming holes. Overrides the three fields above when enabled;
  pair with hole filling below for whatever gaps remain.
- **Hole filling** (checkbox) — after cleanup, triangulates leftover mesh
  boundary loops to close remaining gaps directly. **Fill hole size ratio**
  caps the max hole radius filled, as a fraction of the cloud's bounding-box
  diagonal. **Fill hole passes** repeats the fill (closing one loop can free up
  a neighbor) until a pass closes nothing new, up to this cap.
- **Mesh cleanup** — **Component min ratio/min triangles** drop small
  disconnected mesh pieces (by relative size and by absolute triangle count);
  **Component max count** caps how many components survive; **Smooth
  iterations** is post-cleanup Taubin smoothing; **Decimate target tris**
  is the quadric-decimation triangle budget for the main output mesh (≤0
  disables).
- **Simplified export** — always also writes `<name>_simplified.glb`: a single
  small vertex-colored binary glTF decimated to roughly **Simplified target
  verts**, meant for dropping straight into a web viewer. Set to 0 to disable.
- **Pose normalization** (checkbox, default off) — recenters the mesh at the
  origin and rotates it (via PCA on the vertices) so its widest cross-section
  lies in the XY plane and its thinnest axis (tablet thickness) lies along Z —
  like a coin lying flat. Applied once, before decimation, so every exported
  variant shares the same pose.

### Alignment tab

Controls `alignment/run.py`, Stage 3 (only runs when a Secondary side folder is
set on the Inputs tab).

- **Method** — `opening`/`fpfh`/`collapse`/`all`; `fpfh` (default) is FPFH
  feature matching + RANSAC, refined with ICP.
- **Voxel (0=auto)** — downsample voxel size used during alignment; 0 picks one
  from the cloud scale automatically.
- **Sample points** — how many points are sampled from each side's cloud for
  feature matching.
- **PLY overrides** — point the alignment step at specific PLY files instead of
  the pipeline's own Stage 2 outputs, for re-running alignment in isolation
  (e.g. after manually editing a cloud). Leave blank to use the normal
  pipeline outputs.

## Structure

```
app.py                  Tkinter reconstruction-pipeline GUI (4 stages)
app_defaults.json       Last-used GUI field values, saved/restored between runs
process_photos.py       Stage 1: background removal
erode_masks.py          Re-erode already-generated masks without rerunning Stage 1
run.sh                  Stage 2 orchestrator (COLMAP; resolves colmap binary)
src/check_config.sh     Dependency checker (invoked by run.sh)
src/main.sh             COLMAP MVS + meshing driver
src/run_colmap_mvs.py   COLMAP MVS stage
src/reconstruct_mesh.py Open3D Poisson meshing stage
src/artifact_info.py    Writes info.txt (website artifact metadata); also runnable standalone
alignment/              FPFH point-cloud alignment (align.py, run.py)
viewer.py               Local model viewer
requirements.txt        Python dependencies
build_colmap.sh         Helper to build CUDA COLMAP into ./colmap_local
BUILDING_COLMAP.md      How to build CUDA-enabled COLMAP
```
