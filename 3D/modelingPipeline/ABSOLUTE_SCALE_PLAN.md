# Plan: absolute scale from a reference ring

Branch: `absolute-scale` (cut from `main` at `5678f80`).
Working dir for everything below: `3D/modelingPipeline/`.

## Goal

Make reconstructions come out in real units: millimeters through Stages 2–4,
meters in the exported glTF/GLB. Today each side's COLMAP run picks its own
arbitrary scale and Stage 3 alignment is rigid (no scaling), so the two sides
can disagree in size. Scaling both sides to millimeters before alignment fixes
that too.

Photos fix a model only up to one overall factor `s`. A ring of known size,
fixed to the turntable and visible in the raw photos, gives
`s = known radius / fitted radius` for every scan.

## Decisions already made

- **Ring:** 200 mm inner / 216 mm outer diameter (8 mm band), 1 mm tall,
  printed black with the top 2–3 layers white, so only the top face is white.
  Fixed to the turntable (rotates with the tablet), roughly centered, on black
  felt. White side walls would bias the edge fits by about 1%, so they must be
  black.
- **Ring size:** the ring can't be measured with calipers (too big). The TUI
  defaults to the design values; a tape or ruler measurement can replace them.
  A small systematic error is acceptable: every scan shares it, so sides and
  scans stay consistent. An optional caliper-measured band width is used only
  as a check on the fit.
- **Rescaling later:** record the ring radius measured in raw COLMAP units for
  every run, so old models can be rescaled exactly if the ring is measured
  better later.
- **Units:** millimeters in Stages 2–4. glTF/GLB in meters (the glTF spec
  requires it; the website viewer fits the camera to the model's size, so it
  is unaffected).
- **Fit:** a general 3D circle (center, tilt, radius) across all views. The
  turntable axis from the camera centers is only a check. Detection runs on the
  raw photos. Partial arcs are fine; keep framing tight on the tablet.
- **Several sparse models:** scale every model or none. Also warn in the TUI
  whenever a side splits (NAS history: tablets never split; the wax seal,
  knife, tape measure and phone tests did).
- **Failure:** continue unscaled with a warning. Never fail the run.
- **UI:** everything wired through the TUI. Push changes as they are made.

## Facts from the code that shape the design

- Stage 2 runs once per side as its own process: `run.sh` → `src/main.sh` →
  `src/run_colmap_mvs.py` on Linux/Docker, `src/run_stage2.py` →
  `run_colmap_mvs.py` on Windows. Hooking into `run_colmap_mvs.py` covers all
  three. Settings travel as `FIPMESH_*` environment variables
  (`pipeline/runner.py::colmap_env`); no command-line changes in
  run.sh/main.sh/run_stage2.py are needed.
- **Hook point:** `run_colmap_mvs.py` after `mapper` and the `_txt_export`
  (around line 3344), before the per-model dense loop (around line 3380).
  Scaling the sparse model there puts the depth maps, `fused.ply`, the `.vis`
  files, `dense/sparse/images.bin` (used by `alignment/confidence.py`) and
  `camera_centers.json` in millimeters.
- **Several models:** `list_sparse_models` can return several; each is
  densified separately, then `merge_component_clouds` registers them rigidly.
  `camera_centers.json` comes only from `model_dirs[0]`.
- **Raw ↔ processed names:** Stage 1 changes the extension
  (`IMG_0001.JPG` → `.png` for a transparent background, `.jpg` for
  white/black; `process_photos.py:550/557`). `images.txt` holds the processed
  name (`cam1/IMG_0001.png`), so match raw photos by relative path without the
  extension.
- **EXIF orientation:** Stage 1 never applies it and exiftool strips it before
  COLMAP, so COLMAP sees the sensor's own pixel layout. Raw photos must be
  loaded without applying orientation (PIL, or `cv2.IMREAD_IGNORE_ORIENTATION`).
- **Stride:** Stage 1 keeps every Nth photo, so only photos named in
  `images.txt` have poses.
- **Image scale:** when `FIPMESH_COLMAP_IMAGE_SCALE < 1`, the intrinsics are for
  downscaled copies. Get the factor per camera from `cameras.txt` width ÷ raw
  photo width; don't trust the setting.
- **Stage 3** is scale-relative (voxel = 1% of the cloud's size;
  `align_voxel_var` defaults to `"0"`). No change needed.
- **Stage 4** constants are ratios or counts by default (hole fill = ratio of
  the bounding box). `compute_pca_pose` is rigid. glTF/GLB are written by our
  own `build_gltf_document`, so a units factor goes there.
- **Not `info.txt`:** the website's parser (`website/tools/build_manifest.py`,
  `FIELD_RE`) only knows Name/Type/Description/Link/Link Label and appends any
  other line to the previous field. A `Scale:` line would end up in the
  Description. Put scale results in `scale.json` and `pipeline_runs.json`.
- **No scipy in `requirements.txt`.** Write the fits with numpy (Gauss–Newton or
  IRLS on a handful of parameters) and use OpenCV (already required) for image
  work. No new dependencies, so Docker and the Windows installer don't change.

## Architecture

```
3D/modelingPipeline/
  scale/                     NEW, no UI imports
    __init__.py
    colmap_model.py          read/write COLMAP TXT models; rescale a model by s;
                             camera centers; undistort pixels per camera model
    ring.py                  ring detection + 3D circle fit -> RingFit / None
    apply.py                 Stage 2 hook: fit every sparse model, all-or-none,
                             rescale, write scale.json; also unscale a workspace
    rescale_session.py       CLI: rescale a finished session's outputs with a new
                             ring measurement (from the recorded raw radius)
  src/run_colmap_mvs.py      calls scale.apply after the mapper (when enabled)
  src/reconstruct_mesh.py    --units mm|none (glTF in meters), --crop-plane
  pipeline/settings.py       new "Scale" tab
  pipeline/runner.py         env vars, both-sides check, split warning, Stage 4 args
  pipeline/runlog.py         records scale results per stage
  tests/test_scale_*.py      synthetic-render tests
```

### `scale/colmap_model.py`

- `read_model_txt(dir)` / `write_model_txt(dir, model)`: cameras, images,
  points3D. Convert BIN to TXT and back with `colmap model_converter` (already
  used here).
- `rescale_model(model, s)`: multiply each image's `t` and each point's `xyz`
  by `s`; rotations and intrinsics are unchanged. Do this ourselves, not with
  `model_transformer`, whose transform-file format has changed between COLMAP
  versions.
- `camera_center(image)`, `undistort_pixels(camera, uv)` for SIMPLE_PINHOLE,
  PINHOLE, SIMPLE_RADIAL, RADIAL and OPENCV (map to `cv2.undistortPoints`
  coefficients). Any other model: return None, which makes the scale step fail
  cleanly.

### `scale/ring.py`: detection and fit

Input: a sparse model (TXT), the raw side folder, and the ring's design values.
Output: `RingFit` (scale, uncertainty, inner/outer radius in model units,
center, normal, per-camera scales, warnings) or `None` with a reason.

1. **Pair photos.** For each image in `images.txt`, find the raw photo
   `<raw>/<rel path without extension>.*`. Skip photos that don't pair. Load
   with orientation ignored. Get the pixel factor from the `cameras.txt` width.
2. **Turntable axis.** Each physical camera's centers lie on a circle around
   the axis. Fit a plane and circle per camera folder (`cam1`, `cam2`, …); the
   axis is the shared plane normal through the circle centers. Used to set up
   the starting guess and as a check.
3. **Edges.** Canny-style edges plus gradient direction on the raw image.
   Exclude the tablet: dilate the Stage 1 mask (the alpha of transparent PNGs,
   or the `processed_masks` folder if present, or pixels in the processed photo
   that differ from the fill color) and drop edges inside it. Gradient
   direction labels each edge as inner or outer: the white band is brighter
   than the felt on both sides, so on the outer edge the gradient points
   toward the axis and on the inner edge it points away.
4. **Starting guess.** Assume a circle centered on the axis and perpendicular
   to it. Grid-search its height along the axis and its radius; score each
   candidate by edge strength along its projection in a subset of images.
5. **Refinement.** Parameters: center (3), normal (2), `r_in`, `r_out`. For
   each edge pixel, undistort it, cast the ray, intersect it with the ring
   plane, and use (distance from center − r_in or r_out) as the residual.
   Robust IRLS (Huber, then Tukey), dropping edges more than about 3σ from the
   fit. Only edges within a band around the current prediction count, so
   tablet and turntable clutter can't take over the fit.
6. **Scale.** `s = ((D_in + D_out)/4) / ((r_in + r_out)/2)`: the centerline,
   so edge bias cancels.
7. **Uncertainty.** Refit each camera folder alone, and bootstrap over images.
   Report σ (in %).
8. **Checks** (any one fails the fit, with a reason):
   - fewer than about 8 images with enough edges along the ring;
   - inner-only and outer-only scales differ by more than 0.5%;
   - per-camera scales spread by more than 0.5%;
   - fitted band width differs from the design width (or the measured width,
     if given) by more than 20% (catches a white edge being mistaken for the
     ring);
   - ring center more than 10 mm from the axis, or tilt more than 3°;
   - σ above 0.3%.

   All thresholds are module constants, tuned on the first real capture.

### `scale/apply.py`: Stage 2 hook

Called from `run_colmap_mvs.py` after the TXT export when
`FIPMESH_SCALE_RING=1`:

1. Fit the ring for every sparse model in `model_dirs`.
2. **All fit:** back up each model to `sparse/<i>_unscaled/`, rescale it in
   place (TXT → rescale → BIN), redo `_txt_export` and `camera_centers.json`.
   **Any fails:** change nothing.
3. Write `<workspace>/scale.json`: `status` (`scaled` / `unscaled` /
   `disabled`), `units`, `scale`, `sigma_pct`, the ring values used, and per
   model: raw `r_in`/`r_out` in model units, center, normal, per-camera
   scales, warnings, and the reason if it failed.
4. Print `[scale] …` lines: a summary plus a warning on failure, for the log
   and the parser.

`unscale_workspace(ws)`: divide the dense model, `fused.ply`, the component
clouds and `camera_centers.json` by `s` and set `status: "reverted"`. Used
when side 1 scaled but side 2 didn't (below).

### Runner and TUI (`pipeline/`)

- **Settings**, in a new `Scale` tab added to `TABS` and `PERSISTED_KEYS`:
  - `scale_ring_var` (BOOL). Default **off** until the first real capture
    validates the fit, then switch to on.
  - `scale_ring_inner_var` (FLOAT, 200.0 mm).
  - `scale_ring_outer_var` (FLOAT, 216.0 mm).
  - `scale_ring_width_var` (FLOAT, 0 = not measured; caliper band width).
  - Help text covers the ring spec, what happens on failure, and that
    measured diameters (tape ÷ π, or a ruler) replace the design values.
- **`colmap_env`:** export `FIPMESH_SCALE_RING`, `_INNER_MM`, `_OUTER_MM`,
  `_WIDTH_MM`. `_colmap_one_side` adds `FIPMESH_SCALE_RAW_IMAGES=<input>/<side>`
  and `FIPMESH_SCALE_PROCESSED=<processed>/<side>` (for the tablet mask).
- **After Stage 2, both sides:** read both `scale.json` files. Exactly one
  `scaled` → call `unscale_workspace` on it and warn: "side 2 couldn't be
  scaled; both sides left unscaled so they still match".
- **Split warning:** watch Stage 2 output for `sparse models found: N` with
  N > 1. Log it as a warning and show it in the Stage 2 card's status text.
- **Stage 4:** pass `--units mm` when every side is `scaled`, else
  `--units none`. Pass `--crop-plane` from side 1's ring plane (the merged
  cloud is in side 1's frame: `save_merged` moves B onto A). Write
  `<session>/scale.json` summarising both sides and the units of
  `model.gltf`.
- **Run records:** `runlog.stage_outputs` adds each side's scale result for
  Stage 2 and the final units for Stage 4, so `pipeline_runs.json` keeps the
  raw radius for later rescaling.
- **Visible result:** a Stage 2 card status like `scaled ±0.08%` or
  `unscaled (see log)`. The log shows `1 model unit = X mm` per side.

### Stage 4 (`src/reconstruct_mesh.py`)

- `--units mm|none`: with `mm`, multiply vertex positions by 0.001 in
  `build_gltf_document`, so only glTF/GLB are in meters. OBJ/PLY stay in mm and
  every internal step is unchanged.
- `--crop-plane nx,ny,nz,d` (optional): drop cloud points at or below the ring
  plane (plus about 0.5 mm) before cleanup. This removes the ring and turntable
  felt if rembg kept them. Skipped when the plane is missing.
- Print the final bounding box in mm, so a quick sanity check against the real
  tablet is in the log.

### `scale/rescale_session.py`

`python -m scale.rescale_session <session> --inner-mm X --outer-mm Y`: recompute
`s` from the recorded raw radii and rescale `model.gltf`, `model_simplified.glb`
and the `scale.json` files in place. For rescaling old models after a better
ring measurement. Low priority; last phase.

## Phases (one or more commits each, pushed as they land)

1. **Plan:** this file.
2. **`scale/colmap_model.py`** + tests: TXT round trip, rescale scales
   centers by `s` and keeps rotations, undistortion matches `cv2.projectPoints`
   round trips.
3. **Synthetic test scene** (`tests/scale_synth.py`): rig of 3 cameras at
   different heights × 24 turntable steps; ring of known size in a random
   similarity frame (random scale, rotation, offset, slight tilt and
   off-center); render white band on black with blur, noise and a tablet-shaped
   occluder; write raw JPGs, processed PNGs with alpha, and a COLMAP TXT model.
   Knobs: ring clipped by the frame, `.JPG` raw vs `.png` processed names,
   an EXIF orientation tag on the raw photos, stride (raw photos without
   poses), downscaled intrinsics, a SIMPLE_RADIAL/OPENCV distortion, no ring.
4. **`scale/ring.py`** against the synthetic scenes: recover `s` within 0.05%;
   the axis check works; each failure check fires on the scene built to
   trigger it; no ring → `None`.
5. **`scale/apply.py` + `run_colmap_mvs.py` hook:** all-or-none across
   models; `scale.json` contents; backup/restore; `unscale_workspace`. Test
   with fake model dirs (no COLMAP needed).
6. **Stage 4:** `--units` and `--crop-plane`, with glTF tests (positions
   ×0.001; OBJ unchanged).
7. **Runner + settings + TUI:** new tab, env vars, both-sides check, split
   warning, Stage 4 arguments, run records. Update the golden tests
   (`test_runner_golden.py`), settings tests, and the TUI tests for the new
   tab.
8. **Docs:** the ring spec and how to print it, the Scale tab, `scale.json`, in
   `README.md` (this folder) and `../README.md`.
9. **First real capture** (needs the printed ring): run one scan with scaling
   on, inspect detections (save a debug overlay per camera under
   `colmap_<side>/scale_debug/`), tune thresholds, compare the reconstructed
   tablet with a caliper measurement of the tablet itself. Then switch
   `scale_ring_var` to default on.
10. **`rescale_session.py`** (optional).

`python3 -m pytest tests` must pass after every phase, with no COLMAP or GPU
needed.

## Open risks (check on the first real capture)

- rembg keeps the ring: handled by `--crop-plane`; confirm it looks right.
- Low cameras: the band looks thin; confirm inner and outer edges separate.
- Felt fibers or glare on the white top creating extra edges: the band check
  and robust weights should handle it; tune if not.
- A camera model other than those listed in `colmap_model.py`: scaling
  fails cleanly; add it if it ever happens.
