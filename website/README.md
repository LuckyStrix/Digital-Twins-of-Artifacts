# Website — artifact gallery

The artifact gallery site. Static HTML/CSS/JS — no build tooling, no
server-side code required to host it.

## Structure

```
index.html          Homepage: header, artifact gallery, MISHA-Imaged Artifacts, "Our Systems", contact footer
viewer.html          Generic 3D viewer for a single artifact (?id=<artifact-folder-name>)
msi.html             Multispectral band viewer for a single artifact (?id=<artifact-folder-name>)
css/style.css         All styles for all pages
js/main.js            Loads artifacts/manifest.json and renders the gallery cards
js/viewer.js          Loads a model + its metadata, lighting, optional background, and a webcam hand-tracked magnifying lens
js/msi.js             Loads a MISHA band stack + its metadata into the wavelength-equalizer viewer
js/artifactInfo.js    Shared info-panel logic (manifest fetch, populate panel, cross-links) used by viewer.js and msi.js
assets/               Fonts, icons
artifacts/            One folder per artifact (model and/or msi/ + txt) — see artifacts/README.md
backgrounds/          One folder per cube-map background — see backgrounds/README.md
tools/build_manifest.py     Scans artifacts/ and backgrounds/, writes their manifest.json files
tools/build_msi_assets.py   Converts a raw MISHA multispectral export into an artifact's msi/ folder — see tools/requirements.txt
```

`msi.html`/`js/msi.js` display data from **MISHA**, an external, independent
imaging project (see `artifacts/README.md`'s `== msi/ ==` section) — not
something captured by this repo's own 2D/3D rigs.

## Adding artifacts

See `artifacts/README.md` for the exact `.txt` format and supported
model formats (`.glb`, `.gltf`, `.obj`+`.mtl`). Short version: drop a
folder with a model file and a `.txt` into `artifacts/`, then run:

```
python3 tools/build_manifest.py
```

This rewrites `artifacts/manifest.json` and `backgrounds/manifest.json`,
which the homepage and viewer read at load time. Re-run it any time
artifacts or backgrounds are added, removed, or edited, and before
deploying.

## Adding MISHA multispectral data

MISHA is an **external, independent project** (see the top-level
[`README.md`](../README.md) and `artifacts/README.md`'s `== msi/ ==`
section) — this only covers turning a raw MISHA export you've been given
into web assets, not capturing new data yourselves.

Install `tools/requirements.txt` (a separate, small imaging-library stack —
`build_manifest.py` itself stays dependency-free), then run:

```
pip install -r tools/requirements.txt
python3 tools/build_msi_assets.py <raw-export-dir> <artifact-id>
```

This writes `artifacts/<artifact-id>/msi/bands/*.webp` +
`msi_manifest.json`. Keep the raw export itself out of the repo — put it
under the gitignored `data/` folder (e.g. `data/msi/<artifact-id>-raw/`),
the same convention the 2D/3D pipelines use for their own raw captures.
Then, optionally, hand-write `artifacts/<artifact-id>/msi/recipes.json`
with curator-picked equalizer presets, and rerun `tools/build_manifest.py`
as above.

## Adding backgrounds

The viewer's "Background" panel lets a visitor toggle on a cube-map
backdrop behind the model. See `backgrounds/README.md` for the face
naming convention (`px`/`nx`/`py`/`ny`/`pz`/`nz`) — drop the six images
into a folder under `backgrounds/` and rerun the build script above.

## Running locally

From this folder:

```
python3 -m http.server 8000
```

then open `http://localhost:8000/`.

## Deploying

Upload the contents of this folder as-is to any static host. Just make
sure `artifacts/manifest.json` is up to date (run the build script)
before you upload.

## Viewer internals: on-demand rendering

The viewer draws a frame only when something changes, rather than redrawing
every rAF tick whether or not anything moved. An artifact left open on screen
costs nothing.

**How it works.** `requestRender()` in `js/viewer.js` marks the next frame as
needed (slider moved, model loaded, texture arrived, window resized).
`isAnimating()` covers things mid-transition that keep changing on their own
(flip tween, hand tracking, lens fade-out, reveal-light lerp). Camera motion
needs neither — both control types fire a `change` event wired to
`requestRender()`, and because damping keeps firing it until the camera
settles, inertia is handled for free. `controls.update()` still runs every
tick; only the draw is skipped.

**Measured** (WebGL draw calls, `sample-papyrus`, software renderer),
against the earlier draw-every-tick viewer:

| | draw every tick | on-demand |
|---|---|---|
| idle, 4 s | 108 | **0** |
| idle again after interaction, 4 s | 108 | **0** |
| wheel zoom, 2 s | 58 | 2 |

Output is pixel-for-pixel identical to the draw-every-tick viewer on both
`sample-papyrus` and `mainDemotablet` (0 differing pixels).

**If you extend this viewer**, anything that changes what is on screen must
call `requestRender()`, or a continuously-changing thing must be added to
`isAnimating()`. A missing call looks like a frozen viewer; a spurious one
costs a single redraw. When unsure, call it.

## To fill in

- `backgrounds/demo-colors/` is a placeholder cube map (six flat
  colors) so the Background panel has something to show — delete it
  once you've added real cube maps, then rerun the build script.
