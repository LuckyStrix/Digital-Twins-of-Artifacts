# Fully Rewritten

A from-scratch rebuild of the artifact gallery site. Static HTML/CSS/JS —
no build tooling, no server-side code required to host it.

## Structure

```
index.html          Homepage: header, artifact gallery, "Our Systems" (blank), contact footer
viewer.html          Generic 3D viewer for a single artifact (?id=<artifact-folder-name>)
css/style.css         All styles for both pages
js/main.js            Loads artifacts/manifest.json and renders the gallery cards
js/viewer.js          Loads a model + its metadata, lighting, optional background, and a webcam hand-tracked magnifying lens
assets/               Fonts, icons
artifacts/            One folder per artifact (model + txt) — see artifacts/README.md
backgrounds/          One folder per cube-map background — see backgrounds/README.md
tools/build_manifest.py   Scans artifacts/ and backgrounds/, writes their manifest.json files
```

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

## To fill in

- `index.html` — `#project-name` and `#project-description` in the
  header, and the contact info in the footer (`#contact`), are
  placeholder text marked with `[ ... ]`. Replace them.
- `#systems` in `index.html` is an intentionally empty section for
  information about your capture/modeling systems.
- `artifacts/example-papyrus-demo/` is a placeholder artifact so the
  gallery isn't empty on first load — delete it once real artifacts
  are in place, then rerun the build script.
- `backgrounds/demo-colors/` is a placeholder cube map (six flat
  colors) so the Background panel has something to show — delete it
  once you've added real cube maps, then rerun the build script.
