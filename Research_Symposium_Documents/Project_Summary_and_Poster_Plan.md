# Digital Twins of Artifacts — Research Symposium Materials

This folder holds the materials prepared for the research symposium:

- **`Project_Summary_and_Poster_Plan.md`** (this file) — a project summary and the plan for the poster.
- **`Poster/poster.html`** — a 36 × 48 in portrait research poster, self-contained (opens in any browser; exports to PDF for printing; also publishable as a shareable web page).

---

## Poster plan

**Size / orientation:** 36 in wide × 48 in tall, portrait. The poster is built as a
single self-contained HTML file with no external assets, so it works three ways:
1. **On screen** — open `Poster/poster.html` in a browser; the sheet scales to fit the window while keeping the exact 3:4 proportion.
2. **Print / PDF** — use the browser's Print dialog. The page is set to `36in × 48in`, so choosing "Save as PDF" (or a large-format printer) produces a true-size file for the print shop. Set margins to *None* and scale to *100% / Default*.
3. **Shareable link** — the same file is published as a claude.ai artifact for easy on-screen review.

**Sections (top → bottom):**
1. Title banner — title, author line, RIT + Cary affiliation, logo slots.
2. Introduction / Motivation.
3. The Two Systems — how an artifact's shape decides which rig captures it.
4. System A — 2D photometric-stereo rig (papyrus) + rig photo.
5. System B — 3D photogrammetry rig (tablets) + rig photo.
6. Methods / Pipelines — the 5-stage papyrus pipeline and 4-stage tablet pipeline as step flows.
7. Results / Outputs — output screenshots.
8. Interactive Web Viewer — including the hand-tracked magnifier.
9. Technical Specifications — a compact reference table.
10. Conclusion & Future Work.
11. Acknowledgements.
12. Footer — contacts + QR to the website.

**Placeholders you still need to fill** (each is a clearly labeled dashed box in the poster):

| Placeholder | What goes there | Suggested aspect |
|---|---|---|
| Header logos (×2) | RIT logo (left) and RIT Cary Graphic Arts Collection logo (right) | logos, ~3:1 |
| System A photo | Photo of the 2D papyrus capture rig (DSLR + lighting array) | landscape ~4:3 |
| System B photo | Photo of the 3D tablet capture rig (multi-camera + turntable) | landscape ~4:3 |
| Output — papyrus maps | Screenshot of the texture maps (normal / albedo / specular / height) | landscape ~4:3 |
| Output — 3D twin | Screenshot of a finished cuneiform-tablet 3D model | landscape ~4:3 |
| Output — web viewer | Screenshot of the interactive viewer (ideally the hand-tracked magnifier) | landscape ~16:9 |
| Author line | Full author names, and any faculty advisor | one line |
| QR code | Live website URL (provide it and I'll generate the QR into the box) | square |

**How to fill them:** open `poster.html` in a text editor and search for `PLACEHOLDER`.
Each image box has a comment showing exactly where to drop an `<img>` tag. For the QR
code and author names, send me the values and I'll insert them directly.

**One accuracy note:** the poster intentionally does **not** claim multispectral / UV
imaging — the project does not do wavelength-selective imaging. Its spectral technique is
**polarization** (cross-polarized vs co-polarized capture). Please keep this in mind if
editing the copy.

---

## Project summary

### Motivation & significance
Historical artifacts — papyrus manuscripts, cuneiform tablets — are fragile, rare, and
often locked away from the people who want to study them. **Digital Twins of Artifacts**
builds accurate, interactive, web-viewable 3D replicas of these objects so they can be
examined, taught, and displayed without ever handling the original. The project began as
RIT's **Freshman Imaging Project (FIP)**, continued as **Extended-FIP (EFIP)**, and is
carried out in collaboration with the **RIT Cary Graphic Arts Collection**, which supplies
the artifacts.

### Two capture systems
The project pairs each class of artifact with a purpose-built rig. Which one an object uses
depends on whether its **shape carries meaning in three dimensions**.

- **System A — 2D photometric-stereo rig (papyrus & flat manuscripts).** A Canon DSLR is
  tethered over USB and paired with an Arduino-driven directional lighting array with lights
  at the four compass points (N/E/S/W) set at an elevation of θ ≈ 37°. A stepper-rotated
  polarizer lets the rig shoot each light direction twice: **cross-polarized** (specular
  highlights suppressed → clean diffuse/geometry) and **co-polarized** (highlights kept →
  reveals ink). Rather than reconstruct geometry, it recovers how the surface *responds to
  light*, revealing texture, ink, and fine relief. A "scan both sides" mode captures front
  and back.

- **System B — 3D photogrammetry rig (cuneiform tablets & objects).** A multi-camera rig
  coordinated by an Arduino-driven turntable photographs the artifact from many angles.
  GPU-accelerated (CUDA) structure-from-motion and dense stereo reconstruct true geometry,
  and the two sides of the object are merged into one watertight model.

### Software pipelines
- **Papyrus — five-stage photometric-stereo pipeline (Python).**
  (0) ML segmentation isolates the fragment from the background;
  (1) flat copy-paper calibration removes vignetting and light falloff;
  (2) core maps are derived — normal, diffuse (albedo), specular, and roughness — using the
  cross-/co-polarized pairs (ink absorbs strongly; papyrus fibres depolarize);
  (3) a height map is integrated from the surface normals via a weighted
  Frankot–Chellappa method;
  (4) maps are prepped and baked into a `render.glb` with three.js for the web viewer.

- **Tablets — four-stage photogrammetry pipeline (Python / COLMAP / Open3D).**
  (1) background removal on every photo;
  (2) CUDA-accelerated COLMAP structure-from-motion + dense stereo → point cloud;
  (3) FPFH-feature + RANSAC global registration, refined with ICP, merges the two sides;
  (4) Open3D Poisson surface reconstruction → a clean textured mesh (`model.gltf`).

### Outputs
Calibrated papyrus texture maps (normal / albedo / specular / roughness / height), fused
point clouds and watertight tablet meshes, and finished baked "digital twins" (`.glb` /
`.gltf`) served through the project's web gallery.

### Interactive web viewer
A static three.js gallery lets anyone rotate and inspect each twin in the browser, with
adjustable lighting and swappable backgrounds. Its standout feature is a webcam
**hand-tracked magnifying lens** (built on Google MediaPipe hand-landmark tracking): a
visitor moves a fingertip over the artifact and a magnifier follows it — a museum-style
interactive exhibit.

### Key technical specifications
- Camera: Canon DSLR, tethered; RAW `.cr2` → 16-bit linear TIFF (~5208 × 3476 px).
- Lighting: 4 directional lights (N/E/S/W) at θ ≈ 37°; polarizer rotated by stepper.
- Polarization: cross-pol (diffuse/geometry) + co-pol (ink/specular) pairs.
- Height integration: weighted Frankot–Chellappa.
- 3D reconstruction: CUDA COLMAP (SfM + MVS) → FPFH/RANSAC + ICP alignment → Open3D Poisson meshing.
- Rendering / viewer: three.js `MeshPhysicalMaterial` baked to GLB; static three.js viewer with MediaPipe hand tracking.

### People & collaborators
Carter Laubach and Iris (Rochester Institute of Technology), in collaboration with the
**RIT Cary Graphic Arts Collection**. Contacts: `cjl6825@rit.edu`, `isa4049@rit.edu`.
Repository: <https://github.com/LuckyStrix/Digital-Twins-of-Artifacts>.
