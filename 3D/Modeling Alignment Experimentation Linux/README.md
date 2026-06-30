# Tablet half-scan alignment

Aligns two open half-scans of a cuneiform tablet (each ~60% of the object, each
with an open face the other has detail on) into a single merged PLY. Two methods,
both finished with ICP; a small GUI lets you eyeball each one.

## Setup (already done here)
A `venv/` with Open3D 0.19 + NumPy is in this folder. If it ever needs rebuilding:

    python3 -m venv venv
    ./venv/bin/python -m pip install open3d numpy

## GUI
    ./venv/bin/python gui.py

1. Browse to **Mesh A** (fixed/target) and **Mesh B** (moving/source).
2. **Load meshes**, then **Run Opening + ICP**, **Run FPFH + ICP**, or **Run Both**.
3. Read the fitness (higher = better, 1.0 = every point found a match) and rmse.
4. **View** opens an Open3D window: green = A, red = B. Good alignment = the two
   colors interleave through the overlapping side walls. **View initial** shows
   the raw unaligned pair for comparison.
5. **Save merged** writes one combined PLY in A's original coordinate frame.

(Drag to rotate, scroll to zoom, `q`/close to return to the control panel.)

## Headless / batch
    ./venv/bin/python run.py A.ply B.ply -o merged.ply --method both [--view]

## The three methods
All three end the same way: enumerate a few candidate poses, ICP each, and pick
the winner by **closedness** (see below) — not by raw ICP fitness.

- **opening** — finds each open face (area-weighted normal sum *and* boundary-rim
  centroid), builds a PCA frame, then makes the open faces face each other and
  slides them together so the rim seams coincide (the "shift inward", sized
  automatically). Enumerates the discrete ambiguities (long-axis sign, winding sign).
- **fpfh** — feature-based global registration (FPFH + RANSAC) on the overlapping
  side-wall detail, then 180° flips about the seam to escape the collapse (below).
  Needs no open-face estimate.
- **collapse** — deliberately stacks the two halves on top of each other facing
  the same way (the easy, reliable overlap-maximum pose), then flips one half 180°
  about the seam to swing it to the correct side. Simplest and very robust.

## Why "closedness", not fitness
These halves are two caps of the *same* surface, so the highest-fitness pose is a
**collapse**: one half stacked onto the other facing the same way (every point
finds a match → fitness ≈ 1.0, but it's wrong). The correct nested assembly only
shares the thin overlap band → fitness ≈ 0.25–0.3. So fitness is *backwards* here.
Instead each candidate is scored by **closedness**: the correct assembly completes
the closed tablet surface, so the merged cloud's unit normals nearly cancel
(closed → 1.0); a collapse leaves an open double-cap whose normals don't cancel
(closed ≈ 0.35). The GUI/CLI report `closed` and choose on it. closed→1 = correct.

## Two ambiguities, two signals: `closed` and `detail`
Getting the halves on the right *side* (not collapsed) is only half the battle.
Among the correctly-facing poses there is still a **180° in-plane spin** about the
seam-normal axis: the tablet closes equally well both ways, so `closed` can't tell
them apart. The only thing that differs is whether the actual surface **detail**
(cuneiform on the overlapping walls) lines up. So selection is two-stage:
1. `closed` gates out the collapse (wrong side).
2. `detail` — fine-scale overlap fitness at a tight threshold — breaks the spin:
   the wrong spin misregisters the detail and loses fine inliers.
Both are reported. **closed→1 = right side, detail→1 = right spin.** (On a perfectly
smooth/symmetric object the spin is genuinely ambiguous and `detail` ties — but a
detailed tablet breaks it cleanly.)

## Tuning
- **voxel** (`auto` = ~1% of mesh size): smaller = finer/slower, larger = coarser.
  If FPFH misfires, try nudging this up or down ~2x.
- **sample pts**: points sampled from each mesh for registration. More = finer ICP.

## Caveat: scale
Both methods assume the two scans share a metric scale. Photogrammetry pipelines
that solve each scan independently (e.g. COLMAP) are scale-free, so if nothing
ever aligns, the halves are probably at different scales — rescale one first.
