# Plan: replace the Tkinter GUI with a Textual (terminal UI) app

Branch: `textual-app` (cut from `main` after `icp-refine` was merged).
Working dir for everything below: `3D/modelingPipeline/`.

## Goal

Replace `app.py` (≈2,400-line Tkinter GUI) with a Textual app that runs in the
terminal (Windows Terminal under WSL today; any Linux terminal / SSH too) and is
**functionally identical** to the current GUI. **No Tk fallback is kept** — when
the new app reaches parity, `app.py`'s Tk code is deleted.

A visual reference exists: `textual_mockup.py` (standalone, fake pipeline; run
`python3 textual_mockup.py` after `pip install textual`). Match its look: left
sidebar of stage cards (rounded borders, blue = running, green = done, red =
failed), tabbed settings on the right, log pane at the bottom, footer key hints,
Windows Terminal "Campbell" theme (plus One Half Dark etc. via `t` / ctrl+p).
**Delete `textual_mockup.py` in the final commit.**

## Decisions already made by the user

- Runs inside **WSL**, launched from Windows Terminal, like today. Must also work
  on plain Linux (desktop or SSH) — Windows-only extras switch off automatically.
- **No Tk fallback.** Remove the Tk GUI once parity is reached.
- **Log pane must be resizable**: drag handle + `ctrl+↑/↓` + a key (`l`) to
  maximise/restore the log. Persist the chosen height.
- **Browse must reach any folder**, not just `$HOME`: location list (`/`,
  `/mnt/*` drives, network mounts, home), an editable path box that also accepts
  Windows paths (`D:\scans`, `\\server\share`, converted with `wslpath -u`),
  recent folders, and (under WSL) a **"Windows dialog…"** button that opens the
  native Explorer folder picker. Suggested default: native dialog under WSL,
  in-terminal tree elsewhere.
- **Every dropdown must list the real options** (e.g. all 19 rembg models).
  Editable comboboxes (artifact Type) become an `Input` with a suggester so
  custom values still work.

## Architecture

```
3D/modelingPipeline/
  app.py                 # becomes the Textual entry point (keep the name: docs/README say `python3 app.py`)
  pipeline/              # NEW — no UI imports allowed in here
    __init__.py
    settings.py          # registry of every setting + load/save app_defaults.json
    options.py           # choice lists (rembg models, camera models, ...)
    parsers.py           # PhotosParser, ColmapParser, AlignParser, ReconParser (moved verbatim)
    paths.py             # session/processed/masks/colmap/merged/recon path helpers, detect_sides
    runner.py            # stage runners + subprocess mgmt + stop; emits events via callbacks
    platform.py          # WSL detection, wslpath, viewer launch, open-folder, native folder dialog
  tui/                   # NEW — Textual UI
    app.py               # App subclass, layout, bindings, themes
    widgets.py           # StageCard, LogPane + drag handle, SettingsForm (built from registry)
    screens.py           # FolderPicker, FilePicker, IcpDetails, PhotoGallery, confirm dialogs
    app.tcss             # styles
  tests/                 # NEW — pytest
```

### `pipeline/settings.py` — single source of truth

One declarative entry per setting, e.g.

```python
Setting("erode_px_var", int, 0, tab="Inputs", section="Background removal",
        label="Erode mask (px)", min=0, max=100, hint="0=off, shrinks mask inward")
```

Fields: key, type (int/float/str/bool/choice/path/text), default, tab, section,
label, hint, long help text (the wrapped grey paragraphs), min/max, choices,
editable-choice flag, persisted flag. The Textual forms are **generated** from
this list, so adding a setting later is one line.

**Keys must stay exactly the `_PERSISTED_VARS` names in the current `app.py`
(`ConfigPanel._PERSISTED_VARS`)** so existing `app_defaults.json` files load
unchanged. Non-persisted per-run fields (input/output dirs, intrinsics file,
PLY A/B overrides, artifact metadata name/type/description/link/link label)
live in the registry with `persisted=False`, as today.

Defaults: use the **code defaults from `app.py`** (they differ from
`app_defaults.json` in places — e.g. `bg_var` code default "white" vs saved
"black"); `app_defaults.json` overrides them at load, exactly as today. Loading
must ignore unknown keys and bad values (today: silently skipped).

Settings access should be a plain object/dict of values so `runner.py` can
build commands without any UI.

### `pipeline/runner.py`

Port `App._run_stage_1_bg_removal` … `_run_stage_4_reconstruction`,
`_colmap_one_side`, `_colmap_flat`, `get_colmap_env`, `get_recon_cmd`,
`_run_proc`, `stop_all`, `run_stage`, `run_all` **with identical command lines,
env vars and fallbacks**. Interface sketch:

```python
class PipelineRunner:
    def __init__(self, settings, on_log, on_stage_state, on_progress): ...
    def run_stage(idx) / run_all() / stop()
```

Callbacks are called from a worker thread; the TUI marshals them with
`app.call_from_thread`. Keep a single-run lock ("Another stage is already
running").

**Improvement to make while porting:** today Stop calls `proc.kill()` on the
top-level process only; for stage 2 that is `bash run.sh`, so COLMAP children
can keep running. Start subprocesses with `start_new_session=True` and stop
with `os.killpg(proc.pid, SIGTERM)` then SIGKILL after a short grace period.

## Behaviour inventory — must all survive the port

Line numbers refer to `app.py` on `main` at the branch point.

**Inputs tab** (`_build_io`, ~L658)
- Input dir / Output dir with browse. On input change (`_on_input_changed`):
  if output is empty, set it to `<input>_recon` next to the input; run
  `detect_sides()`; set side1/side2 and the "Detected sides…" status line
  (2+ sides / 1 side "no alignment stage" / flat structure); then reconcile
  stage states. Output change → reconcile.
- Artifact info for `info.txt`: Name, Type (editable combobox: tablet,
  papyrus), Description (multi-line → Textual `TextArea`), Link, Link Label;
  "Leave Name blank to skip writing info.txt."
- Side folders: Primary, Secondary ("blank = skip alignment").
- Background removal: Background (white/black/transparent), Hard mask
  cutoff, Black/White/Value/Chroma thresholds (0–255, 0=off), Edge band,
  Erode mask, Grow by colour, Convex-hull fill, Passes (1–5), Seg. quality
  slider 10–100 % (no slider in Textual core — use an int input with
  validation or a small custom slider), rembg model (19 options).
  Keep all the grey help paragraphs (collapsible `Collapsible`/tooltip is fine).

**COLMAP tab** (`_build_colmap`, ~L916): Quality (extreme/high/medium/low), Use
GPU, GPU index (−1=auto), Image scale, Image stride (1–10), SIFT (max
features, peak, edge, domain-size pooling, affine), Matching (guided, max
matches), Threads/cache ×6 (0=auto → env var only set when >0), Camera
intrinsics (share between sides, camera model incl. "(COLMAP default)" which
means "don't set", intrinsics JSON file with file picker), Secondary camera
(rotate deg/axis, extra rotate XYZ, translate XYZ, align mode auto/manual).

**Reconstruct tab** (`_build_recon`, ~L1047): all spinboxes with their
ranges/increments; **Hole reduction checkbox applies a preset** to density
trim / crop scale / normal max NN (`HOLE_REDUCTION_ON/OFF`, ~L1116) when
toggled; hole filling (ratio, passes, fill, smooth); mesh cleanup; simplified
export; pose normalization.

**Alignment tab** (`_build_align`, ~L1234): method, voxel ("0" = don't pass),
sample points; ICP refinement toggle + stage dists / iters / dense points /
tolerance / seam band / robust σ; "Show ICP details…"; seam resolution toggle
+ mode / conflict gap / confidence floor / window / min other pts / passes;
PLY A/B overrides with file pickers.

**Stage cards** (`StageWidget`, ~L460): 4 stages, each with status text,
progress (indeterminate until first parser progress, then determinate), **Run
Stage** button and **View** button (enabled when done): View Photos / View
Cloud / View Merged / View Mesh. Plus Run All + Stop.

**Stage state reconcile** (`_reconcile_stage_states`, ~L1822): on startup and
on input/output change, mark stages "done" whose expected output exists
(`_expected_output_for_stage`).

**Log**: timestamps on app messages, command header lines (`$ cmd…`) styled,
subprocess stdout+stderr streamed. Auto-scroll. Consider a line cap for very
long COLMAP runs.

**View actions** (`view_stage`, `_spawn_viewer`, `resolve_viewer_launch`,
~L424/1943): stage 1 → photo gallery; others → `viewer.py` via native Windows
Python under WSL (with open3d) else venv python + WSLg env workaround; watch
the viewer process and report "Failed creating OpenGL window" / non-zero exit.
In the TUI, when no display is available (SSH, no `DISPLAY`/`WAYLAND_DISPLAY`
and not WSL), show a notification with the path instead of launching.

**Photo gallery** (`PhotoGallery`, ~L1517): default = open the processed
folder in Explorer (WSL) / `xdg-open` (Linux desktop) / show path (SSH).
Optional stretch: inline thumbnails via `textual-image` (sixel works in
Windows Terminal ≥1.22) — only if cheap.

**ICP details** (`IcpDetailsWindow`, ~L1577): modal screen with the params
line, before/after metrics table, per-stage table (`DataTable`), convergence
plot and residual histogram (use `textual-plotext`, or Sparkline + bar
characters). If the session has no `icp_report.json`, open a file picker.

**Menu → keybindings**: Open session folder, Save current settings as
default (confirm via notification; errors via notification), Quit.

**Stage-specific logic to keep verbatim** (see runners, ~L2036–2345):
stride applied in stage 1 and `FIPMESH_COLMAP_IMAGE_STRIDE=1` for COLMAP;
optional stage-1 flags only added when non-default; `--grow-hull-fill` only
with grow-chroma > 0; seg-scale only when < 100; stage 2 split 0–50 / 50–100
progress for two sides; intrinsics in/out env vars and the sharing rules;
run.sh called with paths relative to `SCRIPT_DIR`; per-side mask dir `-m`;
stage 3 skip for single side; seam/refine flag fallbacks (`or "0.3"` etc.);
`--report` path; stage 4 `--prune-fill-color` from background, optional
`--side1-camera-centers`, copy gltf → `model.gltf` and simplified glb →
`model_simplified.glb` with `shutil.copyfile` (network drives), write
`info.txt` via `src.artifact_info.write_info_txt` when Name is set.

**Known cosmetic bug to fix while porting parsers:** in `AlignParser`,
`=== resolve seam ===` (98) comes after `[icp] after` (98) in the real output
order (refine runs before seam), so the seam step never updates the label.
Give seam 99 or reorder.

## Textual notes (from building the mockup, Textual 8.2.x)

- `compact=True` exists on `Input`, `Select`, `Button`, `Checkbox` — **not**
  on `Switch`; use compact `Checkbox` for booleans in dense forms.
- `DataTable.update_cell(row_key, col_key, …)` needs columns added with
  `add_column(label, key=…)` (and an explicit `width=` or the column won't
  grow on update).
- Register custom themes with `App.register_theme(Theme(...))`.
- Long-running work: `@work(thread=True)` + `call_from_thread`, or asyncio
  subprocesses; one exclusive worker group for the pipeline.
- No built-in splitter: implement the log drag handle as a 1-row widget that
  captures the mouse on press and sets the log container's `styles.height`
  on mouse move.
- Native Windows folder dialog from WSL: run `powershell.exe -NoProfile -STA
  -Command` with `System.Windows.Forms.FolderBrowserDialog`, read stdout,
  convert with `wslpath -u`. Run it in a worker so the UI stays live.

## Phases (commit after each; keep the app runnable at every commit)

1. **Extract `pipeline/`** — parsers, paths, options, settings registry,
   platform helpers, runner. Add pytest tests: parsers fed with sample log
   lines; settings load/save round-trip with the current `app_defaults.json`;
   command/env construction for each stage compared against the exact
   commands the current `app.py` builds for the same settings (write these
   golden tests *before* deleting the Tk code).
   **Done.** Notes for the next phases:
   - `Settings` is a dict (`Settings.load()` / `.save()`); the form builder
     iterates `settings.REGISTRY` (field `widget="slider"` marks seg quality,
     `validate=` hints numeric `str` fields, `editable=True` marks the Type
     combobox). Hole-reduction presets: `settings.hole_reduction_preset(on)`.
   - `PipelineRunner(settings, on_log, on_stage_state, on_progress)`:
     `run_stage(idx)` / `run_all()` start a thread and return `False` when a
     run is already active; `stop()`. `on_log(text, kind)` with kind
     `info` (timestamped), `header` (`$ cmd`), `output`.
   - `paths.SessionPaths(settings)` (expected outputs / readiness for
     reconcile), `paths.describe_sides()` for the "Detected sides…" line.
   - `platform`: `has_display()`, `open_folder()`, `launch_viewer()` +
     `wait_viewer()`, `native_folder_dialog()` / `native_file_dialog()`,
     `to_posix_path()`, `browse_locations()`.
   - The Tk `app.py` now imports parsers/helpers from `pipeline/`; its own
     command building stays until phase 5 (golden tests cover it).
   - Tests: `python3 -m pytest tests` from this folder.
2. **Textual shell** — `tui/` layout, themes, generated settings forms,
   stage cards, log pane (resizable), footer bindings, save/load defaults,
   input-change behaviour, reconcile. **Done.** UI prefs (log height,
   theme, recent folders, "Windows dialog first") live in the gitignored
   `.tui_state.json`, not in `app_defaults.json`. Deliberate differences
   from Tk: input-folder changes are debounced and ignored until the path
   is an existing folder (Tk reacted per keystroke, so typing a path set
   the output folder from its first character); Windows paths typed into
   path fields are converted when the field is left.
   After review: an output folder the app filled in follows later input
   changes (one the user set is kept); quoted pasted paths are unquoted;
   out-of-range numbers are accepted as in Tk (spinbox ranges only limited
   the arrows) but unparseable ones block Run/Save; runs use a snapshot of
   the settings taken at start; bare `r`/`s`/`q` are ignored while a
   settings field has focus (`ctrl+r` runs from anywhere); closing the
   terminal (SIGHUP/SIGTERM) or quitting kills running stage processes.
3. **Wire the runner** — run stage / run all / stop (process-group kill),
   live progress + log. **Done.** Runner events go through a queue drained
   every 50 ms. Quitting while a stage runs asks first.
4. **Pickers and extras** — folder/file pickers (locations, path box,
   recents, Windows dialog), view actions, open session folder, photo
   gallery, ICP details. **Done.** Under WSL the picker opens the Windows
   dialog first and falls back to the in-terminal browser on cancel. The
   photo gallery opens the folder (no inline thumbnails). ICP plots use
   block-character sparklines/histogram (no textual-plotext).
5. **Cut over** — `app.py` becomes the Textual entry point; delete Tk code
   and `textual_mockup.py`; headless UI tests with `App.run_test()` (drive a
   fake runner through all four stages, assert card states, log contents,
   saved defaults). **Done.**
6. **Docs/setup** — add `textual` (and `textual-plotext` if used) to
   `requirements.txt`; update `README.md` (L4, L62–79, L321 describe the Tk
   GUI), add "Running on Linux / over SSH"; `SETUP.md` L116–122: WSLg is no
   longer needed for this app, but **keep `python3-tk`** because the capture
   GUI still uses Tk (`SETUP.md` L38/L122). **Done** (also `3D/README.md`,
   which described the Tk GUI too). What's left from the definition of done
   is the manual check in Windows Terminal under WSL with a real dataset.

## Definition of done

- Every item in the behaviour inventory works in the TUI.
- Golden command/env tests pass; `app_defaults.json` loads and saves with the
  same keys.
- Headless `run_test` suite passes; manual check in Windows Terminal under
  WSL by the user with a real dataset.
- No Tk imports left in the reconstruction app.

## Testing tips

- Screenshot during headless tests with `app.save_screenshot("x.svg")`; on
  Windows, render SVG → PNG with headless Edge
  (`msedge --headless --screenshot=out.png --window-size=1900,1250 file:///…svg`)
  to eyeball layouts.
- Real pipeline stages need WSL + CUDA + COLMAP; unit tests must not require
  them.
