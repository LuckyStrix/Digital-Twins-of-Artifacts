"""Modal screens: confirm dialog, folder/file pickers, ICP details."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable, Iterator

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Checkbox, DataTable, DirectoryTree, Input, OptionList, Static, Tree,
)
from textual.widgets.option_list import Option

from pipeline import platform as plat
from pipeline.paths import is_dir, is_file


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel"),
                Binding("y", "dismiss(True)", "Yes", show=False),
                Binding("n", "dismiss(False)", "No", show=False)]

    def __init__(self, title: str, message: str, yes: str = "Yes", no: str = "Cancel",
                 yes_variant: str = "error"):
        super().__init__()
        self._title, self._message, self._yes, self._no = title, message, yes, no
        self._yes_variant = yes_variant

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm", classes="dialog"):
            yield Static(self._title, classes="dialog-title")
            yield Static(self._message, classes="dialog-body")
            with Horizontal(classes="dialog-btns"):
                yield Button(self._yes, variant=self._yes_variant, id="yes", compact=True)
                yield Button(self._no, id="no", compact=True)

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    @on(Button.Pressed, "#yes")
    def _yes_pressed(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no_pressed(self) -> None:
        self.dismiss(False)


class NewFolderScreen(ModalScreen["str | None"]):
    """Ask for a folder name; returns it, or None on cancel."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def __init__(self, parent: Path):
        super().__init__()
        self.parent_dir = parent

    def compose(self) -> ComposeResult:
        with Vertical(id="new-folder", classes="dialog"):
            yield Static("New folder", classes="dialog-title")
            yield Static(f"In {self.parent_dir}", classes="dialog-body")
            yield Input(id="folder-name", compact=True, placeholder="Folder name")
            with Horizontal(classes="dialog-btns"):
                yield Button("Create", variant="primary", id="create", compact=True)
                yield Button("Cancel", id="cancel", compact=True)

    def on_mount(self) -> None:
        self.query_one("#folder-name", Input).focus()

    @on(Input.Submitted, "#folder-name")
    @on(Button.Pressed, "#create")
    def _create(self) -> None:
        name = self.query_one("#folder-name", Input).value.strip()
        if not name:
            self.notify("Type a folder name.", severity="warning")
        elif name in (".", "..") or "/" in name or "\\" in name:
            self.notify("A folder name can't contain / or \\.", severity="warning")
        else:
            self.dismiss(name)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)


# ── folder / file picker ──────────────────────────────────────────────────────

class _FilteredTree(DirectoryTree):
    """DirectoryTree that filters, lists folders quickly, and opens on click.

    Clicking a folder only ever expands it; collapse with its arrow, space or
    left. (With Textual's toggle-on-select, a second click made while a slow
    folder was still loading queued up behind the loader and collapsed it
    again, so it took a third click to see the contents.)"""

    def __init__(self, path, *, dirs_only: bool, pattern: str = "", **kw):
        self.dirs_only, self.pattern = dirs_only, pattern
        self._is_dir_cache: dict[Path, bool] = {}
        super().__init__(path, **kw)
        self.auto_expand = False

    @on(Tree.NodeSelected)
    def _expand_on_select(self, event: Tree.NodeSelected) -> None:
        if event.node.allow_expand and not event.node.is_expanded:
            event.node.expand()

    def _directory_content(self, location: Path, worker) -> Iterator[Path]:
        # scandir says whether each entry is a folder without a stat per entry
        # (DirectoryTree stats every entry three times, which is slow on
        # /mnt/c and network drives); remember the answer for _safe_is_dir.
        try:
            with os.scandir(location) as it:
                for entry in it:
                    if worker.is_cancelled:
                        break
                    try:
                        entry_is_dir = entry.is_dir()
                    except OSError:
                        entry_is_dir = False
                    p = location / entry.name
                    self._is_dir_cache[p] = entry_is_dir
                    yield p
        except OSError:
            pass

    def _safe_is_dir(self, path: Path) -> bool:
        cached = self._is_dir_cache.get(path)
        if cached is not None:
            return cached
        try:
            return path.is_dir()
        except OSError:
            return False

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        out = []
        for p in paths:
            if p.name.startswith("."):
                continue
            is_dir = self._safe_is_dir(p)
            if is_dir or (not self.dirs_only and (
                    not self.pattern or fnmatch.fnmatch(p.name.lower(), self.pattern.lower()))):
                out.append(p)
        return out


def start_dir(initial: str, recents: list[str]) -> Path:
    """First existing folder among the current value and the recents."""
    for cand in [initial] + recents:
        if not cand:
            continue
        p = Path(cand)
        if is_file(p):
            p = p.parent
        if is_dir(p):
            return p
    return Path.home()


class PathPicker(ModalScreen["str | None"]):
    """Browse to a folder (mode "dir") or a file (mode "file").

    Left: locations (home, /, drives under /mnt, network mounts) and recent
    folders. Right: a directory tree. Bottom: an editable path box that also
    takes Windows paths. Under WSL or Windows a "Windows dialog…" button opens the
    native Explorer picker (automatically on open, if enabled)."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel"),
                Binding("ctrl+n", "new_folder", "New folder")]

    def __init__(self, mode: str, title: str, initial: str = "", pattern: str = "",
                 recents: list[str] | None = None, auto_native: bool = False):
        super().__init__()
        self.mode, self.title_text, self.pattern = mode, title, pattern
        self.recents = [r for r in (recents or []) if r]
        self.auto_native = auto_native and plat.has_native_dialogs()
        self.initial = initial
        self.start = start_dir(initial, self.recents)
        self._locations: list[Path] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="picker", classes="dialog"):
            yield Static(self.title_text, classes="dialog-title")
            with Horizontal(id="picker-body"):
                opts: list[Option | None] = []
                for label, p in plat.browse_locations():
                    self._locations.append(p)
                    opts.append(Option(label, id=f"loc-{len(self._locations) - 1}"))
                if self.recents:
                    opts.append(None)
                    for r in self.recents:
                        self._locations.append(Path(r))
                        opts.append(Option(Text(r, overflow="ellipsis", no_wrap=True),
                                           id=f"loc-{len(self._locations) - 1}"))
                yield OptionList(*opts, id="locations")
                yield _FilteredTree(self.start, dirs_only=self.mode == "dir",
                                    pattern=self.pattern, id="tree")
            yield Input(self.initial or str(self.start), id="picker-path", compact=True,
                        placeholder="Type or paste a path (Windows paths work too)")
            with Horizontal(classes="dialog-btns"):
                yield Button("New folder…", id="new-folder-btn", compact=True)
                if plat.has_native_dialogs():
                    yield Checkbox("Open Windows dialog first", self.auto_native,
                                   id="auto-native", compact=True)
                    yield Button("Windows dialog…", id="native", compact=True)
                yield Button("Select", variant="primary", id="ok", compact=True)
                yield Button("Cancel", id="cancel", compact=True)

    def on_mount(self) -> None:
        self.query_one("#tree").focus()
        if self.auto_native:
            self.open_native()

    # ── native dialog ─────────────────────────────────────────────────────────
    @work(thread=True, exclusive=True, group="native-dialog")
    def open_native(self) -> None:
        self.app.call_from_thread(self.notify, "Opening the Windows dialog…", timeout=2)
        start = self.query_one("#picker-path", Input).value.strip()
        if self.mode == "dir":
            result = plat.native_folder_dialog(self.title_text, start)
        else:
            folder = str(start_dir(start, []))
            result = plat.native_file_dialog(self.title_text, folder, self.pattern)
        self.app.call_from_thread(self._native_done, result)

    def _native_done(self, result: str | None) -> None:
        if not self.is_attached or self.app.screen is not self:
            return
        if result:
            self.dismiss(result)
        else:
            self.notify("No selection from the Windows dialog; browse here instead.", timeout=3)

    @on(Button.Pressed, "#native")
    def _native_pressed(self) -> None:
        self.open_native()

    @on(Checkbox.Changed, "#auto-native")
    def _auto_changed(self, event: Checkbox.Changed) -> None:
        ui = getattr(self.app, "ui", None)
        if ui is not None:
            ui.native_dialog = event.value
            ui.save(self.app.ui_state_path)

    # ── browsing ──────────────────────────────────────────────────────────────
    def go(self, p: Path) -> None:
        self.query_one("#tree", _FilteredTree).path = p
        self.query_one("#picker-path", Input).value = str(p)

    @on(OptionList.OptionSelected, "#locations")
    def _location(self, event: OptionList.OptionSelected) -> None:
        p = self._locations[int(event.option.id.split("-")[1])]
        if is_dir(p):
            self.go(p)
        else:
            self.notify(f"Not available: {p}", severity="warning")

    @on(DirectoryTree.DirectorySelected)
    def _dir_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self.query_one("#picker-path", Input).value = str(event.path)

    @on(DirectoryTree.FileSelected)
    def _file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.query_one("#picker-path", Input).value = str(event.path)
        if self.mode == "file":
            self.accept()

    @on(Input.Submitted, "#picker-path")
    def _path_submitted(self) -> None:
        p = self.typed_path()
        if is_dir(p) and self.mode == "file":
            self.go(p)
        else:
            self.accept()

    def typed_path(self) -> Path:
        p = Path(plat.to_posix_path(self.query_one("#picker-path", Input).value))
        try:
            return p.expanduser()
        except RuntimeError:          # ~unknownuser
            return p

    def current_folder(self) -> Path:
        """The folder a new folder goes in: the path box's folder, else the tree's."""
        p = self.typed_path()
        if is_dir(p):
            return p
        if is_dir(p.parent):
            return p.parent
        return Path(self.query_one("#tree", _FilteredTree).path)

    def action_new_folder(self) -> None:
        base = self.current_folder()

        def named(name: str | None) -> None:
            if not name:
                return
            new = base / name
            try:
                new.mkdir()
            except FileExistsError:
                if not is_dir(new):
                    self.notify(f"A file named {name} already exists.", severity="warning")
                    return
            except OSError as e:
                self.notify(f"Could not create {new}:\n{e}", severity="error")
                return
            else:
                self.notify(f"Created {new}", timeout=3)
            tree = self.query_one("#tree", _FilteredTree)
            if Path(tree.path) == base:
                tree.reload()
            else:
                tree.path = base
            self.query_one("#picker-path", Input).value = str(new)

        self.app.push_screen(NewFolderScreen(base), named)

    @on(Button.Pressed, "#new-folder-btn")
    def _new_folder_pressed(self) -> None:
        self.action_new_folder()

    def accept(self) -> None:
        p = self.typed_path()
        if self.mode == "dir" and not is_dir(p):
            if p.is_absolute() and not is_file(p):
                self.offer_create(p)
            else:
                self.notify(f"Not a folder: {p}", severity="warning")
            return
        if self.mode == "file" and not is_file(p):
            self.notify(f"Not a file: {p}", severity="warning")
            return
        self.dismiss(str(p))

    def offer_create(self, p: Path) -> None:
        """A typed folder that doesn't exist yet: offer to create and use it."""
        def answered(yes: bool | None) -> None:
            if not yes:
                return
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                self.notify(f"Could not create {p}:\n{e}", severity="error")
                return
            self.dismiss(str(p))

        self.app.push_screen(ConfirmScreen(
            "Create folder?", f"{p}\ndoesn't exist yet. Create it and use it?",
            yes="Create", yes_variant="primary"), answered)

    @on(Button.Pressed, "#ok")
    def _ok(self) -> None:
        self.accept()

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(None)


# ── ICP details ───────────────────────────────────────────────────────────────

_STAGE_KEYS = ("stage", "mult", "n_src", "n_tgt", "iters", "move_translation", "move_rotation_deg")
_METRIC_KEYS = ("fitness", "rmse", "median_vox", "p95", "mean")


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def icp_report_problem(d) -> str | None:
    """Why `d` can't be shown as an icp_report.json, or None if it can.

    The picker accepts any *.json, so check the shape align.refine_icp()
    writes instead of crashing on some other file."""
    if not isinstance(d, dict):
        return "not a JSON object"
    if "stages" not in d and "before" not in d:
        return "no 'stages' or 'before' section"
    if "voxel" in d and not _is_num(d["voxel"]):
        return "'voxel' is not a number"
    for sec in ("params", "before", "after"):
        if sec in d and not isinstance(d[sec], dict):
            return f"'{sec}' is not an object"
    for sec in ("before", "after"):
        for k in _METRIC_KEYS:
            if k in d.get(sec, {}) and not _is_num(d[sec][k]):
                return f"'{sec}.{k}' is not a number"
    stages = d.get("stages", [])
    if not isinstance(stages, list):
        return "'stages' is not a list"
    for i, s in enumerate(stages):
        if not isinstance(s, dict):
            return f"stage {i + 1} is not an object"
        missing = [k for k in _STAGE_KEYS if not _is_num(s.get(k))]
        if missing:
            return f"stage {i + 1}: missing/invalid {', '.join(missing)}"
        tr = s.get("trace", [])
        if not isinstance(tr, list) or any(
                not isinstance(t, dict) or not _is_num(t.get("rmse")) for t in tr):
            return f"stage {i + 1}: bad 'trace'"
    for sec in ("hist_before", "hist_after"):
        h = d.get(sec)
        if h is None:
            continue
        if (not isinstance(h, dict) or not isinstance(h.get("counts"), list)
                or not all(_is_num(c) for c in h["counts"])
                or ("hi_vox" in h and not _is_num(h["hi_vox"]))):
            return f"bad '{sec}'"
    return None


_BLOCKS = " ▁▂▃▄▅▆▇█"
BEFORE_STYLE = "#94a3b8"


def convergence_rows(stages: list[dict], width: int = 60) -> list[tuple[str, str, str]]:
    """Per stage: (label, sparkline of RMSE per logged iteration on one scale
    shared by all stages, "first → last")."""
    values = [tp["rmse"] for s in stages for tp in s.get("trace", [])]
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    rows = []
    for s in stages:
        tr = [tp["rmse"] for tp in s.get("trace", [])]
        if not tr:
            continue
        if len(tr) > width:                     # downsample to fit
            tr = [tr[int(i * len(tr) / width)] for i in range(width)]
        spark = "".join(_BLOCKS[1 + round((v - lo) / span * (len(_BLOCKS) - 2))] for v in tr)
        rows.append((f"s{s['stage']} {s.get('mult', 0):g} vox", spark,
                     f"{s['trace'][0]['rmse']:.4g} → {s['trace'][-1]['rmse']:.4g}"))
    return rows


def histogram_text(hb: dict, ha: dict, height: int, before_style: str, after_style: str) -> Text:
    """Vertical before/after histogram, two character columns per bin, each
    normalised to its own total (fractions), on a shared vertical scale."""
    nb = len(hb["counts"])
    tb, ta = max(1, sum(hb["counts"])), max(1, sum(ha["counts"]))
    fb = [c / tb for c in hb["counts"]]
    fa = [c / ta for c in ha["counts"]]
    top = max(fb + fa) or 1.0
    t = Text()
    for row in range(height, 0, -1):
        t.append(f"{top:6.1%} │" if row == height else "       │", style="dim")
        for i in range(nb):
            for f, style in ((fb[i], before_style), (fa[i], after_style)):
                level = f / top * height - (row - 1)      # filled part of this cell, 0..1
                idx = max(0, min(8, round(level * 8)))
                t.append(_BLOCKS[idx], style=style)
        t.append("\n")
    t.append("       └" + "─" * (2 * nb) + "\n", style="dim")
    right = f">= {hb.get('hi_vox', 2):g} vox"
    t.append("        0" + " " * max(1, 2 * nb - 1 - len(right)) + right, style="dim")
    return t


class IcpDetailsScreen(ModalScreen[None]):
    """Read-only view of an icp_report.json: before/after summary, per-stage
    table, RMSE convergence per stage and residual histograms."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close"),
                Binding("q", "dismiss(None)", "Close", show=False)]

    def __init__(self, data: dict, path: Path):
        super().__init__()
        self.data, self.path = data, path

    def compose(self) -> ComposeResult:
        d = self.data
        vox = d.get("voxel", 1.0) or 1.0
        pr = d.get("params", {})
        primary = self.app.current_theme.primary
        with Vertical(id="icp", classes="dialog"):
            yield Static(f"ICP details - {self.path.parent.parent.name}", classes="dialog-title")
            with VerticalScroll():
                yield Static(
                    f"method={d.get('method', '?')}   voxel={vox:.5g}   "
                    f"stages={pr.get('thresholds')}   iters/stage={pr.get('max_iters')}   "
                    f"points={pr.get('points')}   band={pr.get('band')}   robust={pr.get('robust')}",
                    id="icp-params")
                yield DataTable(id="icp-summary", cursor_type="none", zebra_stripes=True)
                yield DataTable(id="icp-stages", cursor_type="none", zebra_stripes=True)
                yield Static("Convergence: RMSE at each logged iteration, one row per stage, all "
                             "on one scale. A flat run means ICP has converged at that stage's "
                             "distance.", classes="help-body")
                conv = Text()
                for label, spark, rng in convergence_rows(d.get("stages", [])):
                    conv.append(f"{label:>12}  ", style="dim")
                    conv.append(spark, style=primary)
                    conv.append(f"  {rng}\n", style="dim")
                yield Static(conv or Text("(no iterations logged)", style="dim"), id="icp-conv")
                yield Static(Text.assemble(
                    "Residual histogram (|point-to-plane|, in voxels): ",
                    ("before", BEFORE_STYLE), " = grey, ", ("after", primary),
                    " = blue. A doubled surface shows as mass away from 0."),
                    classes="help-body")
                hb, ha = d.get("hist_before"), d.get("hist_after")
                if (hb and ha and hb.get("counts") and ha.get("counts")
                        and len(hb["counts"]) == len(ha["counts"])):
                    yield Static(histogram_text(hb, ha, 10, BEFORE_STYLE, primary), id="icp-hist")
                else:
                    yield Static(Text("(no histogram in report)", style="dim"), id="icp-hist")
            with Horizontal(classes="dialog-btns"):
                yield Button("Close", id="close", compact=True)

    def on_mount(self) -> None:
        d = self.data
        vox = d.get("voxel", 1.0) or 1.0
        b, a = d.get("before", {}), d.get("after", {})
        sm = self.query_one("#icp-summary", DataTable)
        sm.add_column("metric (eval dist = 2 voxels)", key="m", width=34)
        sm.add_column("before refine", key="b", width=14)
        sm.add_column("after refine", key="a", width=14)
        for label, key, fmt in (("fitness (inlier fraction)", "fitness", "{:.4f}"),
                                ("inlier RMSE", "rmse", "{:.5g}"),
                                ("median |point-to-plane| (voxels)", "median_vox", "{:.4f}"),
                                ("95th pct |point-to-plane|", "p95", "{:.5g}"),
                                ("mean |point-to-plane|", "mean", "{:.5g}")):
            sm.add_row(label,
                       Text(fmt.format(b[key]) if key in b else "-", justify="right"),
                       Text(fmt.format(a[key]) if key in a else "-", justify="right"))
        st = self.query_one("#icp-stages", DataTable)
        for label, key, w in (("stage", "s", 6), ("dist (vox)", "t", 10), ("src/tgt pts", "n", 18),
                              ("iters", "i", 6), ("final rmse", "r", 11), ("moved (vox)", "mv", 11),
                              ("rotated (deg)", "rot", 13)):
            st.add_column(label, key=key, width=w)
        for s_ in d.get("stages", []):
            last = s_["trace"][-1] if s_.get("trace") else {}
            vals = (s_["stage"], f"{s_['mult']:g}", f"{s_['n_src']}/{s_['n_tgt']}", s_["iters"],
                    f"{last.get('rmse', float('nan')):.5g}",
                    f"{s_['move_translation'] / vox:.3f}", f"{s_['move_rotation_deg']:.4f}")
            st.add_row(*(Text(str(v), justify="right") for v in vals))

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss(None)
