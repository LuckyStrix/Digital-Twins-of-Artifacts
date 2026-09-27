"""
Textual mockup of the Tablet Reconstruction GUI.

Pure mockup: nothing here touches the pipeline. "Run" plays back a scripted,
fake pipeline so you can see how progress, logs and charts would feel.

Keys:  r run   s stop   v viewer   ctrl+s save defaults   t theme   ctrl+p palette   q quit
"""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button, Checkbox, DataTable, DirectoryTree, Footer, Header, Input, Label,
    ProgressBar, RichLog, Select, Sparkline, Static, TabbedContent, TabPane,
)

# ── Windows Terminal colour schemes ──────────────────────────────────────────
CAMPBELL = Theme(
    name="campbell",
    primary="#3B78FF", secondary="#61D6D6", accent="#B4009E",
    foreground="#CCCCCC", background="#0C0C0C", surface="#141414",
    panel="#1F1F1F", success="#16C60C", warning="#F9F1A5", error="#E74856",
    dark=True,
)
ONE_HALF_DARK = Theme(
    name="one-half-dark",
    primary="#61AFEF", secondary="#56B6C2", accent="#C678DD",
    foreground="#DCDFE4", background="#282C34", surface="#2F333D",
    panel="#3A3F4B", success="#98C379", warning="#E5C07B", error="#E06C75",
    dark=True,
)
THEMES = ["campbell", "one-half-dark", "tokyo-night", "textual-dark"]

C = {"hdr": "#3B78FF", "ok": "#16C60C", "dim": "#767676", "warn": "#F9F1A5",
     "cyan": "#61D6D6", "err": "#E74856"}


# ── Small form helpers ───────────────────────────────────────────────────────
def field(label: str, widget, hint: str = "") -> Horizontal:
    kids = [Label(label, classes="lbl"), widget]
    if hint:
        kids.append(Label(hint, classes="hint"))
    return Horizontal(*kids, classes="row")


def num(value, width: int = 12) -> Input:
    w = Input(str(value), compact=True, classes="num")
    w.styles.width = width
    return w


def pick(options: list[str], value: str) -> Select:
    return Select([(o, o) for o in options], value=value,
                  allow_blank=False, compact=True, classes="pick")


def check(label: str, value: bool) -> Checkbox:
    return Checkbox(label, value, compact=True, classes="chk")


def section(title: str) -> Static:
    return Static(title, classes="section")


# ── Stage card ───────────────────────────────────────────────────────────────
ICONS = {"waiting": "○", "running": "◐", "done": "●", "failed": "✕", "skipped": "–"}


class StageCard(Vertical):
    def __init__(self, idx: int, name: str) -> None:
        super().__init__(classes="stage waiting")
        self.idx, self.name_ = idx, name
        self.state, self.started, self.msg = "waiting", 0.0, "waiting"
        self.border_title = f"{ICONS['waiting']} {idx}  {name}"

    def compose(self) -> ComposeResult:
        yield ProgressBar(total=100, show_eta=False)
        yield Label("waiting", classes="detail")

    def set(self, state: str, pct: float | None = None, msg: str | None = None):
        if state != self.state:
            self.remove_class(self.state)
            self.add_class(state)
            if state == "running":
                self.started = time.monotonic()
            self.state = state
            self.border_title = f"{ICONS[state]} {self.idx}  {self.name_}"
        if pct is not None:
            self.query_one(ProgressBar).update(progress=pct)
        if msg is not None:
            self.msg = msg
        self.refresh_detail()

    def refresh_detail(self):
        el = ""
        if self.state in ("running", "done") and self.started:
            s = int(time.monotonic() - self.started)
            el = f"  [{C['dim']}]{s // 60:d}:{s % 60:02d}[/]"
        self.query_one(".detail", Label).update(f"{self.msg}{el}")


# ── Folder picker (stand-in for the native file dialog) ──────────────────────
class FolderPicker(ModalScreen[str | None]):
    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Static("Select input folder", classes="section")
            yield DirectoryTree(str(Path.home()), id="tree")
            with Horizontal(id="picker-btns"):
                yield Button("Select", variant="primary", id="ok", compact=True)
                yield Button("Cancel", id="cancel", compact=True)

    def on_mount(self):
        self.chosen = str(Path.home())

    @on(DirectoryTree.DirectorySelected)
    def _sel(self, e: DirectoryTree.DirectorySelected):
        self.chosen = str(e.path)

    @on(Button.Pressed, "#ok")
    def _ok(self):
        self.dismiss(self.chosen)

    @on(Button.Pressed, "#cancel")
    def _cancel(self):
        self.dismiss(None)


# ── Scripted fake run ────────────────────────────────────────────────────────
def script():
    """Yields (stage_idx, pct, status_msg, log_line_or_None)."""
    n1, n2 = 84, 79
    for i in range(1, n1 + n2 + 1, 7):
        side = "side1" if i <= n1 else "side2"
        yield 1, 100 * i / (n1 + n2), f"{i}/{n1 + n2} images", \
            f"[{C['dim']}]{side}/[/]IMG_{i:04d}.JPG  mask ok  erode 20px"
    yield 1, 100, "163 masks", f"[{C['ok']}]✓[/] 163 masks written (birefnet-general, cuda:0)"

    for side in ("side1", "side2"):
        base = 0 if side == "side1" else 50
        yield 2, base + 2, f"{side}: extracting", f"[{C['cyan']}]feature_extractor[/] {side}  sift_max_features=16000  dsp=on"
        yield 2, base + 10, f"{side}: matching", f"[{C['cyan']}]exhaustive_matcher[/] guided=on  max_matches=65536"
        for k in range(10, 85, 15):
            yield 2, base + 10 + k * 0.2, f"{side}: mapping", f"mapper: registered {k} / 84 images"
        yield 2, base + 30, f"{side}: stereo", f"[{C['cyan']}]patch_match_stereo[/] gpu 0 …"
        for k in range(1, 6):
            yield 2, base + 30 + k * 3, f"{side}: stereo", f"  depth maps {k * 17}/84"
        pts = random.randint(1_900_000, 2_600_000)
        yield 2, base + 50, f"{side}: fused", f"[{C['ok']}]✓[/] stereo_fusion: fused {pts:,} points"

    yield 3, 10, "voxel 0.42 mm", f"[{C['cyan']}]fpfh[/] voxel=auto → 0.42 mm, 60,000 samples"
    yield 3, 30, "RANSAC", "RANSAC fitness 0.612  inlier rmse 0.371"
    for stage, (lo, hi) in enumerate([(0.61, 0.78), (0.78, 0.88), (0.88, 0.93)], 1):
        for k in range(8):
            fit = lo + (hi - lo) * (1 - 0.6 ** (k + 1))
            yield 3, 30 + stage * 15 + k, f"ICP stage {stage}/3", ("icp", fit)
        yield 3, 30 + stage * 15 + 8, f"ICP stage {stage}/3", f"  icp stage {stage}: fitness {hi:.3f}"
    yield 3, 90, "seam resolution", "seam: mode=point τ=0.3 floor=0.35 → dropped 3.1% of points"
    yield 3, 100, "merged", f"[{C['ok']}]✓[/] output/aligned_cloud/merged_fpfh.ply"

    steps = [
        ("outliers", "statistical outliers: nn=32 → removed 41,220"),
        ("DBSCAN", "DBSCAN eps=2.2 → kept 1 cluster (98.1%)"),
        ("normals", "normals: max_nn=80, orient_k=64"),
        ("Poisson d=11", "Poisson depth 11 (linear fit) → 1,842,003 tris"),
        ("density trim", "density trim 0.003 → removed 2.4% of vertices"),
        ("hole filling", "fill holes: ratio 0.3, 5 passes → 37 patched"),
        ("cleanup", "cleanup: kept 1 component, smooth ×10"),
        ("simplify", "simplified export: 60,000 verts → model_web.glb"),
        ("pose", "pose normalization: PCA, thin axis → Z"),
    ]
    for k, (st, line) in enumerate(steps, 1):
        yield 4, 100 * k / len(steps), st, line
    yield 4, 100, "model.gltf", f"[{C['ok']}]✓[/] output/model.gltf"


# ── App ──────────────────────────────────────────────────────────────────────
class ReconApp(App):
    TITLE = "Tablet Reconstruction"
    SUB_TITLE = "Digital Twins of Artifacts · mockup"
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }

    /* sidebar */
    #side { width: 40; padding: 0 1; border-right: vkey $panel-lighten-1; }
    #proj { height: auto; padding: 0 0 1 0; }
    #proj-title { color: $secondary; text-style: bold; }
    #detected { color: $text-muted; }
    .stage { height: auto; border: round $panel-lighten-2; padding: 0 1;
             border-title-color: $text-muted; margin-bottom: 0; }
    .stage.running { border: round $primary; border-title-color: $primary;
                     border-title-style: bold; }
    .stage.done    { border: round $success; border-title-color: $success; }
    .stage.failed  { border: round $error;   border-title-color: $error; }
    .stage ProgressBar { width: 100%; }
    .stage Bar { width: 1fr; }
    .stage.done Bar > .bar--complete, .stage.done Bar > .bar--bar { color: $success; }
    .detail { color: $text-muted; }
    #actions { height: auto; margin-top: 1; }
    #actions Button { width: 1fr; min-width: 0; margin-right: 1; }
    #actions Button:last-of-type { margin-right: 0; }
    #overall { margin-top: 1; height: auto; }
    #overall-lbl { color: $text-muted; }

    /* main */
    #main { width: 1fr; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 0 1; }
    .section { color: $secondary; text-style: bold; margin-top: 1;
               border-bottom: solid $panel-lighten-1; width: 100%; }
    .row { height: 1; margin: 0; }
    .lbl { width: 20; color: $text-muted; }
    .hint { color: $text-disabled; margin-left: 2; }
    .pick { width: 26; }
    .path { width: 1fr; }
    .chk { margin-left: 20; height: 1; }
    .cols { height: auto; }
    .col { width: 1fr; height: auto; }

    #chart-box { height: auto; border: round $panel-lighten-2; padding: 0 1;
                 margin-top: 1; border-title-color: $secondary; }
    #icp { height: 5; }
    #icp-caption { color: $text-muted; }

    /* log */
    #log-wrap { height: 14; border-top: hkey $panel-lighten-1; }
    #log { background: $surface; padding: 0 1; scrollbar-size-vertical: 1; }

    /* folder picker */
    FolderPicker { align: center middle; }
    #picker { width: 70%; height: 70%; border: round $primary; background: $surface; padding: 0 1; }
    #tree { height: 1fr; }
    #picker-btns { height: auto; align-horizontal: right; }
    #picker-btns Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("r", "run", "Run all"),
        Binding("s", "stop", "Stop"),
        Binding("v", "viewer", "Viewer"),
        Binding("ctrl+s", "save", "Save defaults"),
        Binding("t", "cycle_theme", "Theme"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.register_theme(CAMPBELL)
        self.register_theme(ONE_HALF_DARK)
        self.icp_data: list[float] = []

    # ── layout ──
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="side"):
                with Vertical(id="proj"):
                    yield Label("PROJECT", id="proj-title")
                    yield Label("tablet_W1  ·  clay tablet", id="proj-name")
                    yield Label("2 sides (side1, side2) · 163 images",
                                id="detected")
                self.cards = [StageCard(i, n) for i, n in enumerate(
                    ["Background Removal", "COLMAP MVS",
                     "FPFH Alignment", "Mesh Reconstruction"], 1)]
                yield from self.cards
                with Vertical(id="overall"):
                    yield Label("overall", id="overall-lbl")
                    yield ProgressBar(total=400, show_eta=True, id="overall-bar")
                with Horizontal(id="actions"):
                    yield Button("▶ Run all", variant="primary", id="run", compact=True)
                    yield Button("■ Stop", variant="error", id="stop", compact=True, disabled=True)
                    yield Button("◈ Viewer", id="viewer", compact=True)
            with Vertical(id="main"):
                with TabbedContent(initial="inputs"):
                    with TabPane("Inputs", id="inputs"):
                        yield from self.tab_inputs()
                    with TabPane("COLMAP", id="colmap"):
                        yield from self.tab_colmap()
                    with TabPane("Reconstruct", id="recon"):
                        yield from self.tab_recon()
                    with TabPane("Alignment", id="align"):
                        yield from self.tab_align()
                    with TabPane("Outputs", id="outputs"):
                        yield DataTable(id="out-table", cursor_type="row", zebra_stripes=True)
                with Vertical(id="log-wrap"):
                    yield RichLog(id="log", markup=True, wrap=True, highlight=False)
        yield Footer()

    def tab_inputs(self):
        with VerticalScroll():
            yield section("Project")
            with Horizontal(classes="row"):
                yield Label("Input folder", classes="lbl")
                yield Input(r"D:\scans\tablet_W1", compact=True, classes="path", id="in-dir")
                yield Button("browse…", compact=True, id="browse")
            yield field("Output folder", Input(r"D:\scans\tablet_W1\output", compact=True, classes="path"))
            yield field("Artifact type", pick(["clay tablet", "cylinder seal", "sherd", "other"], "clay tablet"))
            yield field("Description", Input("Old Babylonian, obverse + reverse", compact=True, classes="path"))
            yield field("Primary side", Input("side1", compact=True, classes="pick"))
            yield field("Secondary side", Input("side2", compact=True, classes="pick"), "blank = skip alignment")
            yield section("Background removal")
            with Horizontal(classes="cols"):
                with Vertical(classes="col"):
                    yield field("Background", pick(["black", "white", "green", "auto"], "black"))
                    yield field("Model", pick(["birefnet-general", "birefnet-hr", "u2net"], "birefnet-general"))
                    yield field("Quality", pick(["low", "medium", "high"], "medium"))
                    yield field("Passes", num(2))
                    yield field("Seg. quality %", num(100))
                with Vertical(classes="col"):
                    yield field("Erode mask (px)", num(20))
                    yield field("Edge band (px)", num(0))
                    yield field("Black threshold", num(0))
                    yield field("White threshold", num(0))
                    yield field("Grow by colour", num(0))
            yield check("Hard mask", True)
            yield check("Fill hull when growing", True)
            yield check("Use GPU", True)

    def tab_colmap(self):
        with VerticalScroll():
            yield section("COLMAP")
            yield field("Image scale", num(1))
            yield field("Image stride", num(1))
            yield section("SIFT")
            yield field("Max features", num(16000))
            yield field("Peak threshold", num("0.0045"))
            yield field("Edge threshold", num(12))
            yield check("Domain-size pooling", True)
            yield check("Affine shape", False)
            yield section("Matching")
            yield check("Guided matching", True)
            yield field("Max matches", num(65536))
            yield section("Threads / cache")
            with Horizontal(classes="cols"):
                with Vertical(classes="col"):
                    yield field("Extract threads", num(0), "0 = auto")
                    yield field("Match threads", num(0))
                    yield field("Mapper threads", num(0))
                with Vertical(classes="col"):
                    yield field("Fusion threads", num(0))
                    yield field("Patch cache (GB)", num(0))
                    yield field("Fusion cache (GB)", num(0))
            yield section("Secondary camera")
            yield field("Align mode", pick(["auto", "manual"], "auto"))
            yield field("Rotate (deg)", num(180))
            yield field("Rotate axis", pick(["primary_frame_x", "primary_frame_y", "primary_frame_z"], "primary_frame_x"))

    def tab_recon(self):
        with VerticalScroll():
            with Horizontal(classes="cols"):
                with Vertical(classes="col"):
                    yield section("Point cloud filtering")
                    yield field("Outlier nn", num(32))
                    yield field("Outlier std", num("0.0"))
                    yield field("Radius factor", num(2.2))
                    yield section("DBSCAN clustering")
                    yield field("eps", num(2.2))
                    yield field("Keep clusters", num(1))
                    yield field("Min ratio", num(0.02))
                    yield section("Normals")
                    yield field("Max nn", num(80))
                    yield field("Orient k", num(64))
                with Vertical(classes="col"):
                    yield section("Poisson reconstruction")
                    yield field("Depth", num(11))
                    yield field("Density trim", num(0.003))
                    yield field("Crop scale", num(1.08))
                    yield check("Linear fit", True)
                    yield section("Holes")
                    yield check("Hole reduction", True)
                    yield check("Fill holes", True)
                    yield field("Fill ratio", num(0.3))
                    yield field("Fill passes", num(5))
                    yield check("Smooth (faired) patches", False)
            yield section("Cleanup & export")
            with Horizontal(classes="cols"):
                with Vertical(classes="col"):
                    yield field("Min comp. ratio", num(0.05))
                    yield field("Smooth iters", num(10))
                with Vertical(classes="col"):
                    yield field("Decimate tris", num(300000))
                    yield field("Web target verts", num(60000))
            yield check("Pose normalization (thin axis → Z)", True)

    def tab_align(self):
        with VerticalScroll():
            yield section("FPFH alignment")
            yield field("Method", pick(["opening", "fpfh", "collapse", "all"], "fpfh"))
            yield field("Voxel", num(0), "0 = auto")
            yield field("Sample points", num(60000))
            yield section("ICP refinement (coarse-to-fine)")
            yield check("Refine with multi-stage ICP", False)
            yield section("Seam resolution")
            yield check("Drop low-confidence points at the seam", True)
            yield field("Mode", pick(["point", "patch"], "point"))
            yield field("Conflict gap τ", num(0.3))
            yield field("Confidence floor", num(0.35))
            yield field("Window", num(10))
            with Vertical(id="chart-box") as box:
                box.border_title = "ICP convergence"
                yield Sparkline([], id="icp", summary_function=max)
                yield Label("run the pipeline to see fitness per iteration", id="icp-caption")

    # ── lifecycle ──
    def on_mount(self):
        self.theme = "campbell"
        t = self.query_one("#out-table", DataTable)
        for col, w in (("File", 34), ("Size", 9), ("Stage", 6), ("Status", 12)):
            t.add_column(col, key=col, width=w)
        for f, sz, st in [("masks/", "—", "1"), ("side1/dense/fused.ply", "—", "2"),
                          ("side2/dense/fused.ply", "—", "2"),
                          ("aligned_cloud/merged_fpfh.ply", "—", "3"),
                          ("recon/mesh_clean.ply", "—", "4"), ("model.gltf", "—", "4"),
                          ("model_web.glb", "—", "4"), ("info.txt", "—", "4")]:
            t.add_row(f, sz, st, "[dim]pending[/]", key=f)
        log = self.query_one(RichLog)
        log.write(f"[b {C['hdr']}]tablet-recon[/] ready  ·  press [b]r[/] to run, "
                  f"[b]ctrl+p[/] for the command palette")
        self.set_interval(1, self._tick)

    def _tick(self):
        for c in self.cards:
            if c.state == "running":
                c.refresh_detail()

    # ── actions ──
    def action_run(self):
        self.run_pipeline()

    def action_stop(self):
        self.workers.cancel_group(self, "pipeline")

    def action_viewer(self):
        self.notify("Would launch viewer.py on output/model.gltf", title="Viewer")

    def action_save(self):
        self.notify("Current settings saved to app_defaults.json (mock)", title="Defaults")

    def action_cycle_theme(self):
        i = THEMES.index(self.theme) if self.theme in THEMES else -1
        self.theme = THEMES[(i + 1) % len(THEMES)]
        self.notify(f"Theme: {self.theme}", timeout=1.5)

    @on(Button.Pressed, "#run")
    def _run_btn(self): self.action_run()

    @on(Button.Pressed, "#stop")
    def _stop_btn(self): self.action_stop()

    @on(Button.Pressed, "#viewer")
    def _viewer_btn(self): self.action_viewer()

    @on(Button.Pressed, "#browse")
    def _browse(self):
        def done(path: str | None):
            if path:
                self.query_one("#in-dir", Input).value = path
        self.push_screen(FolderPicker(), done)

    # ── fake pipeline ──
    @work(exclusive=True, group="pipeline")
    async def run_pipeline(self):
        log = self.query_one(RichLog)
        overall = self.query_one("#overall-bar", ProgressBar)
        spark = self.query_one("#icp", Sparkline)
        table = self.query_one("#out-table", DataTable)
        run_btn, stop_btn = self.query_one("#run", Button), self.query_one("#stop", Button)
        names = ["background removal", "COLMAP MVS", "FPFH alignment", "mesh reconstruction"]
        outputs = {1: ["masks/"], 2: ["side1/dense/fused.ply", "side2/dense/fused.ply"],
                   3: ["aligned_cloud/merged_fpfh.ply"],
                   4: ["recon/mesh_clean.ply", "model.gltf", "model_web.glb", "info.txt"]}
        sizes = {"masks/": "412 MB", "side1/dense/fused.ply": "61 MB",
                 "side2/dense/fused.ply": "58 MB", "aligned_cloud/merged_fpfh.ply": "97 MB",
                 "recon/mesh_clean.ply": "88 MB", "model.gltf": "54 MB",
                 "model_web.glb": "2.1 MB", "info.txt": "1 KB"}

        for c in self.cards:
            c.set("waiting", 0, "waiting")
        overall.update(progress=0)
        self.icp_data = []
        spark.data = []
        for key in sizes:
            table.update_cell(key, "Status", "[dim]pending[/]")
            table.update_cell(key, "Size", "—")
        run_btn.disabled, stop_btn.disabled = True, False
        log.write(f"\n[{C['dim']}]{time.strftime('%H:%M:%S')}[/]  run started")

        current = 0
        try:
            for stage, pct, msg, line in script():
                if stage != current:
                    if current:
                        self.cards[current - 1].set("done", 100, "done")
                        for key in outputs[current]:
                            table.update_cell(key, "Status", f"[{C['ok']}]✓ written[/]")
                            table.update_cell(key, "Size", sizes[key])
                    current = stage
                    log.write(f"[b {C['hdr']}]=== stage {stage}: {names[stage - 1]} ===[/]")
                card = self.cards[stage - 1]
                card.set("running", pct, msg)
                overall.update(progress=(stage - 1) * 100 + pct)
                if isinstance(line, tuple):
                    self.icp_data.append(line[1])
                    spark.data = list(self.icp_data)
                    self.query_one("#icp-caption", Label).update(
                        f"fitness {line[1]:.3f}  ·  {len(self.icp_data)} iterations")
                elif line:
                    log.write(line)
                await asyncio.sleep(random.uniform(0.05, 0.25))
            self.cards[current - 1].set("done", 100, "done")
            for key in outputs[current]:
                table.update_cell(key, "Status", f"[{C['ok']}]✓ written[/]")
                table.update_cell(key, "Size", sizes[key])
            overall.update(progress=400)
            log.write(f"[b {C['ok']}]pipeline complete[/] → output/model.gltf")
            self.notify("Pipeline complete: output/model.gltf", title="Done",
                        severity="information")
        except asyncio.CancelledError:
            if current:
                self.cards[current - 1].set("failed", None, "stopped by user")
            log.write(f"[b {C['err']}]stopped[/] during stage {current}")
            self.notify("Run stopped", severity="warning")
            raise
        finally:
            run_btn.disabled, stop_btn.disabled = False, True


if __name__ == "__main__":
    ReconApp().run()
