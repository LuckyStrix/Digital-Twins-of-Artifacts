"""Textual app: layout, bindings, themes, and the glue between the settings
form, the stage cards, the log and the pipeline runner."""

from __future__ import annotations

import json
import queue
from pathlib import Path
from typing import Callable

from textual import on, work
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import Button, Footer, Header, Input, Label

from pipeline import DEFAULTS_PATH
from pipeline import platform as plat
from pipeline import settings as S
from pipeline.options import STAGE_NAMES
from pipeline.paths import SessionPaths, default_output_for, describe_sides, detect_sides
from pipeline.runner import PipelineRunner

from .screens import ConfirmScreen, IcpDetailsScreen, PathPicker
from .uistate import UI_STATE_PATH, UiState
from .widgets import LOG_MIN_HEIGHT, LogHandle, LogPane, SettingsForm, StageCard

# ── Windows Terminal colour schemes ───────────────────────────────────────────
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
THEMES = ["campbell", "one-half-dark", "tokyo-night", "textual-dark", "textual-light"]

INPUT_DEBOUNCE = 0.4
EVENT_POLL = 0.05       # seconds between draining runner events into the UI

RunnerFactory = Callable[..., PipelineRunner]

BROWSE_TITLES = {
    "input_var": "Select input image directory",
    "output_var": "Select output directory",
    "intr_file_var": "Camera intrinsics JSON",
    "align_a_var": "Select PLY file (A, fixed)",
    "align_b_var": "Select PLY file (B, moving)",
}


class ReconApp(App):
    TITLE = "Tablet Reconstruction"
    SUB_TITLE = "Digital Twins of Artifacts"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("r", "run_all", "Run all"),
        Binding("s", "stop", "Stop"),
        Binding("o", "open_session", "Open folder"),
        Binding("ctrl+s", "save_defaults", "Save defaults"),
        Binding("l", "toggle_log", "Log ↕"),
        Binding("ctrl+up", "log_resize(2)", "Log +", show=False),
        Binding("ctrl+down", "log_resize(-2)", "Log −", show=False),
        Binding("i", "icp_details", "ICP details", show=False),
        Binding("t", "cycle_theme", "Theme"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, settings: S.Settings | None = None,
                 defaults_path: Path = DEFAULTS_PATH,
                 ui_state_path: Path = UI_STATE_PATH,
                 runner_factory: RunnerFactory = PipelineRunner):
        super().__init__()
        self.defaults_path = Path(defaults_path)
        self.ui_state_path = Path(ui_state_path)
        self.settings = settings if settings is not None else S.Settings.load(self.defaults_path)
        self.paths = SessionPaths(self.settings)
        self.ui = UiState.load(self.ui_state_path)
        self.register_theme(CAMPBELL)
        self.register_theme(ONE_HALF_DARK)
        self._debounce_timers: dict[str, object] = {}
        # Runner callbacks arrive on its worker thread (and stop() logs from
        # this one); queue them and apply them on a timer, in order.
        self._events: queue.SimpleQueue = queue.SimpleQueue()
        self.runner = runner_factory(
            self.settings,
            on_log=lambda text, kind: self._events.put(("log", text, kind)),
            on_stage_state=lambda idx, state, status: self._events.put(("state", idx, state, status)),
            on_progress=lambda idx, pct, text: self._events.put(("progress", idx, pct, text)),
        )
        self._run_active = False
        self._run_results: dict[int, str] = {}

    # ── layout ────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with VerticalScroll(id="side"):
                with Vertical(id="proj"):
                    yield Label("SESSION", id="proj-title")
                    yield Label("", id="proj-path")
                    yield Label("", id="detected")
                self.cards = [StageCard(i, n) for i, n in enumerate(STAGE_NAMES)]
                yield from self.cards
                with Horizontal(id="actions"):
                    yield Button("▶ Run all", variant="primary", id="run-all", compact=True)
                    yield Button("■ Stop", variant="error", id="stop", compact=True, disabled=True)
            with Vertical(id="main"):
                self.form = SettingsForm(self.settings, id="form")
                yield self.form
                yield LogHandle(id="log-handle")
                with Vertical(id="log-wrap"):
                    yield LogPane(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = self.ui.theme if self.ui.theme in self.available_themes else "campbell"
        self.theme_changed_signal.subscribe(self, self._on_theme_changed)
        self._apply_log_height(self.ui.log_height)
        self.log_pane = self.query_one(LogPane)
        self.set_interval(1, self._tick)
        self.set_interval(EVENT_POLL, self._drain_events)
        self._update_session_info()
        self.reconcile_stage_states()
        self.log_pane.add("Ready. Pick an input folder on the Inputs tab, then press r to "
                          "run all stages (or Run Stage on a card).", "info")

    def _tick(self) -> None:
        for c in self.cards:
            if c.state == "running":
                c.refresh_detail()

    # ── session info / stage reconcile ────────────────────────────────────────
    def _update_session_info(self, detected: str | None = None) -> None:
        self.query_one("#proj-path", Label).update(str(self.paths.session_dir()))
        s1, s2 = self.paths.active_sides()
        sides = ", ".join(x for x in (s1, s2) if x) or "flat image set"
        self.query_one("#detected", Label).update(
            f"sides: {sides}" + ("" if s2 else " · no alignment"))
        if detected is not None:
            self.form.set_status(detected)

    def reconcile_stage_states(self) -> None:
        """Mark stages done whose expected output already exists."""
        for idx, card in enumerate(self.cards):
            if card.state == "running":
                continue
            if self.paths.stage_output_ready(idx):
                card.set_state("done")

    # ── settings edits ────────────────────────────────────────────────────────
    def _debounce(self, name: str, callback) -> None:
        t = self._debounce_timers.pop(name, None)
        if t is not None:
            t.stop()
        self._debounce_timers[name] = self.set_timer(INPUT_DEBOUNCE, callback)

    @on(SettingsForm.SettingChanged)
    def _setting_changed(self, event: SettingsForm.SettingChanged) -> None:
        key = event.key
        if key == "input_var":
            self._debounce("input", self.apply_input_change)
        elif key == "output_var":
            self._debounce("output", self._output_changed)
        elif key in ("side1_var", "side2_var"):
            self._update_session_info()
        elif key == "r_hole_reduction":
            for k, v in S.hole_reduction_preset(bool(event.value)).items():
                self.form.set_value(k, v)

    def apply_input_change(self) -> None:
        """Input folder changed: default the output folder, detect sides."""
        inp = self.settings.text("input_var")
        if not inp or plat.looks_like_windows_path(inp):
            return            # a Windows path is converted when the field is left
        p = Path(inp)
        if not p.is_dir():
            self._update_session_info(f"Folder not found: {inp}")
            return
        if not self.settings.text("output_var"):
            self.form.set_value("output_var", default_output_for(inp))
        s1, s2, msg = describe_sides(detect_sides(p))
        self.form.set_value("side1_var", s1)
        self.form.set_value("side2_var", s2)
        self._update_session_info(msg)
        self.reconcile_stage_states()

    def _output_changed(self) -> None:
        self._update_session_info()
        self.reconcile_stage_states()

    @on(Input.Submitted)
    @on(Input.Blurred)
    def _path_field_left(self, event) -> None:
        """Convert a typed/pasted Windows path (D:\\scans, \\\\server\\share)."""
        wid = event.input.id or ""
        key = wid[4:] if wid.startswith("set-") else ""
        st = S.BY_KEY.get(key)
        if st is None or st.type != S.PATH:
            return
        value = event.input.value.strip()
        if plat.looks_like_windows_path(value):
            conv = plat.to_posix_path(value)
            if conv != value:
                self.form.set_value(key, conv, notify=True)

    # ── log pane ──────────────────────────────────────────────────────────────
    def _max_log_height(self) -> int:
        return max(LOG_MIN_HEIGHT, self.query_one("#main").size.height - 6)

    def _apply_log_height(self, h: int) -> int:
        h = max(LOG_MIN_HEIGHT, int(h))
        main_h = self.query_one("#main").size.height
        if main_h:
            h = min(h, self._max_log_height())
        self.query_one("#log-wrap").styles.height = h
        return h

    @on(LogHandle.Resized)
    def _log_dragged(self, event: LogHandle.Resized) -> None:
        self.screen.remove_class("-log-max")
        h = self._apply_log_height(event.height)
        if event.final:
            self.ui.log_height = h
            self.ui.save(self.ui_state_path)

    def action_log_resize(self, delta: int) -> None:
        self.screen.remove_class("-log-max")
        cur = self.query_one("#log-wrap").styles.height
        cur_v = int(cur.value) if cur is not None and cur.unit.name == "CELLS" else self.ui.log_height
        self.ui.log_height = self._apply_log_height(cur_v + delta)
        self.ui.save(self.ui_state_path)

    def action_toggle_log(self) -> None:
        self.screen.toggle_class("-log-max")
        if self.screen.has_class("-log-max"):
            self.query_one("#log-wrap").styles.height = "1fr"
        else:
            self._apply_log_height(self.ui.log_height)

    # ── themes ────────────────────────────────────────────────────────────────
    def action_cycle_theme(self) -> None:
        i = THEMES.index(self.theme) if self.theme in THEMES else -1
        self.theme = THEMES[(i + 1) % len(THEMES)]
        self.notify(f"Theme: {self.theme}", timeout=1.5)

    def _on_theme_changed(self, theme: Theme) -> None:
        self.ui.theme = theme.name
        self.ui.save(self.ui_state_path)

    # ── menu actions ──────────────────────────────────────────────────────────
    def action_save_defaults(self) -> None:
        try:
            self.settings.save(self.defaults_path)
        except OSError as e:
            self.notify(f"Could not save defaults:\n{e}", title="Save defaults",
                        severity="error", timeout=8)
            return
        self.notify("Current settings saved as default.\n"
                    "They'll be pre-filled next time the app opens.", title="Save defaults")

    def action_open_session(self) -> None:
        err = plat.open_folder(self.paths.session_dir())
        if err:
            self.notify(err, title="Open folder", severity="warning", timeout=10)

    def get_system_commands(self, screen):
        yield from super().get_system_commands(screen)
        yield SystemCommand("Open session folder", "Open the output folder in the file manager",
                            self.action_open_session)
        yield SystemCommand("Save current settings as default",
                            "Pre-fill these settings next time the app opens",
                            self.action_save_defaults)
        yield SystemCommand("Show ICP details", "Open the last alignment's icp_report.json",
                            self.action_icp_details)
        yield SystemCommand("Run all stages", "Run stages 1-4 in order", self.action_run_all)
        yield SystemCommand("Stop", "Stop the running stage", self.action_stop)

    # ── pickers ───────────────────────────────────────────────────────────────
    def browse(self, key: str) -> None:
        st = S.BY_KEY[key]

        def chosen(path: str | None) -> None:
            if not path:
                return
            self.ui.add_recent(path if st.path_kind == "dir" else str(Path(path).parent))
            self.ui.save(self.ui_state_path)
            self.form.set_value(key, path, notify=True)

        self.push_screen(PathPicker(
            st.path_kind, BROWSE_TITLES.get(key, st.label), initial=self.settings.text(key),
            pattern=st.file_glob, recents=self.ui.recent_dirs, auto_native=self.ui.native_dialog,
        ), chosen)

    @on(SettingsForm.BrowseRequested)
    def _browse_requested(self, event: SettingsForm.BrowseRequested) -> None:
        self.browse(event.key)

    @on(SettingsForm.ActionRequested)
    def _form_action(self, event: SettingsForm.ActionRequested) -> None:
        if event.action == "icp_details":
            self.action_icp_details()

    # ── view / ICP details ────────────────────────────────────────────────────
    @on(StageCard.ViewPressed)
    def _view_pressed(self, event: StageCard.ViewPressed) -> None:
        self.view_stage(event.idx)

    def view_stage(self, idx: int) -> None:
        path = self.paths.expected_output_for_stage(idx)
        if not path.exists():
            self.notify(f"Not found:\n{path}", title="File not found", severity="warning")
            return
        if idx == 0:
            # Photo gallery: the processed images in the desktop file manager.
            err = plat.open_folder(path)
            if err:
                self.notify(err, title="Processed photos", timeout=15)
            return
        if not plat.has_display():
            self.notify(f"No display available to open the 3D viewer. File:\n{path}",
                        title="Viewer", timeout=15)
            return
        self.spawn_viewer(path)

    @work(thread=True, group="viewer")
    def spawn_viewer(self, path: Path) -> None:
        try:
            proc = plat.launch_viewer(path)
        except (FileNotFoundError, OSError) as e:
            cmd, _ = plat.resolve_viewer_launch(path)
            self._events.put(("log", f"[viewer] interpreter not found: {cmd[0]!r}", "info"))
            self.call_from_thread(self.notify, f"Could not launch viewer: {e}",
                                  title="Viewer failed", severity="error", timeout=10)
            return
        failed, out = plat.wait_viewer(proc)
        if failed and out.strip():
            self._events.put(("log", f"[viewer] {out.strip()}", "info"))
        if failed:
            self.call_from_thread(self.notify,
                                  f"Viewer could not open a window:\n\n{out.strip()[-600:]}",
                                  title="Viewer failed", severity="error", timeout=15)

    def action_icp_details(self) -> None:
        rp = self.paths.icp_report_path()
        if rp.exists():
            self.show_icp_report(rp)
            return

        def chosen(path: str | None) -> None:
            if path:
                self.show_icp_report(Path(path))

        self.push_screen(PathPicker(
            "file", "No report for this session - pick an icp_report.json",
            initial=str(rp.parent if rp.parent.is_dir() else ""), pattern="*.json",
            recents=self.ui.recent_dirs, auto_native=self.ui.native_dialog), chosen)

    def show_icp_report(self, rp: Path) -> None:
        try:
            data = json.loads(rp.read_text())
        except (OSError, ValueError) as e:
            self.notify(f"Could not read {rp}:\n{e}", title="ICP details", severity="error")
            return
        self.push_screen(IcpDetailsScreen(data, rp))

    # ── running ───────────────────────────────────────────────────────────────
    def _started(self) -> None:
        self._run_active = True
        self._run_results = {}
        self.query_one("#run-all", Button).disabled = True
        self.query_one("#stop", Button).disabled = False

    def action_run_all(self) -> None:
        if not self.runner.run_all():
            self.notify("Pipeline is already running.", title="Busy", severity="warning")
            return
        self._started()

    def run_stage(self, idx: int) -> None:
        if not self.runner.run_stage(idx):
            self.notify("Another stage is already running.", title="Busy", severity="warning")
            return
        self._started()

    def action_stop(self) -> None:
        if self.runner.is_running:
            self.runner.stop()

    @on(StageCard.RunPressed)
    def _run_stage_pressed(self, event: StageCard.RunPressed) -> None:
        self.run_stage(event.idx)

    def _drain_events(self) -> None:
        batch = 0
        while batch < 2000:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            batch += 1
            kind = ev[0]
            if kind == "log":
                self.log_pane.add(ev[1], ev[2])
            elif kind == "state":
                _, idx, state, status = ev
                self.cards[idx].set_state(state, status)
                if state in ("done", "failed"):
                    self._run_results[idx] = state
            elif kind == "progress":
                _, idx, pct, text = ev
                self.cards[idx].set_progress(pct, text)
        if self._run_active and not self.runner.is_running and batch == 0:
            self._finished()

    def _finished(self) -> None:
        self._run_active = False
        self.query_one("#run-all", Button).disabled = False
        self.query_one("#stop", Button).disabled = True
        failed = [i for i, st in self._run_results.items() if st == "failed"]
        if failed:
            self.notify(f"Stage {failed[0] + 1} failed — see the log.", title="Pipeline",
                        severity="error", timeout=8)
        elif self._run_results:
            last = max(self._run_results)
            self.notify(f"{STAGE_NAMES[last]} finished.", title="Pipeline")

    async def action_quit(self) -> None:
        if not self.runner.is_running:
            self.exit()
            return

        def answer(yes: bool | None) -> None:
            if yes:
                self.runner.stop()
                self.exit()

        self.push_screen(ConfirmScreen(
            "Quit", "A stage is still running. Stop it and quit?", yes="Stop and quit"), answer)

    @on(Button.Pressed, "#run-all")
    def _run_all_btn(self) -> None:
        self.action_run_all()

    @on(Button.Pressed, "#stop")
    def _stop_btn(self) -> None:
        self.action_stop()


def main() -> None:
    ReconApp().run()


if __name__ == "__main__":
    main()
