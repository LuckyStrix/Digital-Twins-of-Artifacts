"""Widgets: stage cards, the resizable log pane, and the settings form that
is generated from pipeline.settings.REGISTRY."""

from __future__ import annotations

import time
from itertools import groupby
from typing import Any

from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.suggester import SuggestFromList
from textual.validation import Function, Integer, Number
from textual.widget import Widget
from textual.widgets import (
    Button, Checkbox, Collapsible, Input, Label, ProgressBar, RichLog, Select, Static,
    TabbedContent, TabPane, TextArea,
)

from pipeline import settings as S
from pipeline.options import VIEW_LABELS

# ── stage card ────────────────────────────────────────────────────────────────

ICONS = {"waiting": "○", "running": "◐", "done": "●", "failed": "✕"}
DEFAULT_STATUS = {"waiting": "Waiting", "running": "Running…", "done": "Done", "failed": "Failed"}


class StageCard(Vertical):
    """One pipeline stage: state, progress, Run Stage + View buttons."""

    class RunPressed(Message):
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx

    class ViewPressed(Message):
        def __init__(self, idx: int):
            super().__init__()
            self.idx = idx

    def __init__(self, idx: int, name: str):
        super().__init__(classes="stage waiting", id=f"stage-{idx}")
        self.idx, self.stage_name = idx, name
        self.state = "waiting"
        self.status = DEFAULT_STATUS["waiting"]
        self.started = 0.0
        self.border_title = f"{ICONS['waiting']} {name}"

    def compose(self) -> ComposeResult:
        yield ProgressBar(total=100, show_eta=False, id=f"bar-{self.idx}")
        yield Label(self.status, classes="detail")
        with Horizontal(classes="stage-btns"):
            yield Button("Run Stage", id=f"run-{self.idx}", compact=True, classes="run-stage")
            yield Button(VIEW_LABELS[self.idx], id=f"view-{self.idx}", compact=True,
                         disabled=True, classes="view-stage")

    def set_state(self, state: str, status: str = "") -> None:
        if state != self.state:
            self.remove_class(self.state)
            self.add_class(state)
            self.state = state
            self.border_title = f"{ICONS[state]} {self.stage_name}"
        self.status = status or DEFAULT_STATUS[state]
        bar = self.query_one(ProgressBar)
        if state == "running":
            self.started = time.monotonic()
            bar.update(total=None, progress=0)     # indeterminate until parsed progress
        elif state == "waiting":
            bar.update(total=100, progress=0)
        else:
            bar.update(total=100, progress=100)
        self.query_one(".run-stage", Button).disabled = state == "running"
        self.query_one(".view-stage", Button).disabled = state != "done"
        self.refresh_detail()

    def set_progress(self, pct: int, text: str = "") -> None:
        """Switch the bar to determinate and update value + label."""
        if self.state != "running":
            return
        self.query_one(ProgressBar).update(total=100, progress=pct)
        if text:
            self.status = text
        self.refresh_detail()

    def refresh_detail(self) -> None:
        t = Text(self.status)
        if self.state == "running" and self.started:
            s = int(time.monotonic() - self.started)
            t.append(f"  {s // 60:d}:{s % 60:02d}", style="dim")
        self.query_one(".detail", Label).update(t)

    @on(Button.Pressed, ".run-stage")
    def _run(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.RunPressed(self.idx))

    @on(Button.Pressed, ".view-stage")
    def _view(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ViewPressed(self.idx))


# ── slider (seg. quality) ─────────────────────────────────────────────────────

class Slider(Widget, can_focus=True):
    """Tiny horizontal slider: ←/→ (shift = ×10), home/end, click or drag."""

    DEFAULT_CSS = """
    Slider { height: 1; width: 32; }
    Slider:focus { text-style: bold; }
    """
    BINDINGS = [
        ("left", "step(-1)", "Less"), ("right", "step(1)", "More"),
        ("shift+left", "step(-10)", ""), ("shift+right", "step(10)", ""),
        ("home", "to_min", ""), ("end", "to_max", ""),
    ]

    value: reactive[int] = reactive(0)

    class Changed(Message):
        def __init__(self, slider: "Slider", value: int):
            super().__init__()
            self.slider = slider
            self.value = value

        @property
        def control(self) -> "Slider":
            return self.slider

    def __init__(self, value: int, minimum: int, maximum: int, suffix: str = "", **kw):
        super().__init__(**kw)
        self.minimum, self.maximum, self.suffix = minimum, maximum, suffix
        self.set_reactive(Slider.value, self._clamp(value))

    def _clamp(self, v: int) -> int:
        return max(self.minimum, min(self.maximum, int(v)))

    def _track_width(self) -> int:
        return max(4, self.size.width - 6)

    def render(self) -> Text:
        w = self._track_width()
        frac = (self.value - self.minimum) / max(1, self.maximum - self.minimum)
        knob = round(frac * (w - 1))
        t = Text()
        t.append("━" * knob, style=f"bold {self.app.current_theme.primary}")
        t.append("●", style="bold")
        t.append("─" * (w - 1 - knob), style="dim")
        t.append(f" {self.value:>3d}{self.suffix}")
        return t

    def watch_value(self, value: int) -> None:
        self.post_message(self.Changed(self, value))

    def action_step(self, n: int) -> None:
        self.value = self._clamp(self.value + n)

    def action_to_min(self) -> None:
        self.value = self.minimum

    def action_to_max(self) -> None:
        self.value = self.maximum

    def _set_from_x(self, x: int) -> None:
        w = self._track_width()
        x = max(0, min(w - 1, x))
        self.value = self._clamp(round(self.minimum + (self.maximum - self.minimum) * x / (w - 1)))

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self.capture_mouse()
        self._set_from_x(event.x)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.app.mouse_captured is self:
            self._set_from_x(event.x)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        self.release_mouse()


# ── log pane ──────────────────────────────────────────────────────────────────

LOG_MAX_LINES = 20000
LOG_MIN_HEIGHT = 3


class LogHandle(Static):
    """1-row bar above the log; drag it to resize the log."""

    class Resized(Message):
        def __init__(self, height: int, final: bool):
            super().__init__()
            self.height = height
            self.final = final

    def __init__(self, **kw):
        super().__init__("", **kw)
        self._dragging = False

    def on_mount(self) -> None:
        self.update(Text.assemble(
            ("─── Output log ", "bold"),
            ("  drag to resize · ctrl+↑/↓ · l maximise", "dim"),
        ))

    def _height_at(self, screen_y: int) -> int:
        log_wrap = self.app.query_one("#log-wrap")
        bottom = log_wrap.region.bottom
        return bottom - screen_y - 1

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._dragging = True
        self.capture_mouse()
        self.add_class("-dragging")

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if self._dragging:
            self.post_message(self.Resized(self._height_at(event.screen_y), final=False))

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()
            self.remove_class("-dragging")
            self.post_message(self.Resized(self._height_at(event.screen_y), final=True))


class LogPane(RichLog):
    """Pipeline output: timestamped app messages, styled command headers,
    raw subprocess output. Capped at LOG_MAX_LINES."""

    def __init__(self, **kw):
        super().__init__(max_lines=LOG_MAX_LINES, wrap=True, markup=False,
                         highlight=False, auto_scroll=True, **kw)

    def add(self, text: str, kind: str = "output") -> None:
        theme = self.app.current_theme
        if kind == "header":
            self.write(Text(""))
            self.write(Text("── " + text, style=f"bold {theme.primary}"))
        elif kind == "info":
            if text.startswith("[") and text[9:11] == "] ":
                t = Text(text[:10], style="dim")
                body = text[11:]
            else:
                t, body = Text(), text
            style = theme.error if ("[error]" in body or "failed" in body) else (theme.secondary or "")
            t.append(" " + body if t else body, style=style)
            self.write(t)
        else:
            self.write(Text(text))


# ── settings form ─────────────────────────────────────────────────────────────

SHORT_HELP = 120   # help text up to this length is shown inline, longer is collapsible
INLINE_HINT = 48   # hints longer than this go on their own line


def _num_ok(v: str, conv) -> bool:
    try:
        conv(v)
        return True
    except ValueError:
        return False


def _str_validator(kind: str):
    if kind == "int":
        return Function(lambda v: not v.strip() or _num_ok(v, int), "Not an integer")
    if kind == "float":
        return Function(lambda v: not v.strip() or _num_ok(v, float), "Not a number")
    if kind == "float_list":
        return Function(lambda v: all(_num_ok(p, float) for p in v.split(",") if p.strip()),
                        "Comma-separated numbers")
    return None


def fmt_value(st: S.Setting, value: Any) -> str:
    return "" if value is None else str(value)


class SettingsForm(Vertical):
    """Four tabs of fields, generated from the settings registry and bound to
    a ``Settings`` dict. Emits ``SettingChanged`` after a user edit."""

    class SettingChanged(Message):
        def __init__(self, key: str, value: Any):
            super().__init__()
            self.key = key
            self.value = value

    class BrowseRequested(Message):
        def __init__(self, key: str):
            super().__init__()
            self.key = key

    class ActionRequested(Message):
        def __init__(self, action: str):
            super().__init__()
            self.action = action

    def __init__(self, settings: S.Settings, **kw):
        super().__init__(**kw)
        self.settings = settings
        self.fields: dict[str, Widget] = {}

    # extra widgets placed after a given setting's row
    def _extras_after(self, key: str):
        if key == "output_var":
            yield Static("", id="struct", classes="status-line")
        elif key == "align_robust_var":
            with Horizontal(classes="row"):
                yield Label("", classes="lbl")
                yield Button("Show ICP details…", id="icp-details", compact=True)
            yield Static("Opens the last alignment's icp_report.json: per-stage "
                         "convergence and the residual histogram before/after.", classes="help")

    def compose(self) -> ComposeResult:
        with TabbedContent(id="tabs"):
            for tab in S.TABS:
                with TabPane(tab, id=f"tab-{tab.lower()}"):
                    with VerticalScroll():
                        items = [s for s in S.REGISTRY if s.tab == tab]
                        for section, group in groupby(items, key=lambda s: s.section):
                            yield Static(section, classes="section")
                            for st in group:
                                yield from self._field(st)
                                yield from self._extras_after(st.key)

    def _field(self, st: S.Setting):
        value = self.settings[st.key]
        wid = f"set-{st.key}"
        hint = st.hint if len(st.hint) <= INLINE_HINT else ""
        long_hint = st.hint if len(st.hint) > INLINE_HINT else ""

        if st.type == S.BOOL:
            w = Checkbox(st.label, bool(value), compact=True, id=wid, classes="chk")
            self.fields[st.key] = w
            yield w
        elif st.type == S.TEXT:
            yield Label(st.label, classes="lbl-above")
            w = TextArea(str(value), soft_wrap=True, id=wid, classes="textarea")
            self.fields[st.key] = w
            yield w
        else:
            if st.type == S.INT and st.widget == "slider":
                w = Slider(int(value), int(st.min), int(st.max), suffix="%", id=wid)
                hint = ""
            elif st.type in (S.INT, S.FLOAT):
                w = Input(fmt_value(st, value), compact=True, id=wid, classes="num",
                          type="integer" if st.type == S.INT else "number",
                          validators=[(Integer if st.type == S.INT else Number)(
                              minimum=st.min, maximum=st.max)],
                          validate_on=["changed", "blur", "submitted"],
                          select_on_focus=False)
            elif st.type == S.CHOICE and st.editable:
                w = Input(str(value), compact=True, id=wid, classes="pick",
                          suggester=SuggestFromList(list(st.choices), case_sensitive=False),
                          select_on_focus=False)
            elif st.type == S.CHOICE:
                w = Select([(c, c) for c in st.choices], value=value, allow_blank=False,
                           compact=True, id=wid, classes="pick")
            else:  # STR / PATH
                v = _str_validator(st.validate)
                w = Input(str(value), compact=True, id=wid,
                          classes="path" if st.type == S.PATH else "text",
                          validators=[v] if v else None,
                          validate_on=["changed", "blur", "submitted"] if v else None,
                          select_on_focus=False)
            self.fields[st.key] = w
            kids: list[Widget] = [Label(st.label, classes="lbl"), w]
            if st.type == S.PATH:
                kids.append(Button("…", id=f"browse-{st.key}", compact=True, classes="browse"))
            if hint:
                kids.append(Label(hint, classes="hint"))
            yield Horizontal(*kids, classes="row")

        if long_hint:
            yield Static(long_hint, classes="help")
        if st.help:
            if len(st.help) <= SHORT_HELP:
                yield Static(st.help, classes="help")
            else:
                yield Collapsible(Static(st.help, classes="help-body"),
                                  title=f"About: {st.label}", classes="about")

    # ── programmatic access ───────────────────────────────────────────────────
    def set_value(self, key: str, value: Any, notify: bool = False) -> None:
        """Store `value` and show it in the field (no SettingChanged unless
        `notify`)."""
        st = S.BY_KEY[key]
        self.settings[key] = value
        w = self.fields.get(key)
        if isinstance(w, Checkbox):
            w.value = bool(value)
        elif isinstance(w, Select):
            w.value = value
        elif isinstance(w, Slider):
            w.value = int(value)
        elif isinstance(w, TextArea):
            if w.text != value:
                w.text = str(value)
        elif isinstance(w, Input):
            if w.value != fmt_value(st, value):
                w.value = fmt_value(st, value)
        if notify:
            self.post_message(self.SettingChanged(key, value))

    def refresh_all(self) -> None:
        for key in self.fields:
            self.set_value(key, self.settings[key])

    def set_status(self, text: str) -> None:
        self.query_one("#struct", Static).update(text)

    # ── user edits ────────────────────────────────────────────────────────────
    def _store(self, key: str, raw: Any) -> None:
        st = S.BY_KEY.get(key)
        if st is None:
            return
        try:
            value = S.coerce(st, raw)
        except (ValueError, TypeError):
            return            # keep the last valid value; the field shows as invalid
        if st.type in (S.INT, S.FLOAT) and (
                (st.min is not None and value < st.min) or (st.max is not None and value > st.max)):
            return
        if self.settings.get(key) == value and type(self.settings.get(key)) is type(value):
            return            # programmatic update echoing back
        self.settings[key] = value
        self.post_message(self.SettingChanged(key, value))

    @staticmethod
    def _key(widget: Widget | None) -> str | None:
        wid = getattr(widget, "id", None) or ""
        return wid[4:] if wid.startswith("set-") else None

    @on(Input.Changed)
    def _input_changed(self, event: Input.Changed) -> None:
        key = self._key(event.input)
        if key:
            event.stop()
            self._store(key, event.value)

    @on(Checkbox.Changed)
    def _checkbox_changed(self, event: Checkbox.Changed) -> None:
        key = self._key(event.checkbox)
        if key:
            event.stop()
            self._store(key, event.value)

    @on(Select.Changed)
    def _select_changed(self, event: Select.Changed) -> None:
        key = self._key(event.select)
        if key and event.value is not Select.NULL:
            event.stop()
            self._store(key, event.value)

    @on(TextArea.Changed)
    def _textarea_changed(self, event: TextArea.Changed) -> None:
        key = self._key(event.text_area)
        if key:
            event.stop()
            self._store(key, event.text_area.text)

    @on(Slider.Changed)
    def _slider_changed(self, event: Slider.Changed) -> None:
        key = self._key(event.slider)
        if key:
            event.stop()
            self._store(key, event.value)

    @on(Button.Pressed, ".browse")
    def _browse(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.BrowseRequested(event.button.id[len("browse-"):]))

    @on(Button.Pressed, "#icp-details")
    def _icp(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.ActionRequested("icp_details"))
