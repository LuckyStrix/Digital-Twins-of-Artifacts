"""Modal screens: confirm dialog, folder/file pickers, ICP details."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape", "dismiss(False)", "Cancel"),
                Binding("y", "dismiss(True)", "Yes", show=False),
                Binding("n", "dismiss(False)", "No", show=False)]

    def __init__(self, title: str, message: str, yes: str = "Yes", no: str = "Cancel"):
        super().__init__()
        self._title, self._message, self._yes, self._no = title, message, yes, no

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm", classes="dialog"):
            yield Static(self._title, classes="dialog-title")
            yield Static(self._message, classes="dialog-body")
            with Horizontal(classes="dialog-btns"):
                yield Button(self._yes, variant="error", id="yes", compact=True)
                yield Button(self._no, id="no", compact=True)

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    @on(Button.Pressed, "#yes")
    def _yes_pressed(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#no")
    def _no_pressed(self) -> None:
        self.dismiss(False)
