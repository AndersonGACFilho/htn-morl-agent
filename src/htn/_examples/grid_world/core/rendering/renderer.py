"""Composition of frame panels into the rendered console output."""

from __future__ import annotations

from rich.console import Console, Group
from rich.layout import Layout
from rich.text import Text

from htn._examples.grid_world.core.rendering.console import build_console
from htn._examples.grid_world.core.rendering.panels import (
    DecompositionPanel,
    GridPanel,
    ObjectivePanel,
    PlanPanel,
)
from htn._examples.grid_world.core.theme import GridWorldTheme
from htn._examples.grid_world.core.view.models import FrameView, MessageRole

LEFT_COLUMN_WIDTH = 52


class GridWorldRenderer:
    """Render a frame as a two-column console layout.

    The console has a fixed size so every exported frame shares the same
    dimensions, which is what keeps the assembled GIF stable.
    """

    def __init__(
        self,
        theme: GridWorldTheme,
        *,
        show_objectives: bool = False,
        console: Console | None = None,
    ) -> None:
        """Initialize the renderer.

        Args:
            theme: Palette, glyphs, and styles used by every panel.
            show_objectives: Whether to include the multi-objective panel.
            console: Recording console; one is created when omitted.
        """
        self.theme = theme
        self.console = console or build_console()
        self._grid_panel = GridPanel(theme)
        self._plan_panel = PlanPanel(theme)
        self._decomposition_panel = DecompositionPanel(theme)
        self._objective_panel = ObjectivePanel(theme) if show_objectives else None

    def render(self, frame: FrameView) -> None:
        """Draw one frame, replacing the previous console contents."""
        self.console.clear()

        layout = Layout()
        layout.split_row(
            Layout(name="left", size=LEFT_COLUMN_WIDTH),
            Layout(name="right"),
        )
        layout["left"].update(self._grid_panel.render(frame))

        right_sections = [
            Layout(self._plan_panel.render(frame), name="plan", size=13),
            Layout(self._decomposition_panel.render(frame), name="methods"),
        ]

        if self._objective_panel is not None:
            right_sections.append(
                Layout(self._objective_panel.render(frame), name="objectives", size=9)
            )

        layout["right"].split_column(*right_sections)

        self.console.print(layout)

    def print_message(
        self, message: str, *, role: MessageRole = MessageRole.NEUTRAL
    ) -> None:
        """Print a standalone message outside a frame."""
        self.console.print(Group(Text(message, style=self.theme.message_style(role))))
