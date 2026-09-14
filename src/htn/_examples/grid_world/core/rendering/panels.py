"""Rich panels composing one rendered frame.

Each panel reads only the frame view models, so none of them knows about the
environment, the agent, or the reward machinery.
"""

from __future__ import annotations

from typing import Protocol

from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from htn._examples.grid_world.core.theme import GridWorldTheme
from htn._examples.grid_world.core.view.models import (
    CellRole,
    DecompositionView,
    FrameView,
    GridView,
    LegendEntryView,
    LegendKey,
    MessageRole,
    ObjectiveView,
    PlanEntryStatus,
    PlanEntryView,
    PlanView,
)

PLACEHOLDER = "—"
MAX_PLAN_ROWS = 8
MAX_TREE_ROWS = 14


class FramePanel(Protocol):
    """A renderable section of a frame."""

    def render(self, frame: FrameView) -> RenderableType:
        """Return the Rich renderable for this section of the frame."""
        ...


class GridPanel:
    """Draw the grid with its route, target, and overlay markers."""

    def __init__(self, theme: GridWorldTheme) -> None:
        """Initialize the panel with a theme."""
        self._theme = theme

    def render(self, frame: FrameView) -> RenderableType:
        """Return the grid panel, centred on both axes of the world canvas.

        The grid and the legend share one left edge: the block is centred as a
        whole, and the legend sits under the world flush with its first column.
        """
        block = Table.grid()
        block.add_column(justify="left")
        block.add_row(self._grid(frame.grid))
        block.add_row("")
        block.add_row(self._legend(frame.grid))

        return Panel(
            Align.center(block, vertical="middle"),
            title=Text("GridWorld", style=f"bold {self._theme.palette.htn_blue}"),
            border_style=self._theme.border_style(),
            padding=(1, 2),
        )

    def _grid(self, grid: GridView) -> Table:
        """Return the grid itself as a borderless table."""
        table = Table.grid(expand=False)

        for _ in range(grid.width):
            table.add_column(justify="center", width=3)

        for row in grid.rows:
            table.add_row(
                *(
                    Text(
                        self._theme.glyph(cell.role), style=self._theme.cell_style(cell)
                    )
                    for cell in row
                )
            )

        return table

    def _legend(self, grid: GridView) -> Table:
        """Return the legend of the scenario, one row per declared element.

        Rows never appear or disappear during an episode: an element the agent
        occludes keeps its row, a door updates its own row when it opens, and a
        collected item is struck through rather than dropped.
        """
        legend = Table.grid(padding=(0, 1))
        legend.add_column()
        legend.add_column()

        for entry in grid.legend:
            legend.add_row(
                Text(
                    self._theme.glyph(entry.role),
                    style=self._theme.legend_glyph_style(
                        entry.role, collected=entry.collected
                    ),
                ),
                Text(
                    self._label(entry),
                    style=self._theme.legend_label_style(collected=entry.collected),
                ),
            )

        return legend

    def _label(self, entry: LegendEntryView) -> str:
        """Return the label of a legend entry in its current state."""
        if entry.key is LegendKey.DOOR:
            return (
                "door (open)" if entry.role is CellRole.DOOR_OPEN else "door (closed)"
            )

        return str(entry.key)


class PlanPanel:
    """Show the tick, the executing task, and the remaining plan."""

    def __init__(self, theme: GridWorldTheme) -> None:
        """Initialize the panel with a theme."""
        self._theme = theme

    def render(self, frame: FrameView) -> RenderableType:
        """Return the plan panel."""
        return Panel(
            self._body(frame.plan),
            title=Text("Plan", style=f"bold {self._theme.palette.htn_blue}"),
            border_style=self._theme.border_style(),
            padding=(0, 1),
        )

    def _body(self, plan: PlanView) -> Group:
        """Return the plan rows, scrolled so the executing task stays visible."""
        header = f"tick {plan.tick}"
        if plan.replanned:
            header = f"{header} · replanned"

        rows: list[RenderableType] = [Text(header, style=self._theme.muted_style())]

        if not plan.entries:
            rows.append(Text(PLACEHOLDER, style=self._theme.muted_style()))
            return Group(*rows)

        start, window = self._window(plan.entries)

        rows.append(self._overflow_row(start, "↑"))

        for entry in window:
            rows.append(self._entry_row(entry, plan.current_role))

        rows.append(self._overflow_row(len(plan.entries) - start - len(window), "↓"))

        return Group(*rows)

    def _window(
        self, entries: tuple[PlanEntryView, ...]
    ) -> tuple[int, tuple[PlanEntryView, ...]]:
        """Return the visible slice of the plan and where it starts.

        The window follows the executing task, so a plan longer than the panel
        scrolls instead of hiding the part the agent is working on.
        """
        if len(entries) <= MAX_PLAN_ROWS:
            return 0, entries

        focus = self._focus_index(entries)
        start = min(max(focus - MAX_PLAN_ROWS // 2, 0), len(entries) - MAX_PLAN_ROWS)

        return start, entries[start : start + MAX_PLAN_ROWS]

    def _focus_index(self, entries: tuple[PlanEntryView, ...]) -> int:
        """Return the index the window should keep visible."""
        for index, entry in enumerate(entries):
            if entry.status is PlanEntryStatus.CURRENT:
                return index

        return len(entries) - 1

    def _entry_row(self, entry: PlanEntryView, current_role: MessageRole) -> Text:
        """Return one plan row, marked by its progress."""
        if entry.status is PlanEntryStatus.DONE:
            return Text(
                f"✓ {entry.name}",
                style=f"{self._theme.muted_style()} strike",
            )

        if entry.status is PlanEntryStatus.CURRENT:
            return Text(
                f"▶ {entry.name}",
                style=f"bold {self._theme.message_color(current_role)}",
            )

        return Text(f"· {entry.name}", style=self._theme.palette.ink_text)

    def _overflow_row(self, hidden: int, arrow: str) -> Text:
        """Return a row announcing tasks scrolled out of view."""
        if hidden <= 0:
            return Text("")

        return Text(f"{arrow} {hidden} more", style=self._theme.muted_style())


class DecompositionPanel:
    """Show the domain decomposition and method applicability."""

    def __init__(self, theme: GridWorldTheme) -> None:
        """Initialize the panel with a theme."""
        self._theme = theme

    def render(self, frame: FrameView) -> RenderableType:
        """Return the decomposition panel."""
        return Panel(
            self._body(frame.decomposition),
            title=Text(
                "HTN methods (observed state)",
                style=f"bold {self._theme.palette.htn_blue}",
            ),
            border_style=self._theme.border_style(),
            padding=(0, 1),
        )

    def _body(self, decomposition: DecompositionView) -> Group:
        """Return the annotated decomposition rows."""
        if not decomposition.nodes:
            return Group(Text(PLACEHOLDER, style=self._theme.muted_style()))

        rows: list[RenderableType] = []

        for node in decomposition.nodes[:MAX_TREE_ROWS]:
            indent = "  " * node.depth
            marker = "▸" if node.is_compound else "·"
            style = (
                f"bold {self._theme.palette.game_green}"
                if node.is_current
                else self._theme.palette.ink_text
            )
            rows.append(Text(f"{indent}{marker} {node.name}", style=style))

            for method in node.methods:
                rows.append(self._method_row(method, node.depth))

        return Group(*rows)

    def _method_row(self, method: object, depth: int) -> Text:
        """Return one method row, marked applicable or rejected."""
        indent = "  " * (depth + 1)
        applicable = getattr(method, "applicable", False)
        name = getattr(method, "id", "")
        failed = getattr(method, "failed_preconditions", ())

        if applicable:
            return Text(f"{indent}✓ {name}", style=self._theme.palette.game_green)

        reason = ", ".join(failed) if failed else "not applicable"
        return Text(f"{indent}✗ {name} ({reason})", style=self._theme.muted_style())


class ObjectivePanel:
    """Show the objective vector, preferences, and scalarized utility.

    Rows are always emitted so the panel keeps a constant height with and
    without a learner, which keeps exported frames uniform and makes the space
    reserved for future MORL output visible.
    """

    def __init__(self, theme: GridWorldTheme) -> None:
        """Initialize the panel with a theme."""
        self._theme = theme

    def render(self, frame: FrameView) -> RenderableType:
        """Return the objective panel."""
        objectives = frame.objectives or ObjectiveView()

        return Panel(
            self._body(objectives),
            title=Text("Objectives", style=f"bold {self._theme.palette.morl_red}"),
            border_style=self._theme.border_style(),
            padding=(0, 1),
        )

    def _body(self, objectives: ObjectiveView) -> Table:
        """Return the objective table."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style=self._theme.muted_style())
        table.add_column(justify="right")
        table.add_column(justify="right")
        table.add_column(justify="right")

        table.add_row(
            Text("objective", style=self._theme.muted_style()),
            Text("r", style=self._theme.muted_style()),
            Text("return", style=self._theme.muted_style()),
            Text("w", style=self._theme.muted_style()),
        )

        for index, name in enumerate(objectives.names):
            table.add_row(
                Text(name, style=self._theme.palette.ink_text),
                self._value(objectives.step_reward, index),
                self._value(objectives.episode_return, index),
                self._value(objectives.weights, index),
            )

        table.add_row(
            Text("WᵀQ", style=self._theme.palette.ink_text),
            Text(""),
            Text(""),
            self._utility(objectives.utility),
        )

        return table

    def _value(self, values: tuple[float, ...] | None, index: int) -> Text:
        """Return a formatted value, or a placeholder when absent."""
        if values is None or index >= len(values):
            return Text(PLACEHOLDER, style=self._theme.muted_style())

        return Text(f"{values[index]:.2f}", style=self._theme.palette.ink_text)

    def _utility(self, utility: float | None) -> Text:
        """Return the scalarized utility, or a placeholder naming the reason."""
        if utility is None:
            return Text(f"{PLACEHOLDER} (no learner)", style=self._theme.muted_style())

        return Text(f"{utility:.2f}", style=f"bold {self._theme.palette.morl_red}")
