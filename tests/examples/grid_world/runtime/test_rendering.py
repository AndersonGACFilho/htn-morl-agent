"""Tests for the renderer, the theme, and frame export."""

from __future__ import annotations

from pathlib import Path

import pytest
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.rendering.export import GifBuilder, SvgFrameExporter
from htn._examples.grid_world.core.rendering.panels import PlanPanel
from htn._examples.grid_world.core.rendering.renderer import GridWorldRenderer
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState
from htn._examples.grid_world.core.theme import GridWorldTheme
from htn._examples.grid_world.core.view.grid_view_builder import GridViewBuilder
from htn._examples.grid_world.core.view.models import (
    CellRole,
    CellView,
    FrameView,
    GridView,
    ObjectiveView,
    PlanEntryStatus,
    PlanEntryView,
    PlanView,
)
from rich.console import Console


def _frame(
    layout: GridLayout,
    rules: GridRules,
    state: GridWorldState,
    *,
    objectives: ObjectiveView | None = None,
) -> FrameView:
    """Return a frame for the given episode state."""
    return FrameView(
        grid=GridViewBuilder(rules).build(layout, state),
        plan=PlanView(
            tick=1,
            entries=(
                PlanEntryView(name="go_to_key", status=PlanEntryStatus.CURRENT),
                PlanEntryView(name="pickup_key", status=PlanEntryStatus.PENDING),
            ),
        ),
        objectives=objectives,
    )


def test_theme_exposes_a_distinct_glyph_for_each_cell_role() -> None:
    theme = GridWorldTheme.dissertation()

    glyphs = {theme.glyph(role) for role in CellRole}

    assert len(glyphs) == len(CellRole)


def test_theme_palettes_differ_between_the_two_manuscripts() -> None:
    assert GridWorldTheme.dissertation().palette != GridWorldTheme.eramia().palette


def test_theme_rejects_an_unknown_palette_name() -> None:
    with pytest.raises(ValueError, match="Unknown palette"):
        GridWorldTheme.by_name("nonexistent")


def test_theme_marks_the_target_cell_differently_from_a_plain_one() -> None:
    theme = GridWorldTheme.dissertation()
    plain = CellView(position=(0, 0), role=CellRole.EMPTY)
    target = CellView(position=(0, 0), role=CellRole.EMPTY, is_target=True)

    assert theme.cell_style(plain) != theme.cell_style(target)


def test_theme_marks_a_risky_cell_differently_from_a_safe_one() -> None:
    theme = GridWorldTheme.dissertation()
    safe = CellView(position=(0, 0), role=CellRole.EMPTY, risk=0.0)
    risky = CellView(position=(0, 0), role=CellRole.EMPTY, risk=0.9)

    assert theme.cell_style(safe) != theme.cell_style(risky)


def test_theme_strikes_the_legend_glyph_of_a_collected_item() -> None:
    theme = GridWorldTheme.dissertation()

    style = theme.legend_glyph_style(CellRole.KEY, collected=True)

    assert "strike" in style


def test_theme_leaves_the_legend_glyph_unstruck_before_collection() -> None:
    theme = GridWorldTheme.dissertation()

    style = theme.legend_glyph_style(CellRole.KEY, collected=False)

    assert "strike" not in style


def test_renderer_exports_frames_with_identical_dimensions(
    tmp_path: Path, layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    theme = GridWorldTheme.dissertation()
    renderer = GridWorldRenderer(theme, show_objectives=True)
    exporter = SvgFrameExporter(tmp_path, theme, title_prefix="test")

    renderer.render(_frame(layout, rules, state))
    first = exporter.export(renderer.console, 0)

    renderer.render(
        _frame(
            layout,
            rules,
            state,
            objectives=ObjectiveView(step_reward=(-1.0, -2.0, -3.0)),
        )
    )
    second = exporter.export(renderer.console, 1)

    assert _svg_size(first) == _svg_size(second)


def test_gif_builder_rejects_an_empty_frame_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one frame"):
        GifBuilder().build([], tmp_path / "episode.gif")


def test_gif_builder_rejects_a_non_positive_frame_duration() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        GifBuilder(frame_duration_ms=0)


def _svg_size(path: Path) -> tuple[str, str]:
    """Return the declared width and height of an exported SVG."""
    content = path.read_text(encoding="utf-8")
    width = content.split('width="', 1)[1].split('"', 1)[0]
    height = content.split('height="', 1)[1].split('"', 1)[0]

    return width, height


def _plan(total: int, current: int) -> PlanView:
    """Return a plan of ``total`` tasks with ``current`` executing."""
    entries = []
    for index in range(total):
        if index < current:
            status = PlanEntryStatus.DONE
        elif index == current:
            status = PlanEntryStatus.CURRENT
        else:
            status = PlanEntryStatus.PENDING
        entries.append(PlanEntryView(name=f"task_{index}", status=status))

    return PlanView(tick=current, entries=tuple(entries))


def _render_plan(plan: PlanView) -> str:
    """Return the plain text of the plan panel for the given plan."""
    console = Console(record=True, width=46, height=24, legacy_windows=False)
    console.print(
        PlanPanel(GridWorldTheme.dissertation()).render(
            FrameView(grid=_empty_grid(), plan=plan)
        )
    )

    return console.export_text()


def _empty_grid() -> GridView:
    """Return a one-tile grid, irrelevant to the plan panel."""
    return GridView(
        width=1,
        height=1,
        rows=((CellView(position=(0, 0), role=CellRole.EMPTY),),),
    )


def test_plan_panel_keeps_completed_tasks_visible() -> None:
    text = _render_plan(_plan(total=4, current=2))

    assert "task_0" in text


def test_plan_panel_marks_completed_tasks_as_done() -> None:
    text = _render_plan(_plan(total=4, current=2))

    assert "✓ task_0" in text


def test_plan_panel_marks_the_executing_task() -> None:
    text = _render_plan(_plan(total=4, current=2))

    assert "▶ task_2" in text


def test_plan_panel_shows_every_task_of_a_plan_that_fits() -> None:
    text = _render_plan(_plan(total=4, current=1))

    assert all(f"task_{index}" in text for index in range(4))


def test_plan_panel_scrolls_a_plan_longer_than_the_window() -> None:
    text = _render_plan(_plan(total=20, current=15))

    assert "▶ task_15" in text


def test_plan_panel_announces_tasks_scrolled_above_the_window() -> None:
    text = _render_plan(_plan(total=20, current=15))

    assert "↑" in text


def test_plan_panel_announces_tasks_scrolled_below_the_window() -> None:
    text = _render_plan(_plan(total=20, current=2))

    assert "↓" in text


def test_plan_panel_reports_an_empty_plan_with_a_placeholder() -> None:
    text = _render_plan(PlanView(tick=0))

    assert "—" in text
