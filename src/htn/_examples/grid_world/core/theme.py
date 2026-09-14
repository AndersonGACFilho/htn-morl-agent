"""Glyphs, colors, and SVG palette for the rendered GridWorld.

The theme is the single authority on appearance. Cell symbols and colors used
to be hardcoded inside the renderer while the theme only remapped ANSI slots at
SVG-export time, so terminal output and exported frames could drift apart. One
palette now drives the terminal, the SVG frames, and the GIF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rich.terminal_theme import TerminalTheme

from htn._examples.grid_world.core.palette import (
    DISSERTATION_PALETTE,
    ERAMIA_PALETTE,
    Palette,
    get_palette,
)
from htn._examples.grid_world.core.view.models import CellRole, CellView, MessageRole

_GLYPHS: Mapping[CellRole, str] = {
    CellRole.AGENT: " A ",
    CellRole.OBSTACLE: " X ",
    CellRole.KEY: " K ",
    CellRole.DOOR_CLOSED: " D ",
    CellRole.DOOR_OPEN: " O ",
    CellRole.GOAL: " G ",
    CellRole.HAZARD: " ! ",
    CellRole.RECHARGE: " + ",
    CellRole.EMPTY: " . ",
}

_HEAT_STEPS = 4


def _to_rgb(color: str) -> tuple[int, int, int]:
    """Convert a ``#rrggbb`` string into an RGB triple."""
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _blend(start: str, end: str, ratio: float) -> str:
    """Return the color obtained by mixing two colors by ``ratio``."""
    bounded = min(max(ratio, 0.0), 1.0)
    start_rgb = _to_rgb(start)
    end_rgb = _to_rgb(end)
    mixed = (
        round(start_rgb[index] + (end_rgb[index] - start_rgb[index]) * bounded)
        for index in range(3)
    )
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


@dataclass(frozen=True, slots=True)
class GridWorldTheme:
    """Map view models onto Rich styles using one palette."""

    palette: Palette

    @classmethod
    def dissertation(cls) -> "GridWorldTheme":
        """Return the theme matching the dissertation figures."""
        return cls(palette=DISSERTATION_PALETTE)

    @classmethod
    def eramia(cls) -> "GridWorldTheme":
        """Return the theme matching the ERAMIA article figures."""
        return cls(palette=ERAMIA_PALETTE)

    @classmethod
    def by_name(cls, name: str) -> "GridWorldTheme":
        """Return the theme whose palette carries the given name."""
        return cls(palette=get_palette(name))

    @property
    def name(self) -> str:
        """Return the palette name of this theme."""
        return self.palette.name

    def glyph(self, role: CellRole) -> str:
        """Return the three-character glyph drawn for a tile role."""
        return _GLYPHS[role]

    def cell_style(self, cell: CellView) -> str:
        """Return the Rich style for a tile.

        The role color wins over any overlay so entities never disappear into a
        cost or risk heat map. Route and target markers are added on top.
        """
        style = self._role_style(cell)

        if cell.is_target:
            style = f"{style} reverse"
        elif cell.on_route and cell.role is CellRole.EMPTY:
            style = f"bold {self.palette.htn_blue} on {self._route_background()}"
        elif cell.on_route:
            style = f"{style} underline"

        return style

    def cell_style_for_role(self, role: CellRole) -> str:
        """Return the style of a bare tile role, as used by the legend."""
        return self._role_style(CellView(position=(0, 0), role=role))

    def legend_glyph_style(self, role: CellRole, *, collected: bool = False) -> str:
        """Return the legend glyph style, struck through once collected."""
        style = self.cell_style_for_role(role)

        return f"{style} strike" if collected else style

    def legend_label_style(self, *, collected: bool = False) -> str:
        """Return the legend label style, struck through once collected."""
        style = self.muted_style()

        return f"{style} strike" if collected else style

    def message_color(self, role: MessageRole) -> str:
        """Return the color conveying a status role."""
        colors = {
            MessageRole.NEUTRAL: self.palette.ink_text,
            MessageRole.SUCCESS: self.palette.game_green,
            MessageRole.FAILURE: self.palette.morl_red,
            MessageRole.RUNNING: self.palette.htn_blue,
            MessageRole.PLAN: self.palette.fallback_orange,
        }
        return colors[role]

    def message_style(self, role: MessageRole) -> str:
        """Return the Rich style for a status message."""
        if role is MessageRole.NEUTRAL:
            return self.palette.ink_text

        return f"bold {self.message_color(role)}"

    def border_style(self) -> str:
        """Return the Rich style used for panel borders."""
        return self.palette.htn_blue

    def muted_style(self) -> str:
        """Return the Rich style for placeholders and secondary text."""
        return f"dim {self.palette.env_gray}"

    def heat_style(self, normalized_value: float, *, kind: str) -> str:
        """Return a background color for a normalized cost or risk value.

        Args:
            normalized_value: Value in ``[0, 1]``.
            kind: Either ``"cost"`` or ``"risk"``.

        Raises:
            ValueError: If ``kind`` is not a known overlay.
        """
        if kind == "cost":
            target = self.palette.fallback_orange
        elif kind == "risk":
            target = self.palette.morl_red
        else:
            raise ValueError(f"Unknown overlay kind: {kind!r}")

        quantized = round(normalized_value * _HEAT_STEPS) / _HEAT_STEPS
        return _blend(self.palette.panel_background, target, quantized)

    def terminal_theme(self) -> TerminalTheme:
        """Return the Rich terminal theme used when exporting SVG frames."""
        background = _to_rgb(self.palette.panel_background)
        foreground = _to_rgb(self.palette.ink_text)
        colors = [
            _to_rgb(self.palette.ink_text),
            _to_rgb(self.palette.morl_red),
            _to_rgb(self.palette.game_green),
            _to_rgb(self.palette.fallback_orange),
            _to_rgb(self.palette.htn_blue),
            _to_rgb(self.palette.morl_red),
            _to_rgb(self.palette.htn_blue),
            _to_rgb(self.palette.panel_background),
        ]

        return TerminalTheme(background, foreground, colors, colors)

    def _role_style(self, cell: CellView) -> str:
        """Return the base style of a tile from its role and overlays."""
        palette = self.palette
        role_styles = {
            CellRole.AGENT: f"bold {palette.panel_background} on {palette.htn_blue}",
            CellRole.OBSTACLE: f"bold {palette.panel_background} on {palette.env_gray}",
            CellRole.KEY: f"bold {palette.ink_text} on {palette.fallback_orange}",
            CellRole.DOOR_CLOSED: (
                f"bold {palette.panel_background} on {palette.morl_red}"
            ),
            CellRole.DOOR_OPEN: f"bold {palette.panel_background} on {palette.game_green}",
            CellRole.GOAL: f"bold {palette.panel_background} on {palette.game_green}",
            CellRole.HAZARD: f"bold {palette.panel_background} on {palette.morl_red}",
            CellRole.RECHARGE: f"bold {palette.ink_text} on {palette.game_green}",
        }

        if cell.role in role_styles:
            return role_styles[cell.role]

        return self._empty_style(cell)

    def _empty_style(self, cell: CellView) -> str:
        """Return the style of an empty tile, including any heat overlay."""
        if cell.risk is not None and cell.risk > 0.0:
            return (
                f"{self.palette.env_gray} on {self.heat_style(cell.risk, kind='risk')}"
            )

        if cell.cost is not None and cell.cost > 0.0:
            return (
                f"{self.palette.env_gray} on {self.heat_style(cell.cost, kind='cost')}"
            )

        return self.palette.env_gray

    def _route_background(self) -> str:
        """Return the background used to mark a planned route tile."""
        return _blend(self.palette.panel_background, self.palette.htn_blue, 0.2)
