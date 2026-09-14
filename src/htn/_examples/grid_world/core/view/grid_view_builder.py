"""Construction of the grid view from a layout and an episode state."""

from __future__ import annotations

from typing import Protocol, Sequence

from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState
from htn._examples.grid_world.core.view.models import (
    CellRole,
    CellView,
    GridView,
    LegendEntryView,
    LegendKey,
)


class CostModel(Protocol):
    """Source of a traversal cost overlay."""

    def cost_at(self, position: Position) -> float:
        """Return the normalized traversal cost of a tile."""
        ...


class RiskField(Protocol):
    """Source of a perceived risk overlay."""

    def risk_at(self, position: Position) -> float:
        """Return the normalized perceived risk of a tile."""
        ...


class GridViewBuilder:
    """Turn environment structures into tiles the renderer can draw.

    The optional cost and risk sources keep the renderer ignorant of what a
    hazard or a movement profile is: a variant that models neither simply omits
    them and the overlay disappears.
    """

    def __init__(
        self,
        rules: GridRules,
        cost_model: CostModel | None = None,
        risk_field: RiskField | None = None,
        *,
        hazards: frozenset[Position] = frozenset(),
        recharge_positions: frozenset[Position] = frozenset(),
    ) -> None:
        """Initialize the builder with the rules and optional terrain sources.

        Args:
            rules: Rules governing the environment.
            cost_model: Optional source of a traversal-cost overlay.
            risk_field: Optional source of a perceived-risk overlay.
            hazards: Tiles a variant marks as hazardous.
            recharge_positions: Tiles a variant marks as recharge points.
        """
        self._rules = rules
        self._cost_model = cost_model
        self._risk_field = risk_field
        self._hazards = hazards
        self._recharge_positions = recharge_positions

    def build(
        self,
        layout: GridLayout,
        state: GridWorldState,
        *,
        route: Sequence[Position] = (),
        target: Position | None = None,
    ) -> GridView:
        """Build the grid view for the current episode.

        Args:
            layout: Entity placement of the episode.
            state: Current episode state.
            route: Planned route to highlight, if any.
            target: Tile the current task aims at, if any.

        Returns:
            The grid ready to be rendered.
        """
        route_positions = frozenset(route)

        rows = tuple(
            tuple(
                self._cell(layout, state, (x, y), route_positions, target)
                for x in range(layout.width)
            )
            for y in range(layout.height)
        )

        return GridView(
            width=layout.width,
            height=layout.height,
            rows=rows,
            legend=self._legend(layout, state),
        )

    def _legend(
        self, layout: GridLayout, state: GridWorldState
    ) -> tuple[LegendEntryView, ...]:
        """Return one legend entry per element this scenario declares.

        Entries depend on the scenario, not on what the current frame happens to
        show, so an element the agent stands on keeps its row instead of
        disappearing while it is occluded.
        """
        entries = [
            LegendEntryView(key=LegendKey.AGENT, role=CellRole.AGENT),
            LegendEntryView(
                key=LegendKey.KEY, role=CellRole.KEY, collected=state.has_key
            ),
            LegendEntryView(
                key=LegendKey.DOOR,
                role=CellRole.DOOR_OPEN if state.door_open else CellRole.DOOR_CLOSED,
            ),
            LegendEntryView(key=LegendKey.GOAL, role=CellRole.GOAL),
        ]

        if layout.obstacles:
            entries.append(
                LegendEntryView(key=LegendKey.OBSTACLE, role=CellRole.OBSTACLE)
            )

        if self._hazards:
            entries.append(LegendEntryView(key=LegendKey.HAZARD, role=CellRole.HAZARD))

        if self._recharge_positions:
            entries.append(
                LegendEntryView(key=LegendKey.RECHARGE, role=CellRole.RECHARGE)
            )

        return tuple(entries)

    def _cell(
        self,
        layout: GridLayout,
        state: GridWorldState,
        position: Position,
        route: frozenset[Position],
        target: Position | None,
    ) -> CellView:
        """Build one tile of the grid view."""
        role = self._role(layout, state, position)

        return CellView(
            position=position,
            role=role,
            cost=self._cost(position, role),
            risk=self._risk(position, role),
            on_route=position in route,
            is_target=target is not None and position == target,
        )

    def _role(
        self,
        layout: GridLayout,
        state: GridWorldState,
        position: Position,
    ) -> CellRole:
        """Return what a tile represents, agent first and obstacles next."""
        if position == state.agent_position:
            return CellRole.AGENT

        if position in layout.obstacles:
            return CellRole.OBSTACLE

        if position == layout.key_position and not state.has_key:
            return CellRole.KEY

        if position == layout.door_position:
            return CellRole.DOOR_OPEN if state.door_open else CellRole.DOOR_CLOSED

        if position == layout.goal_position:
            return CellRole.GOAL

        if position in self._recharge_positions:
            return CellRole.RECHARGE

        if position in self._hazards:
            return CellRole.HAZARD

        return CellRole.EMPTY

    def _cost(self, position: Position, role: CellRole) -> float | None:
        """Return the cost overlay of an empty tile, when a model is present."""
        if self._cost_model is None or role is not CellRole.EMPTY:
            return None

        return self._cost_model.cost_at(position)

    def _risk(self, position: Position, role: CellRole) -> float | None:
        """Return the risk overlay of an empty tile, when a field is present."""
        if self._risk_field is None or role is not CellRole.EMPTY:
            return None

        return self._risk_field.risk_at(position)
