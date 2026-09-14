"""Route planning that honours the environment's own passability rules."""

from __future__ import annotations

from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.pathfinder import GridContext, GridPathfinder
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState


class RoutePlanner:
    """Derive the blocked set from the rules and search a route through it.

    Building the blocked set here keeps navigation and the environment from
    disagreeing about which tiles are reachable.
    """

    def __init__(self, pathfinder: GridPathfinder, rules: GridRules) -> None:
        """Initialize the planner with a pathfinder and the environment rules."""
        self._pathfinder = pathfinder
        self._rules = rules

    def find_route(
        self,
        layout: GridLayout,
        state: GridWorldState,
        target: Position,
    ) -> list[Position]:
        """Return the shortest route from the agent to a target tile.

        The target itself is never treated as blocked, so the agent can reach a
        door or a goal tile that would otherwise be excluded.

        Returns:
            Positions from the agent to the target, or an empty list when no
            route exists.
        """
        blocked = set(self._rules.blocked_positions(layout, state))
        blocked.discard(target)

        context = GridContext(
            width=layout.width,
            height=layout.height,
            blocked=frozenset(blocked),
        )

        return self._pathfinder.find_path(state.agent_position, target, context)
