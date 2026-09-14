"""Reactive navigation shared by every GridWorld variant."""

from __future__ import annotations

from typing import cast

from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.movement import action_from_step
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.core.world import GridWorld
from htn.actions.action import Action
from htn.actions.action_status import ActionStatus
from htn.world.world import World

MINIMUM_ROUTE_LENGTH = 2


class NavigateToPositionAction(Action):
    """Walk one tile per tick along a route recomputed every tick.

    Recomputing keeps navigation reactive to a world that changed since the
    plan was built, at the cost of repeating the search.
    """

    def __init__(
        self,
        target: Position,
        route_planner: RoutePlanner,
        *,
        profile_name: str | None = None,
    ) -> None:
        """Initialize the action.

        Args:
            target: Tile the agent must reach.
            route_planner: Planner used to search a route each tick.
            profile_name: Movement profile to travel with; the first profile of
                the environment when omitted.
        """
        self.target = target
        self._route_planner = route_planner
        self._profile_name = profile_name

    def execute(self, world: World) -> ActionStatus:
        """Advance one tile toward the target.

        Returns:
            ``SUCCESS`` once the agent stands on the target, ``RUNNING`` while
            it advances, and ``FAILURE`` when no route exists or the step made
            no progress.
        """
        grid_world = cast(GridWorld, world)
        env = grid_world.env

        if env.state.agent_position == self.target:
            return ActionStatus.SUCCESS

        route = self._route_planner.find_route(env.layout, env.state, self.target)

        if len(route) < MINIMUM_ROUTE_LENGTH:
            return ActionStatus.FAILURE

        origin = env.state.agent_position
        profile_index = self._resolve_profile_index(grid_world)
        action = action_from_step(
            route[0], route[1], env.codec, profile_index=profile_index
        )
        grid_world.step(action)

        if env.state.agent_position == self.target:
            return ActionStatus.SUCCESS

        if env.state.agent_position == origin:
            return ActionStatus.FAILURE

        return ActionStatus.RUNNING

    def _resolve_profile_index(self, grid_world: GridWorld) -> int:
        """Return the index of the movement profile this action travels with."""
        if self._profile_name is None:
            return 0

        return grid_world.env.codec.profile_index(self._profile_name)
