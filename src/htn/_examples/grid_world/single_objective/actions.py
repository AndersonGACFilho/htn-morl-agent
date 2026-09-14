"""Interaction actions of the key-door scenario."""

from __future__ import annotations

from typing import cast

from htn._examples.grid_world.core.rules import OPEN_DOOR, PICKUP_KEY
from htn._examples.grid_world.core.world import GridWorld
from htn.actions.action import Action
from htn.actions.action_status import ActionStatus
from htn.world.world import World


class PickupKeyAction(Action):
    """Pick up the key when the agent stands on the key tile."""

    def execute(self, world: World) -> ActionStatus:
        """Attempt the pickup and report whether the agent now holds the key."""
        grid_world = cast(GridWorld, world)
        grid_world.step(grid_world.env.codec.encode_interaction(PICKUP_KEY))

        if grid_world.env.state.has_key:
            return ActionStatus.SUCCESS

        return ActionStatus.FAILURE


class OpenDoorAction(Action):
    """Open the door when the agent stands at it holding the key."""

    def execute(self, world: World) -> ActionStatus:
        """Attempt to open the door and report whether it is now open."""
        grid_world = cast(GridWorld, world)
        grid_world.step(grid_world.env.codec.encode_interaction(OPEN_DOOR))

        if grid_world.env.state.door_open:
            return ActionStatus.SUCCESS

        return ActionStatus.FAILURE
