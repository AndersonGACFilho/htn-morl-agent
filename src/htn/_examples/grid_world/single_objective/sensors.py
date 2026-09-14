"""Symbolic facts published by the key-door scenario."""

from __future__ import annotations

from htn._examples.grid_world.core.world import GridWorld
from htn.sensors import Sensor
from htn.world.state import WorldState


class KeyDoorSensor(Sensor[GridWorld]):
    """Map the concrete GridWorld environment into symbolic HTN facts.

    The domain depends on these names: changing one side without the other
    makes the plan impossible or incorrect.
    """

    def sense(self, world: GridWorld, world_state: WorldState) -> None:
        """Read the environment and refresh every fact the domain uses."""
        layout = world.env.layout
        state = world.env.state

        agent_x, agent_y = state.agent_position
        key_x, key_y = layout.key_position
        door_x, door_y = layout.door_position
        goal_x, goal_y = layout.goal_position

        world_state.set_state("agent_x", agent_x)
        world_state.set_state("agent_y", agent_y)

        world_state.set_state("key_x", key_x)
        world_state.set_state("key_y", key_y)

        world_state.set_state("door_x", door_x)
        world_state.set_state("door_y", door_y)

        world_state.set_state("goal_x", goal_x)
        world_state.set_state("goal_y", goal_y)

        world_state.set_state("has_key", state.has_key)
        world_state.set_state("door_open", state.door_open)
        world_state.set_state("done", state.done)

        world_state.set_state("at_key", state.agent_position == layout.key_position)
        world_state.set_state("at_door", state.agent_position == layout.door_position)
        world_state.set_state(
            "at_goal",
            state.agent_position == layout.goal_position and state.door_open,
        )
