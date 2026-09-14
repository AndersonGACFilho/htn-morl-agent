"""Movement and interaction semantics of the GridWorld environment.

``GridRules`` is the single authority on what a step does. The environment
applies it, and navigation asks it which tiles are passable, so the planner's
route and the environment's movement can no longer disagree.
"""

from __future__ import annotations

from htn._examples.grid_world.core.geometry import (
    Position,
    is_inside_grid,
    translate,
)
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.movement_profile import (
    InteractionCommand,
    MoveCommand,
)
from htn._examples.grid_world.core.state import GridWorldState

PICKUP_KEY = "pickup_key"
OPEN_DOOR = "open_door"


class GridRules:
    """Apply commands to an episode state without mutating the input."""

    def apply(
        self,
        layout: GridLayout,
        state: GridWorldState,
        command: MoveCommand | InteractionCommand,
    ) -> GridWorldState:
        """Return the state produced by executing one command.

        A command that is not permitted leaves the world unchanged; the caller
        still advances the clock, so a refused step costs time.

        Args:
            layout: Entity placement for the episode.
            state: State before the command.
            command: The movement or interaction to execute.

        Returns:
            The resulting state, with the terminal flag refreshed.

        Raises:
            ValueError: If the interaction is unknown.
        """
        if isinstance(command, MoveCommand):
            moved = self._move(layout, state, command)
        else:
            moved = self._interact(layout, state, command)

        return self._evaluate_done(layout, moved)

    def is_passable(
        self,
        layout: GridLayout,
        state: GridWorldState,
        position: Position,
    ) -> bool:
        """Check whether the agent may occupy a tile in the current state.

        The goal tile stays locked until the door is open, which is why route
        planning must consult this method rather than the obstacle set alone.
        """
        if not is_inside_grid(position, layout.width, layout.height):
            return False

        if position in layout.obstacles:
            return False

        if position == layout.goal_position and not state.door_open:
            return False

        return True

    def blocked_positions(
        self,
        layout: GridLayout,
        state: GridWorldState,
    ) -> frozenset[Position]:
        """Return every tile that route planning must avoid right now."""
        blocked = set(layout.obstacles)

        if not state.door_open:
            blocked.add(layout.goal_position)

        return frozenset(blocked)

    def _move(
        self,
        layout: GridLayout,
        state: GridWorldState,
        command: MoveCommand,
    ) -> GridWorldState:
        """Return the state after attempting a move."""
        next_position = translate(state.agent_position, command.delta)

        if not self.is_passable(layout, state, next_position):
            return state

        return state.with_changes(agent_position=next_position)

    def _interact(
        self,
        layout: GridLayout,
        state: GridWorldState,
        command: InteractionCommand,
    ) -> GridWorldState:
        """Return the state after attempting an interaction."""
        if command.name == PICKUP_KEY:
            return self._pickup_key(layout, state)

        if command.name == OPEN_DOOR:
            return self._open_door(layout, state)

        raise ValueError(f"Unknown interaction: {command.name!r}")

    def _pickup_key(self, layout: GridLayout, state: GridWorldState) -> GridWorldState:
        """Pick up the key when the agent stands on the key tile."""
        if state.agent_position != layout.key_position:
            return state

        return state.with_changes(has_key=True)

    def _open_door(self, layout: GridLayout, state: GridWorldState) -> GridWorldState:
        """Open the door when the agent stands at it holding the key."""
        if state.agent_position != layout.door_position or not state.has_key:
            return state

        return state.with_changes(door_open=True)

    def _evaluate_done(
        self, layout: GridLayout, state: GridWorldState
    ) -> GridWorldState:
        """Refresh the terminal flag for the given state."""
        done = state.agent_position == layout.goal_position and state.door_open

        if done == state.done:
            return state

        return state.with_changes(done=done)
