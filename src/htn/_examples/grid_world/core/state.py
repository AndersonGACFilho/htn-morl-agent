"""Immutable episode state for the GridWorld environment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Self

from htn._examples.grid_world.core.geometry import Position


@dataclass(frozen=True, slots=True)
class GridWorldState:
    """Mutable facts of one episode, represented immutably.

    The state is frozen so that rules are pure functions of the layout, the
    current state, and a command. That keeps them testable without an
    environment and lets callers evaluate a hypothetical step without
    disturbing the live episode.

    Attributes:
        agent_position: Where the agent currently stands.
        has_key: Whether the agent holds the key.
        door_open: Whether the door has been opened.
        done: Whether the episode reached its terminal condition.
    """

    agent_position: Position
    has_key: bool = False
    door_open: bool = False
    done: bool = False

    def with_changes(self, **changes: Any) -> Self:
        """Return a copy of this state with the given fields replaced.

        The return type follows the concrete subclass, so a variant that adds
        fields keeps them through every transformation.
        """
        return replace(self, **changes)
