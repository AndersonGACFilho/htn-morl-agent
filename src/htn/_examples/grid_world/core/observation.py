"""Translation of episode state into a Gymnasium observation."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.state import GridWorldState


class GridObservationBuilder:
    """Build the observation space and observations for a grid of fixed size."""

    def __init__(self, width: int, height: int) -> None:
        """Initialize the builder for a grid of the given dimensions."""
        self._width = width
        self._height = height

    def space(self) -> spaces.Dict:
        """Return the observation space described by this builder."""
        return spaces.Dict(
            {
                "agent": self._position_space(),
                "key": self._position_space(),
                "door": self._position_space(),
                "goal": self._position_space(),
                "has_key": spaces.Discrete(2),
                "door_open": spaces.Discrete(2),
                "done": spaces.Discrete(2),
            }
        )

    def build(self, layout: GridLayout, state: GridWorldState) -> dict[str, Any]:
        """Return the observation for the given layout and state."""
        return {
            "agent": np.array(state.agent_position, dtype=np.int32),
            "key": np.array(layout.key_position, dtype=np.int32),
            "door": np.array(layout.door_position, dtype=np.int32),
            "goal": np.array(layout.goal_position, dtype=np.int32),
            "has_key": int(state.has_key),
            "door_open": int(state.door_open),
            "done": int(state.done),
        }

    def _position_space(self) -> spaces.Box:
        """Return the space describing one grid coordinate."""
        return spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([self._width - 1, self._height - 1], dtype=np.int32),
            shape=(2,),
            dtype=np.int32,
        )
