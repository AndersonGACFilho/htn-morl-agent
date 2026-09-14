"""Resolution of concrete entity placements for one GridWorld episode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from htn._examples.grid_world.core.config import GridWorldConfig
from htn._examples.grid_world.core.geometry import (
    Position,
    all_positions,
    is_inside_grid,
)


@dataclass(frozen=True, slots=True)
class GridLayout:
    """Immutable placement of every entity for one episode.

    Attributes:
        width: Grid width in tiles.
        height: Grid height in tiles.
        start_position: Where the agent begins the episode.
        key_position: Where the key lies.
        door_position: Where the door stands.
        goal_position: The tile that ends the episode.
        obstacles: Impassable tiles.
    """

    width: int
    height: int
    start_position: Position
    key_position: Position
    door_position: Position
    goal_position: Position
    obstacles: frozenset[Position]


class LayoutResolver:
    """Turn a configuration into a concrete layout for one episode.

    The resolver receives the random generator instead of owning one, so the
    same configuration and seed always produce the same layout and the class
    stays testable without a Gymnasium environment.
    """

    def __init__(self, config: GridWorldConfig) -> None:
        """Initialize the resolver for the given configuration."""
        self._config = config

    def resolve(self, rng: np.random.Generator) -> GridLayout:
        """Resolve every entity position and obstacle for a new episode.

        Args:
            rng: Random generator used for positions configured as None.

        Returns:
            The layout to use for the episode.

        Raises:
            ValueError: If a fixed position is invalid or the grid has no free
                cell left for a randomized entity.
        """
        obstacles = set(self._config.fixed_obstacles)
        reserved: set[Position] = set()

        start_position = self._resolve_position(
            self._config.start_position, reserved, obstacles, rng, name="start_position"
        )
        reserved.add(start_position)

        key_position = self._resolve_position(
            self._config.key_position, reserved, obstacles, rng, name="key_position"
        )
        reserved.add(key_position)

        door_position = self._resolve_position(
            self._config.door_position, reserved, obstacles, rng, name="door_position"
        )
        reserved.add(door_position)

        goal_position = self._resolve_position(
            self._config.goal_position, reserved, obstacles, rng, name="goal_position"
        )
        reserved.add(goal_position)

        self._add_random_obstacles(reserved, obstacles, rng)

        return GridLayout(
            width=self._config.width,
            height=self._config.height,
            start_position=start_position,
            key_position=key_position,
            door_position=door_position,
            goal_position=goal_position,
            obstacles=frozenset(obstacles),
        )

    def _resolve_position(
        self,
        configured_position: Position | None,
        reserved: set[Position],
        obstacles: set[Position],
        rng: np.random.Generator,
        *,
        name: str,
    ) -> Position:
        """Return a configured position after validation, or sample a free one.

        Args:
            configured_position: Fixed position, or None to sample one.
            reserved: Positions already taken by other entities.
            obstacles: Impassable tiles placed so far.
            rng: Random generator used when sampling.
            name: Entity name used in error messages.

        Returns:
            A position that is inside the grid and free.

        Raises:
            ValueError: If the configured position is outside the grid or
                overlaps an obstacle or another entity.
        """
        if configured_position is None:
            return self._sample_free_position(reserved | obstacles, obstacles, rng)

        if not is_inside_grid(
            configured_position, self._config.width, self._config.height
        ):
            raise ValueError(f"{name} is outside the grid: {configured_position}")

        if configured_position in obstacles:
            raise ValueError(f"{name} overlaps an obstacle: {configured_position}")

        if configured_position in reserved:
            raise ValueError(
                f"{name} overlaps another special position: {configured_position}"
            )

        return configured_position

    def _add_random_obstacles(
        self,
        reserved: set[Position],
        obstacles: set[Position],
        rng: np.random.Generator,
    ) -> None:
        """Place the configured number of random obstacles in free cells."""
        for _ in range(self._config.random_obstacle_count):
            obstacle = self._sample_free_position(reserved | obstacles, obstacles, rng)
            obstacles.add(obstacle)

    def _sample_free_position(
        self,
        blocked: set[Position],
        obstacles: set[Position],
        rng: np.random.Generator,
    ) -> Position:
        """Sample one grid position that is neither blocked nor an obstacle.

        Raises:
            ValueError: If every cell of the grid is already taken.
        """
        candidates = [
            position
            for position in all_positions(self._config.width, self._config.height)
            if position not in blocked and position not in obstacles
        ]

        if not candidates:
            raise ValueError("There are no free cells available for random placement.")

        return candidates[int(rng.integers(0, len(candidates)))]
