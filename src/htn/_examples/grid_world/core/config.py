"""Static GridWorld configuration and its validation rules."""

from __future__ import annotations

from dataclasses import dataclass, field

from htn._examples.grid_world.core.geometry import Position, is_inside_grid


@dataclass(frozen=True, slots=True)
class GridWorldConfig:
    """Static description of a GridWorld scenario.

    A ``None`` position is resolved randomly when the episode starts, so the
    same configuration can produce either a deterministic or a randomized
    layout.

    Attributes:
        width: Grid width in tiles.
        height: Grid height in tiles.
        start_position: Initial agent position, or None to randomize.
        key_position: Key position, or None to randomize.
        door_position: Door position, or None to randomize.
        goal_position: Goal position, or None to randomize.
        fixed_obstacles: Obstacles always present in the grid.
        random_obstacle_count: Additional obstacles sampled at reset.
        initial_has_key: Whether the agent starts holding the key.
        initial_door_open: Whether the door starts open.
    """

    width: int = 3
    height: int = 3

    start_position: Position | None = (0, 0)
    key_position: Position | None = (2, 0)
    door_position: Position | None = (2, 2)
    goal_position: Position | None = (0, 2)

    fixed_obstacles: frozenset[Position] = field(
        default_factory=lambda: frozenset({(1, 1)})
    )
    random_obstacle_count: int = 0

    initial_has_key: bool = False
    initial_door_open: bool = False


def validate_grid_world_config(config: GridWorldConfig) -> None:
    """Validate a configuration that does not depend on episode randomness.

    Args:
        config: The configuration to validate.

    Raises:
        ValueError: If dimensions, obstacle count, or fixed obstacles are
            inconsistent with the declared grid.
    """
    if config.width <= 0:
        raise ValueError("Grid width must be greater than zero.")

    if config.height <= 0:
        raise ValueError("Grid height must be greater than zero.")

    if config.random_obstacle_count < 0:
        raise ValueError("random_obstacle_count cannot be negative.")

    for obstacle in config.fixed_obstacles:
        if not is_inside_grid(obstacle, config.width, config.height):
            raise ValueError(f"Obstacle is outside the grid: {obstacle}")
