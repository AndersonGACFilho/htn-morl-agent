"""Conversion between adjacent grid steps and discrete action ids."""

from __future__ import annotations

from htn._examples.grid_world.core.geometry import DIRECTIONS, Position
from htn._examples.grid_world.core.movement_profile import MovementActionCodec


def action_from_step(
    current: Position,
    next_position: Position,
    codec: MovementActionCodec,
    *,
    profile_index: int = 0,
) -> int:
    """Convert a step onto an adjacent tile into an action id.

    Args:
        current: Current grid position.
        next_position: Orthogonally adjacent position to move to.
        codec: Encoding used by the target environment.
        profile_index: Movement profile to travel with.

    Returns:
        The action id that performs the step.

    Raises:
        ValueError: If the two positions are not orthogonally adjacent.
    """
    delta = (next_position[0] - current[0], next_position[1] - current[1])

    if delta not in DIRECTIONS:
        raise ValueError(f"Invalid movement from {current} to {next_position}")

    return codec.encode_move(profile_index, DIRECTIONS.index(delta))
