"""Grid coordinate primitives shared by layout, rules, and rendering."""

from __future__ import annotations

from typing import TypeAlias

Position: TypeAlias = tuple[int, int]

UP: Position = (0, -1)
RIGHT: Position = (1, 0)
DOWN: Position = (0, 1)
LEFT: Position = (-1, 0)

DIRECTIONS: tuple[Position, ...] = (UP, RIGHT, DOWN, LEFT)


def is_inside_grid(position: Position, width: int, height: int) -> bool:
    """Check whether a position lies inside a grid of the given dimensions."""
    x, y = position
    return 0 <= x < width and 0 <= y < height


def translate(position: Position, delta: Position) -> Position:
    """Return the position obtained by applying a movement delta."""
    return position[0] + delta[0], position[1] + delta[1]


def manhattan_distance(origin: Position, destination: Position) -> int:
    """Return the four-connected grid distance between two positions."""
    return abs(origin[0] - destination[0]) + abs(origin[1] - destination[1])


def all_positions(width: int, height: int) -> list[Position]:
    """Return every position of the grid in row-major order."""
    return [(x, y) for y in range(height) for x in range(width)]
