from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from htn._examples.grid_world.core.geometry import Position
from htn.pathfinding.pathfinder import Pathfinder


@dataclass(frozen=True)
class GridContext:
    width: int
    height: int
    blocked: frozenset[Position]


class GridPathfinder(Pathfinder[Position, GridContext]):
    def find_path(
        self, start: Position, goal: Position, context: GridContext
    ) -> list[Position]:
        """
        Breadth-first search pathfinder for an unweighted grid.

        This service is intentionally separate from the HTN planner.

        HTN decides:
            "go to key", "go to door", "go to goal"

        Pathfinder decides:
            which tile should be visited next.

        Args:
            start: Starting grid position.
            goal: Target grid position.
            context: Grid context containing width, height, and blocked positions.
        Returns:
            Path from start to goal, or an empty list if no path exists.
        """
        if start == goal:
            return [start]

        frontier: deque[Position] = deque([start])
        came_from: dict[Position, Position | None] = {start: None}

        while frontier:
            current = frontier.popleft()

            if current == goal:
                break

            for neighbor in self._neighbors(current, context.width, context.height):
                if neighbor in context.blocked:
                    continue

                if neighbor in came_from:
                    continue

                came_from[neighbor] = current
                frontier.append(neighbor)

        if goal not in came_from:
            return []

        return self._reconstruct_path(came_from, goal)

    def _neighbors(
        self,
        position: Position,
        width: int,
        height: int,
    ) -> Iterable[Position]:
        """
        Yield valid neighboring positions for a grid cell.

        Args:
            position: Grid position whose neighbors should be found.
            width: Grid width in tiles.
            height: Grid height in tiles.
        Returns:
            Iterable of valid neighboring positions.
        """
        x, y = position

        candidates = [
            (x, y - 1),  # up
            (x + 1, y),  # right
            (x, y + 1),  # down
            (x - 1, y),  # left
        ]

        for next_x, next_y in candidates:
            if 0 <= next_x < width and 0 <= next_y < height:
                yield next_x, next_y

    def _reconstruct_path(
        self,
        came_from: dict[Position, Position | None],
        goal: Position,
    ) -> list[Position]:
        """
        Reconstruct a path from the predecessor map.

        Args:
            came_from: Mapping from each visited position to its predecessor.
            goal: Goal position where reconstruction starts.
        Returns:
            Ordered path from start to goal.
        """
        path: list[Position] = []
        current: Position | None = goal

        while current is not None:
            path.append(current)
            current = came_from[current]

        path.reverse()
        return path
