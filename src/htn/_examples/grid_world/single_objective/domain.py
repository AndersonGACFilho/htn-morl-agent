"""HTN domain of the key-door scenario."""

from __future__ import annotations

from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.navigation import NavigateToPositionAction
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.single_objective.actions import (
    OpenDoorAction,
    PickupKeyAction,
)
from htn.tasks.domains.domain import Domain
from htn.tasks.types.compound_task import CompoundTask
from htn.tasks.types.effects import Effects
from htn.tasks.types.method import Method
from htn.tasks.types.primitive_task import PrimitiveTask


def _position_effects(
    position: Position,
    *,
    at_key: bool = False,
    at_door: bool = False,
    at_goal: bool = False,
) -> Effects:
    """Build the symbolic effects of arriving at a position.

    Args:
        position: Position reached by the navigation task.
        at_key: Whether the agent should be considered at the key.
        at_door: Whether the agent should be considered at the door.
        at_goal: Whether the agent should be considered at the goal.

    Returns:
        Effects compatible with a primitive task.
    """
    x, y = position

    return {
        "agent_x": ("=", x),
        "agent_y": ("=", y),
        "at_key": ("=", at_key),
        "at_door": ("=", at_door),
        "at_goal": ("=", at_goal),
    }


def build_key_door_domain(layout: GridLayout, route_planner: RoutePlanner) -> Domain:
    """Build the key-door HTN domain for a resolved layout.

    The domain expresses only symbolic intentions — ensure the key is
    collected, ensure the door is open, reach the goal — while the concrete
    coordinates come from the episode layout.

    Args:
        layout: Entity placement of the current episode.
        route_planner: Planner used by every navigation task.

    Returns:
        The domain rooted at ``escape_grid``.
    """
    go_to_key = PrimitiveTask(
        name="go_to_key",
        action=NavigateToPositionAction(layout.key_position, route_planner),
        preconditions={"has_key": ("=", False)},
        effects=_position_effects(layout.key_position, at_key=True),
    )

    pickup_key = PrimitiveTask(
        name="pickup_key",
        action=PickupKeyAction(),
        preconditions={"at_key": ("=", True), "has_key": ("=", False)},
        effects={"has_key": ("=", True), "at_key": ("=", True)},
    )

    go_to_door = PrimitiveTask(
        name="go_to_door",
        action=NavigateToPositionAction(layout.door_position, route_planner),
        preconditions={"has_key": ("=", True), "door_open": ("=", False)},
        effects=_position_effects(layout.door_position, at_door=True),
    )

    open_door = PrimitiveTask(
        name="open_door",
        action=OpenDoorAction(),
        preconditions={
            "at_door": ("=", True),
            "has_key": ("=", True),
            "door_open": ("=", False),
        },
        effects={"door_open": ("=", True)},
    )

    go_to_goal_effects = _position_effects(layout.goal_position, at_goal=True)
    go_to_goal_effects["done"] = ("=", True)

    go_to_goal = PrimitiveTask(
        name="go_to_goal",
        action=NavigateToPositionAction(layout.goal_position, route_planner),
        preconditions={"door_open": ("=", True), "done": ("=", False)},
        effects=go_to_goal_effects,
    )

    ensure_has_key = CompoundTask(
        name="ensure_has_key",
        methods=[
            Method(
                id="ensure_has_key.already_has_key",
                name="Already has key",
                preconditions={"has_key": ("=", True)},
                tasks=[],
            ),
            Method(
                id="ensure_has_key.acquire_key",
                name="Acquire key",
                preconditions={"has_key": ("=", False)},
                tasks=[go_to_key, pickup_key],
            ),
        ],
    )

    ensure_door_open = CompoundTask(
        name="ensure_door_open",
        methods=[
            Method(
                id="ensure_door_open.already_open",
                name="Door already open",
                preconditions={"door_open": ("=", True)},
                tasks=[],
            ),
            Method(
                id="ensure_door_open.open_door",
                name="Open door",
                preconditions={"door_open": ("=", False), "has_key": ("=", True)},
                tasks=[go_to_door, open_door],
            ),
        ],
    )

    escape_grid = CompoundTask(
        name="escape_grid",
        methods=[
            Method(
                id="escape_grid.exit_through_open_door",
                name="Exit through open door",
                preconditions={"done": ("=", False), "door_open": ("=", True)},
                tasks=[go_to_goal],
            ),
            Method(
                id="escape_grid.prepare_and_exit",
                name="Prepare and exit",
                preconditions={"done": ("=", False), "door_open": ("=", False)},
                tasks=[ensure_has_key, ensure_door_open, go_to_goal],
            ),
        ],
    )

    return Domain(tasks=[escape_grid])
