"""HTN domain of the multi-objective scenario.

The scenario keeps the key-door structure and adds competing ways to reach the
goal, so a method choice expresses a real trade-off between time, energy, and
safety rather than a fixed execution order.
"""

from __future__ import annotations

from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.navigation import NavigateToPositionAction
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.multi_objective.actions import RechargeAction
from htn._examples.grid_world.multi_objective.config import (
    MOVEMENT_PROFILES,
    MultiObjectiveGridConfig,
)
from htn._examples.grid_world.single_objective.actions import (
    OpenDoorAction,
    PickupKeyAction,
)
from htn._examples.grid_world.single_objective.domain import _position_effects
from htn.tasks.domains.domain import Domain
from htn.tasks.types.compound_task import CompoundTask
from htn.tasks.types.method import Method
from htn.tasks.types.primitive_task import PrimitiveTask

RUN_PROFILE = MOVEMENT_PROFILES[1].name
SAFE_PROFILE = MOVEMENT_PROFILES[2].name


def _navigate(
    name: str,
    target: Position,
    route_planner: RoutePlanner,
    *,
    profile_name: str | None = None,
    preconditions: dict | None = None,
    effects: dict | None = None,
) -> PrimitiveTask:
    """Build a navigation task travelling with a given movement profile."""
    return PrimitiveTask(
        name=name,
        action=NavigateToPositionAction(
            target, route_planner, profile_name=profile_name
        ),
        preconditions=preconditions,
        effects=effects,
    )


def build_multi_objective_domain(
    layout: GridLayout,
    route_planner: RoutePlanner,
    config: MultiObjectiveGridConfig,
) -> Domain:
    """Build the multi-objective HTN domain for a resolved layout.

    Args:
        layout: Entity placement of the current episode.
        route_planner: Planner used by every navigation task.
        config: Scenario parameters, used for the recharge tile.

    Returns:
        The domain rooted at ``escape_grid``.
    """
    go_to_key = _navigate(
        "go_to_key",
        layout.key_position,
        route_planner,
        preconditions={"has_key": ("=", False)},
        effects=_position_effects(layout.key_position, at_key=True),
    )

    pickup_key = PrimitiveTask(
        name="pickup_key",
        action=PickupKeyAction(),
        preconditions={"at_key": ("=", True), "has_key": ("=", False)},
        effects={"has_key": ("=", True), "at_key": ("=", True)},
    )

    go_to_door = _navigate(
        "go_to_door",
        layout.door_position,
        route_planner,
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

    goal_effects = _position_effects(layout.goal_position, at_goal=True)
    goal_effects["done"] = ("=", True)

    run_to_goal = _navigate(
        "run_to_goal",
        layout.goal_position,
        route_planner,
        profile_name=RUN_PROFILE,
        preconditions={"door_open": ("=", True), "done": ("=", False)},
        effects=dict(goal_effects),
    )

    walk_to_goal = _navigate(
        "walk_to_goal",
        layout.goal_position,
        route_planner,
        preconditions={"door_open": ("=", True), "done": ("=", False)},
        effects=dict(goal_effects),
    )

    move_safely_to_goal = _navigate(
        "move_safely_to_goal",
        layout.goal_position,
        route_planner,
        profile_name=SAFE_PROFILE,
        preconditions={"door_open": ("=", True), "done": ("=", False)},
        effects=dict(goal_effects),
    )

    recharge_position = _recharge_position(config, layout)

    go_to_recharge = _navigate(
        "go_to_recharge",
        recharge_position,
        route_planner,
        preconditions={"energy_is_low": ("=", True)},
        effects={
            "agent_x": ("=", recharge_position[0]),
            "agent_y": ("=", recharge_position[1]),
            "at_recharge": ("=", True),
        },
    )

    recharge = PrimitiveTask(
        name="recharge",
        action=RechargeAction(),
        preconditions={"at_recharge": ("=", True)},
        effects={"energy_is_low": ("=", False), "at_recharge": ("=", True)},
    )

    reach_goal = CompoundTask(
        name="reach_goal",
        methods=[
            Method(
                id="reach_goal.recharge_first",
                name="Recharge before crossing",
                preconditions={"energy_is_low": ("=", True), "done": ("=", False)},
                tasks=[go_to_recharge, recharge, walk_to_goal],
            ),
            Method(
                id="reach_goal.safe_route",
                name="Safe route",
                preconditions={
                    "route_is_dangerous": ("=", True),
                    "done": ("=", False),
                },
                tasks=[move_safely_to_goal],
            ),
            Method(
                id="reach_goal.direct_route",
                name="Direct route",
                preconditions={"done": ("=", False)},
                tasks=[run_to_goal],
            ),
        ],
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
                tasks=[reach_goal],
            ),
            Method(
                id="escape_grid.prepare_and_exit",
                name="Prepare and exit",
                preconditions={"done": ("=", False), "door_open": ("=", False)},
                tasks=[ensure_has_key, ensure_door_open, reach_goal],
            ),
        ],
    )

    return Domain(tasks=[escape_grid])


def _recharge_position(
    config: MultiObjectiveGridConfig, layout: GridLayout
) -> Position:
    """Return the recharge tile the domain navigates to.

    Falls back to the start position when the scenario declares no recharge
    tile, so the domain stays well formed even without one.
    """
    if config.recharge_positions:
        return sorted(config.recharge_positions)[0]

    return layout.start_position
