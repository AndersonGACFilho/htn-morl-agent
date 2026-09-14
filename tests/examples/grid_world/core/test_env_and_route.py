"""Tests for the environment facade and route planning."""

from __future__ import annotations

import pytest
from htn._examples.grid_world.core.config import GridWorldConfig
from htn._examples.grid_world.core.env import GridWorldEnv
from htn._examples.grid_world.core.geometry import DIRECTIONS, DOWN
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.pathfinder import GridPathfinder
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState

DOWN_INDEX = DIRECTIONS.index(DOWN)


def _env() -> GridWorldEnv:
    """Return the deterministic 3x3 environment after a seeded reset."""
    env = GridWorldEnv(GridWorldConfig())
    env.reset(seed=42)
    return env


def test_reset_returns_an_observation_inside_the_observation_space() -> None:
    env = _env()

    observation, _ = env.reset(seed=42)

    assert env.observation_space.contains(observation)


def test_reset_places_the_agent_at_the_configured_start() -> None:
    env = _env()

    assert env.state.agent_position == env.layout.start_position


def test_step_rejects_an_action_outside_the_action_space() -> None:
    env = _env()

    with pytest.raises(ValueError, match="Invalid action"):
        env.step(env.codec.action_count)


def test_step_advances_the_episode_clock() -> None:
    env = _env()

    env.step(env.codec.encode_move(0, 1))

    assert env.clock.elapsed_time == pytest.approx(1.0)


def test_env_terminates_when_the_agent_reaches_the_goal_with_an_open_door() -> None:
    config = GridWorldConfig(
        width=2,
        height=2,
        start_position=(1, 0),
        key_position=(0, 0),
        door_position=(0, 1),
        goal_position=(1, 1),
        fixed_obstacles=frozenset(),
        initial_has_key=True,
        initial_door_open=True,
    )
    env = GridWorldEnv(config)
    env.reset(seed=1)

    _, reward, terminated, _, _ = env.step(env.codec.encode_move(0, DOWN_INDEX))

    assert terminated and reward == pytest.approx(1.0)


def test_route_planner_returns_an_empty_path_when_the_goal_is_enclosed() -> None:
    layout = GridLayout(
        width=3,
        height=3,
        start_position=(0, 0),
        key_position=(1, 0),
        door_position=(0, 1),
        goal_position=(2, 2),
        obstacles=frozenset({(2, 1), (1, 2)}),
    )
    planner = RoutePlanner(GridPathfinder(), GridRules())
    state = GridWorldState(agent_position=(0, 0), door_open=True)

    assert planner.find_route(layout, state, (2, 2)) == []


def test_route_planner_reaches_the_locked_goal_tile_as_an_explicit_target(
    layout: GridLayout,
) -> None:
    planner = RoutePlanner(GridPathfinder(), GridRules())
    state = GridWorldState(agent_position=(0, 0), door_open=False)

    route = planner.find_route(layout, state, layout.goal_position)

    assert route[-1] == layout.goal_position
