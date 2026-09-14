"""Tests for the movement and interaction semantics."""

from __future__ import annotations

import pytest
from htn._examples.grid_world.core.geometry import DOWN, RIGHT, UP
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.movement_profile import (
    WALK,
    InteractionCommand,
    MoveCommand,
)
from htn._examples.grid_world.core.rules import OPEN_DOOR, PICKUP_KEY, GridRules
from htn._examples.grid_world.core.state import GridWorldState


def test_move_into_obstacle_leaves_agent_position_unchanged(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(1, 0))

    result = rules.apply(layout, state, MoveCommand(profile=WALK, delta=DOWN))

    assert result.agent_position == (1, 0)


def test_move_outside_grid_leaves_agent_position_unchanged(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    result = rules.apply(layout, state, MoveCommand(profile=WALK, delta=UP))

    assert result.agent_position == (0, 0)


def test_move_into_goal_is_blocked_while_door_is_closed(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 1), door_open=False)

    result = rules.apply(layout, state, MoveCommand(profile=WALK, delta=DOWN))

    assert result.agent_position == (0, 1)


def test_move_into_goal_is_allowed_once_the_door_is_open(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 1), door_open=True)

    result = rules.apply(layout, state, MoveCommand(profile=WALK, delta=DOWN))

    assert result.agent_position == (0, 2)


def test_reaching_the_goal_with_an_open_door_terminates_the_episode(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 1), door_open=True)

    result = rules.apply(layout, state, MoveCommand(profile=WALK, delta=DOWN))

    assert result.done


def test_apply_does_not_mutate_the_input_state(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    rules.apply(layout, state, MoveCommand(profile=WALK, delta=RIGHT))

    assert state.agent_position == (0, 0)


def test_pickup_key_requires_standing_on_the_key_tile(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    result = rules.apply(layout, state, InteractionCommand(name=PICKUP_KEY))

    assert not result.has_key


def test_pickup_key_succeeds_on_the_key_tile(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=layout.key_position)

    result = rules.apply(layout, state, InteractionCommand(name=PICKUP_KEY))

    assert result.has_key


def test_open_door_requires_the_key(layout: GridLayout, rules: GridRules) -> None:
    state = GridWorldState(agent_position=layout.door_position, has_key=False)

    result = rules.apply(layout, state, InteractionCommand(name=OPEN_DOOR))

    assert not result.door_open


def test_open_door_succeeds_at_the_door_with_the_key(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=layout.door_position, has_key=True)

    result = rules.apply(layout, state, InteractionCommand(name=OPEN_DOOR))

    assert result.door_open


def test_unknown_interaction_raises(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    with pytest.raises(ValueError, match="Unknown interaction"):
        rules.apply(layout, state, InteractionCommand(name="teleport"))


def test_is_passable_rejects_the_locked_goal_tile(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    assert not rules.is_passable(layout, state, layout.goal_position)


def test_blocked_positions_include_the_goal_while_the_door_is_closed(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    assert layout.goal_position in rules.blocked_positions(layout, state)


def test_blocked_positions_exclude_the_goal_once_the_door_is_open(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 0), door_open=True)

    assert layout.goal_position not in rules.blocked_positions(layout, state)
