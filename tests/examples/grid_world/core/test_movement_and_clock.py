"""Tests for the action codec, the step converter, and the episode clock."""

from __future__ import annotations

import pytest
from htn._examples.grid_world.core.clock import EpisodeClock
from htn._examples.grid_world.core.geometry import DIRECTIONS
from htn._examples.grid_world.core.movement import action_from_step
from htn._examples.grid_world.core.movement_profile import (
    WALK,
    InteractionCommand,
    MoveCommand,
    MovementActionCodec,
    MovementProfile,
)

RUN = MovementProfile(name="run", time_cost=0.5, energy_cost=3.0)


def _codec() -> MovementActionCodec:
    """Return a two-profile codec with two interactions."""
    return MovementActionCodec(
        profiles=(WALK, RUN), interactions=("pickup_key", "open_door")
    )


def test_single_profile_codec_reproduces_the_six_action_space() -> None:
    codec = MovementActionCodec(
        profiles=(WALK,), interactions=("pickup_key", "open_door")
    )

    assert codec.action_count == 6


def test_codec_round_trips_every_move_command() -> None:
    codec = _codec()

    decoded = [codec.decode(codec.encode_move(0, index)) for index in range(4)]

    assert [command.delta for command in decoded] == list(DIRECTIONS)


def test_codec_decodes_an_interaction_by_name() -> None:
    codec = _codec()

    decoded = codec.decode(codec.encode_interaction("open_door"))

    assert decoded == InteractionCommand(name="open_door")


def test_codec_decodes_the_second_profile_for_its_own_action_block() -> None:
    codec = _codec()

    decoded = codec.decode(codec.encode_move(1, 0))

    assert decoded == MoveCommand(profile=RUN, delta=DIRECTIONS[0])


def test_codec_rejects_an_action_outside_the_action_space() -> None:
    codec = _codec()

    with pytest.raises(ValueError, match="Invalid action"):
        codec.decode(codec.action_count)


def test_codec_requires_at_least_one_movement_profile() -> None:
    with pytest.raises(ValueError, match="movement profile is required"):
        MovementActionCodec(profiles=(), interactions=())


def test_codec_rejects_a_profile_with_negative_energy_cost() -> None:
    invalid = MovementProfile(name="free", time_cost=1.0, energy_cost=-1.0)

    with pytest.raises(ValueError, match="negative energy cost"):
        MovementActionCodec(profiles=(invalid,), interactions=())


def test_action_from_step_raises_for_non_adjacent_positions() -> None:
    with pytest.raises(ValueError, match="Invalid movement"):
        action_from_step((0, 0), (2, 2), _codec())


def test_action_from_step_encodes_a_step_to_the_right() -> None:
    codec = _codec()

    action = action_from_step((0, 0), (1, 0), codec)

    assert codec.decode(action).delta == (1, 0)


def test_clock_advances_by_the_movement_profile_duration() -> None:
    clock = EpisodeClock()

    clock.advance(RUN.time_cost)

    assert clock.elapsed_time == pytest.approx(0.5)


def test_clock_rejects_a_negative_duration() -> None:
    clock = EpisodeClock()

    with pytest.raises(ValueError, match="cannot be negative"):
        clock.advance(-1.0)


def test_clock_copy_is_independent_of_the_episode_clock() -> None:
    clock = EpisodeClock(elapsed_time=3.0)

    hypothetical = clock.copy()
    hypothetical.advance(5.0)

    assert clock.elapsed_time == pytest.approx(3.0)
