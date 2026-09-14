"""Tests for configuration validation and layout resolution."""

from __future__ import annotations

import numpy as np
import pytest
from htn._examples.grid_world.core.config import (
    GridWorldConfig,
    validate_grid_world_config,
)
from htn._examples.grid_world.core.layout import LayoutResolver

RANDOMIZED_CONFIG = GridWorldConfig(
    width=5,
    height=5,
    start_position=None,
    key_position=None,
    door_position=None,
    goal_position=None,
    fixed_obstacles=frozenset({(2, 2)}),
    random_obstacle_count=3,
)


def test_validate_rejects_a_non_positive_width() -> None:
    config = GridWorldConfig(width=0)

    with pytest.raises(ValueError, match="width"):
        validate_grid_world_config(config)


def test_validate_rejects_an_obstacle_outside_the_grid() -> None:
    config = GridWorldConfig(fixed_obstacles=frozenset({(9, 9)}))

    with pytest.raises(ValueError, match="outside the grid"):
        validate_grid_world_config(config)


def test_validate_rejects_a_negative_random_obstacle_count() -> None:
    config = GridWorldConfig(random_obstacle_count=-1)

    with pytest.raises(ValueError, match="random_obstacle_count"):
        validate_grid_world_config(config)


def test_resolver_returns_an_identical_layout_for_the_same_seed() -> None:
    resolver = LayoutResolver(RANDOMIZED_CONFIG)

    first = resolver.resolve(np.random.default_rng(42))
    second = resolver.resolve(np.random.default_rng(42))

    assert first == second


def test_resolver_never_places_the_key_on_an_obstacle(
    seeded_rng: np.random.Generator,
) -> None:
    resolved = LayoutResolver(RANDOMIZED_CONFIG).resolve(seeded_rng)

    assert resolved.key_position not in resolved.obstacles


def test_resolver_places_every_entity_on_a_distinct_tile(
    seeded_rng: np.random.Generator,
) -> None:
    resolved = LayoutResolver(RANDOMIZED_CONFIG).resolve(seeded_rng)
    entities = {
        resolved.start_position,
        resolved.key_position,
        resolved.door_position,
        resolved.goal_position,
    }

    assert len(entities) == 4


def test_resolver_rejects_a_fixed_position_overlapping_an_obstacle(
    seeded_rng: np.random.Generator,
) -> None:
    config = GridWorldConfig(
        width=3, height=3, start_position=(1, 1), fixed_obstacles=frozenset({(1, 1)})
    )

    with pytest.raises(ValueError, match="overlaps an obstacle"):
        LayoutResolver(config).resolve(seeded_rng)


def test_resolver_raises_when_no_free_cell_remains(
    seeded_rng: np.random.Generator,
) -> None:
    config = GridWorldConfig(
        width=2,
        height=2,
        start_position=None,
        key_position=None,
        door_position=None,
        goal_position=None,
        fixed_obstacles=frozenset({(0, 0), (1, 1)}),
    )

    with pytest.raises(ValueError, match="no free cells"):
        LayoutResolver(config).resolve(seeded_rng)
