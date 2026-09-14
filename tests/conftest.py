"""Shared fixtures for the GridWorld and reward test suites."""

from __future__ import annotations

import numpy as np
import pytest
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState
from htn.world.state import WorldState


@pytest.fixture
def layout() -> GridLayout:
    """Return a 3x3 layout with one obstacle at the centre."""
    return GridLayout(
        width=3,
        height=3,
        start_position=(0, 0),
        key_position=(2, 0),
        door_position=(2, 2),
        goal_position=(0, 2),
        obstacles=frozenset({(1, 1)}),
    )


@pytest.fixture
def rules() -> GridRules:
    """Return the key-door rules."""
    return GridRules()


@pytest.fixture
def state(layout: GridLayout) -> GridWorldState:
    """Return the starting state of the 3x3 layout."""
    return GridWorldState(agent_position=layout.start_position)


@pytest.fixture
def seeded_rng() -> np.random.Generator:
    """Return a deterministic random generator."""
    return np.random.default_rng(42)


def build_world_state(**facts: object) -> WorldState:
    """Return a symbolic state holding the given facts."""
    world_state = WorldState()

    for key, value in facts.items():
        world_state.set_state(key, value)  # type: ignore[arg-type]

    return world_state
