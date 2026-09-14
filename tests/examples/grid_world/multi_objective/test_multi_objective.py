"""Tests for the multi-objective rules, risk model, and sensor."""

from __future__ import annotations

import pytest
from htn._examples.grid_world.core.config import GridWorldConfig
from htn._examples.grid_world.core.geometry import RIGHT
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.movement_profile import (
    WALK,
    InteractionCommand,
    MoveCommand,
)
from htn._examples.grid_world.multi_objective.config import (
    CLIMB,
    RECHARGE,
    MultiObjectiveGridConfig,
    Threat,
    validate_multi_objective_config,
)
from htn._examples.grid_world.multi_objective.risk import ThreatRiskModel
from htn._examples.grid_world.multi_objective.rules import MultiObjectiveGridRules
from htn._examples.grid_world.multi_objective.state import MultiObjectiveGridState

GRID = GridWorldConfig(
    width=4,
    height=1,
    start_position=(0, 0),
    key_position=(1, 0),
    door_position=(2, 0),
    goal_position=(3, 0),
    fixed_obstacles=frozenset(),
)


def _config(**overrides: object) -> MultiObjectiveGridConfig:
    """Return a multi-objective configuration with the given overrides."""
    return MultiObjectiveGridConfig(grid=GRID, **overrides)  # type: ignore[arg-type]


def _layout() -> GridLayout:
    """Return the flat four-tile layout used by these tests."""
    return GridLayout(
        width=4,
        height=1,
        start_position=(0, 0),
        key_position=(1, 0),
        door_position=(2, 0),
        goal_position=(3, 0),
        obstacles=frozenset(),
    )


def _state(**overrides: object) -> MultiObjectiveGridState:
    """Return a starting multi-objective state with the given overrides."""
    defaults: dict[str, object] = {
        "agent_position": (0, 0),
        "energy": 10.0,
        "consumed_energy": 0.0,
        "health": 10.0,
        "elapsed_time": 0.0,
    }
    defaults.update(overrides)

    return MultiObjectiveGridState(**defaults)  # type: ignore[arg-type]


def test_moving_charges_the_cumulative_consumption_counter() -> None:
    rules = MultiObjectiveGridRules(_config())

    result = rules.apply(_layout(), _state(), MoveCommand(profile=WALK, delta=RIGHT))

    assert result.consumed_energy == pytest.approx(1.0)


def test_moving_advances_the_episode_time_by_the_profile_duration() -> None:
    rules = MultiObjectiveGridRules(_config())

    result = rules.apply(_layout(), _state(), MoveCommand(profile=CLIMB, delta=RIGHT))

    assert result.elapsed_time == pytest.approx(2.0)


def test_a_move_the_agent_cannot_afford_leaves_the_position_unchanged() -> None:
    rules = MultiObjectiveGridRules(_config())

    result = rules.apply(
        _layout(), _state(energy=0.0), MoveCommand(profile=WALK, delta=RIGHT)
    )

    assert result.agent_position == (0, 0)


def test_rough_terrain_blocks_a_profile_that_cannot_traverse_it() -> None:
    rules = MultiObjectiveGridRules(_config(rough_terrain=frozenset({(1, 0)})))

    result = rules.apply(_layout(), _state(), MoveCommand(profile=WALK, delta=RIGHT))

    assert result.agent_position == (0, 0)


def test_climbing_enters_rough_terrain() -> None:
    rules = MultiObjectiveGridRules(_config(rough_terrain=frozenset({(1, 0)})))

    result = rules.apply(_layout(), _state(), MoveCommand(profile=CLIMB, delta=RIGHT))

    assert result.agent_position == (1, 0)


def test_entering_a_hazard_costs_health() -> None:
    rules = MultiObjectiveGridRules(
        _config(hazards=frozenset({(1, 0)}), hazard_damage=3.0)
    )

    result = rules.apply(_layout(), _state(), MoveCommand(profile=WALK, delta=RIGHT))

    assert result.health == pytest.approx(7.0)


def test_recharging_does_not_reduce_cumulative_consumption() -> None:
    rules = MultiObjectiveGridRules(
        _config(recharge_positions=frozenset({(0, 0)}), recharge_amount=5.0)
    )

    result = rules.apply(
        _layout(),
        _state(energy=2.0, consumed_energy=8.0),
        InteractionCommand(name=RECHARGE),
    )

    assert result.consumed_energy == pytest.approx(8.0)


def test_recharging_restores_available_energy() -> None:
    rules = MultiObjectiveGridRules(
        _config(recharge_positions=frozenset({(0, 0)}), recharge_amount=5.0)
    )

    result = rules.apply(
        _layout(), _state(energy=2.0), InteractionCommand(name=RECHARGE)
    )

    assert result.energy == pytest.approx(7.0)


def test_risk_model_returns_hazard_risk_without_perceived_threats() -> None:
    model = ThreatRiskModel(
        threats=(),
        decay=0.5,
        vulnerability_weight=1.0,
        max_health=10.0,
        hazards=frozenset({(1, 0)}),
        static_hazard_risk=1.5,
    )

    assert model.risk_at((1, 0), health=10.0) == pytest.approx(1.5)


def test_risk_model_increases_risk_as_health_decreases() -> None:
    model = ThreatRiskModel(
        threats=(Threat(position=(0, 0), threat_weight=1.0),),
        decay=0.5,
        vulnerability_weight=1.0,
        max_health=10.0,
    )

    healthy = model.risk_at((0, 0), health=10.0)
    wounded = model.risk_at((0, 0), health=2.0)

    assert wounded > healthy


def test_risk_model_decays_with_distance() -> None:
    model = ThreatRiskModel(
        threats=(Threat(position=(0, 0), threat_weight=1.0),),
        decay=0.5,
        vulnerability_weight=0.0,
        max_health=10.0,
    )

    assert model.risk_at((3, 0), health=10.0) < model.risk_at((1, 0), health=10.0)


def test_validation_rejects_a_non_positive_risk_decay() -> None:
    with pytest.raises(ValueError, match="risk_decay"):
        validate_multi_objective_config(_config(risk_decay=0.0))


def test_validation_rejects_a_negative_threat_weight() -> None:
    config = _config(threats=(Threat(position=(0, 0), threat_weight=-1.0),))

    with pytest.raises(ValueError, match="negative weight"):
        validate_multi_objective_config(config)
