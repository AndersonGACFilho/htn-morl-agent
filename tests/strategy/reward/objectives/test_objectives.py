"""Tests for the concrete reward objectives."""

from __future__ import annotations

import pytest
from htn.strategy.reward.functions import MultiObjectiveRewardFunction
from htn.strategy.reward.objectives import (
    EnergyConsumptionObjective,
    NetEnergyBalanceObjective,
    RiskExposureObjective,
    TimeObjective,
)

from tests.conftest import build_world_state


def test_time_objective_returns_the_negative_elapsed_time_delta() -> None:
    objective = TimeObjective()

    reward = objective.calculate(
        build_world_state(elapsed_time=4.0), build_world_state(elapsed_time=6.0)
    )

    assert reward == pytest.approx(-2.0)


def test_time_objective_raises_when_the_clock_goes_backwards() -> None:
    objective = TimeObjective()

    with pytest.raises(ValueError, match="monotonic"):
        objective.calculate(
            build_world_state(elapsed_time=6.0), build_world_state(elapsed_time=4.0)
        )


def test_time_objective_raises_when_the_fact_is_missing() -> None:
    objective = TimeObjective()

    with pytest.raises(ValueError, match="missing fact"):
        objective.calculate(build_world_state(), build_world_state(elapsed_time=1.0))


def test_energy_consumption_objective_penalizes_energy_spent() -> None:
    objective = EnergyConsumptionObjective()

    reward = objective.calculate(
        build_world_state(energy_consumed=5.0), build_world_state(energy_consumed=8.0)
    )

    assert reward == pytest.approx(-3.0)


def test_energy_consumption_objective_ignores_energy_recovery() -> None:
    objective = EnergyConsumptionObjective()

    reward = objective.calculate(
        build_world_state(energy_consumed=9.0, energy=2.0),
        build_world_state(energy_consumed=9.0, energy=22.0),
    )

    assert reward == pytest.approx(0.0)


def test_energy_consumption_objective_raises_when_consumption_decreases() -> None:
    objective = EnergyConsumptionObjective()

    with pytest.raises(ValueError, match="decreased"):
        objective.calculate(
            build_world_state(energy_consumed=9.0),
            build_world_state(energy_consumed=4.0),
        )


def test_net_energy_balance_objective_rewards_recovery() -> None:
    objective = NetEnergyBalanceObjective()

    reward = objective.calculate(
        build_world_state(energy=2.0), build_world_state(energy=7.0)
    )

    assert reward == pytest.approx(5.0)


def test_risk_exposure_objective_scales_risk_by_step_duration() -> None:
    objective = RiskExposureObjective()

    reward = objective.calculate(
        build_world_state(risk=0.5, elapsed_time=0.0),
        build_world_state(risk=0.5, elapsed_time=2.0),
    )

    assert reward == pytest.approx(-1.0)


def test_risk_exposure_objective_uses_endpoint_risk_for_unit_steps() -> None:
    objective = RiskExposureObjective(scale_by_duration=False)

    reward = objective.calculate(
        build_world_state(risk=0.5), build_world_state(risk=0.8)
    )

    assert reward == pytest.approx(-0.8)


def test_risk_exposure_objective_still_penalizes_unchanged_danger() -> None:
    objective = RiskExposureObjective(scale_by_duration=False)

    reward = objective.calculate(
        build_world_state(risk=0.7), build_world_state(risk=0.7)
    )

    assert reward == pytest.approx(-0.7)


def test_multi_objective_reward_function_returns_time_energy_safety_in_order() -> None:
    function = MultiObjectiveRewardFunction(
        TimeObjective(), EnergyConsumptionObjective(), RiskExposureObjective()
    )

    reward = function.calculate(
        build_world_state(elapsed_time=0.0, energy_consumed=0.0, risk=0.0),
        build_world_state(elapsed_time=2.0, energy_consumed=3.0, risk=0.5),
    )

    assert list(reward) == pytest.approx([-2.0, -3.0, -1.0])
