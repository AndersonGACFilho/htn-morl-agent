"""Reward vector of the multi-objective scenario."""

from __future__ import annotations

from htn.strategy.reward.functions import MultiObjectiveRewardFunction
from htn.strategy.reward.objectives import (
    EnergyConsumptionObjective,
    RiskExposureObjective,
    TimeObjective,
)


def build_reward_function() -> MultiObjectiveRewardFunction:
    """Build the ``[time, energy, safety]`` reward function.

    The component order is fixed and shared with the weights, the logs, and the
    rendered panel, so a vector read anywhere in the system means the same
    thing. The function reports one transition only: accumulating a return,
    discounting it, and applying preferences belong to the layer above.
    """
    return MultiObjectiveRewardFunction(
        TimeObjective(),
        EnergyConsumptionObjective(),
        RiskExposureObjective(),
    )
