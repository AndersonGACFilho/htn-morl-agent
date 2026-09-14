"""Risk-exposure objective."""

from __future__ import annotations

from htn.strategy.reward.objectives.numeric_fact import read_numeric_fact
from htn.strategy.reward.reward_objective import RewardObjective
from htn.world.state import WorldState

RISK_KEY = "risk"
ELAPSED_TIME_KEY = "elapsed_time"


class RiskExposureObjective(RewardObjective):
    """Penalize exposure to risk at the end of a transition.

    For steps of equal duration the reward is ``-Risk(S')``. When durations
    differ, exposure is approximated by ``-Risk(S') * delta t`` from the
    endpoint risk, so standing in danger twice as long costs twice as much.

    Endpoint risk, rather than the change in risk, is what keeps constant
    danger costly: ``-delta Risk`` would score an unchanged threat as free.
    """

    def __init__(
        self,
        risk_key: str = RISK_KEY,
        elapsed_time_key: str = ELAPSED_TIME_KEY,
        *,
        scale_by_duration: bool = True,
    ) -> None:
        """Initialize the objective.

        Args:
            risk_key: Fact holding the perceived risk of a state.
            elapsed_time_key: Fact holding the episode clock.
            scale_by_duration: Whether to weight risk by the step duration.
        """
        self._risk_key = risk_key
        self._elapsed_time_key = elapsed_time_key
        self._scale_by_duration = scale_by_duration

    def calculate(self, current_state: WorldState, next_state: WorldState) -> float:
        """Return the negative risk exposure of the transition.

        Raises:
            ValueError: If a required fact is missing or not numeric, or if the
                elapsed time decreased.
        """
        risk = read_numeric_fact(next_state, self._risk_key)

        if not self._scale_by_duration:
            return -risk

        before = read_numeric_fact(current_state, self._elapsed_time_key)
        after = read_numeric_fact(next_state, self._elapsed_time_key)

        if after < before:
            raise ValueError(
                f"Elapsed time decreased from {before} to {after}; "
                "the episode clock must be monotonic."
            )

        return -risk * (after - before)
