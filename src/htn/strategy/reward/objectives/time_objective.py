"""Elapsed-time objective."""

from __future__ import annotations

from htn.strategy.reward.objectives.numeric_fact import read_numeric_fact
from htn.strategy.reward.reward_objective import RewardObjective
from htn.world.state import WorldState

ELAPSED_TIME_KEY = "elapsed_time"


class TimeObjective(RewardObjective):
    """Penalize the time a transition consumes, ``r_time = -delta t``.

    Elapsed time is read from the state rather than counted in steps, because
    movement profiles have different durations: an action count is not a
    duration once running and walking cost differently.
    """

    def __init__(self, elapsed_time_key: str = ELAPSED_TIME_KEY) -> None:
        """Initialize the objective with the fact holding elapsed time."""
        self._elapsed_time_key = elapsed_time_key

    def calculate(self, current_state: WorldState, next_state: WorldState) -> float:
        """Return the negative elapsed time of the transition.

        Raises:
            ValueError: If the elapsed-time fact is missing, is not numeric, or
                decreased during the transition.
        """
        before = read_numeric_fact(current_state, self._elapsed_time_key)
        after = read_numeric_fact(next_state, self._elapsed_time_key)

        if after < before:
            raise ValueError(
                f"Elapsed time decreased from {before} to {after}; "
                "the episode clock must be monotonic."
            )

        return -(after - before)
