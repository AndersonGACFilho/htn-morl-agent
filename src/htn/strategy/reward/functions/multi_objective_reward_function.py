from typing import Sequence

import numpy as np

from htn.strategy.reward.reward_function import RewardFunction
from htn.strategy.reward.reward_objective import RewardObjective
from htn.world import WorldState


class MultiObjectiveRewardFunction(RewardFunction[np.ndarray]):
    """
    Aggregates multiple RewardObjective instances into a 1D vector.

    Example:
        ```python
        reward_fn = MultiObjectiveRewardFunction(
            TimeReward(),
            EnergyReward(),
            SafetyReward(),
        )
        ```
    """

    objectives: Sequence[RewardObjective]

    def __init__(
        self,
        *objectives: RewardObjective,
    ):
        """
        Constructor.

        Args:
            objectives: list of reward objectives
        Raises:
            ValueError: If objectives is empty
        """

        if not objectives:
            raise ValueError(
                "MultiObjectiveRewardFunction requires at least one RewardObjective."
            )
        self.objectives = objectives

    def calculate(
        self, current_state: WorldState, next_state: WorldState
    ) -> np.ndarray:
        """
        Calculate the reward function.

        Args:
            current_state (WorldState): current state
            next_state (WorldState): next state
        Returns:
            np.ndarray: reward values
        """
        return np.array(
            [
                objective.calculate(current_state, next_state)
                for objective in self.objectives
            ],
            dtype=np.float64,
        )
