from htn.strategy.reward.reward_function import RewardFunction
from htn.strategy.reward.reward_objective import RewardObjective
from htn.world import WorldState


class SingleObjectiveRewardFunction(RewardFunction[float]):
    """
    Evaluates a single scalar RewardObjective.

    Example:
        ```python
        reward_fn = SingleObjectiveRewardFunction(TimeReward())
        ```
    """

    objective: RewardObjective

    def __init__(
        self,
        objective: RewardObjective,
    ):
        """
        Constructor.

        Args:
            objective(RewardObjective): reward objective
        """
        self.objective = objective

    def calculate(self, current_state: WorldState, next_state: WorldState) -> float:
        """
        Calculate the reward function.

        Args:
            current_state (WorldState): current state
            next_state (WorldState): next state
        Returns:
            the reward value
        """
        return self.objective.calculate(current_state, next_state)
