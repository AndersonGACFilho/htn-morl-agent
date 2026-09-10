from abc import ABC, abstractmethod

from htn.world import WorldState


class RewardObjective(ABC):
    """
    An abstract class representing a reward objective.
    """

    @abstractmethod
    def calculate(
        self,
        current_state: WorldState,
        next_state: WorldState,
    ) -> float:
        """
        Calculate the reward objective.

        Args:
            current_state: The current state.
            next_state: The next state resulting from an action or transition.
        Returns:
            the reward value
        """
        pass
