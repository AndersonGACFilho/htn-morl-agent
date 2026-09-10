from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from htn.world import WorldState

T = TypeVar("T", float, np.ndarray)


class RewardFunction(ABC, Generic[T]):
    @abstractmethod
    def calculate(self, current_state: WorldState, next_state: WorldState) -> T:
        """
        Calculates based in different strategies
        Args:
            current_state (WorldState): current state
            next_state (WorldState): next state
        Returns:
            reward (T): reward value
        """
        pass
