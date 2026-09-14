"""Net energy-balance objective."""

from __future__ import annotations

from htn.strategy.reward.objectives.numeric_fact import read_numeric_fact
from htn.strategy.reward.reward_objective import RewardObjective
from htn.world.state import WorldState

ENERGY_KEY = "energy"


class NetEnergyBalanceObjective(RewardObjective):
    """Reward the net change in available energy, ``E_{t+1} - E_t``.

    This is the explicitly named alternative to penalizing consumption: spending
    and recovering the same amount nets to zero here, which is the right
    semantics only when the objective really is the balance.
    """

    def __init__(self, energy_key: str = ENERGY_KEY) -> None:
        """Initialize the objective with the fact holding available energy."""
        self._energy_key = energy_key

    def calculate(self, current_state: WorldState, next_state: WorldState) -> float:
        """Return the net change in available energy.

        Raises:
            ValueError: If the energy fact is missing or is not numeric.
        """
        before = read_numeric_fact(current_state, self._energy_key)
        after = read_numeric_fact(next_state, self._energy_key)

        return after - before
