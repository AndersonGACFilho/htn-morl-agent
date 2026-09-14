"""Energy-consumption objective."""

from __future__ import annotations

from htn.strategy.reward.objectives.numeric_fact import read_numeric_fact
from htn.strategy.reward.reward_objective import RewardObjective
from htn.world.state import WorldState

CONSUMED_ENERGY_KEY = "energy_consumed"


class EnergyConsumptionObjective(RewardObjective):
    """Penalize energy consumed by a transition, ``r_energy = -c_E``.

    Consumption is not the net energy balance. Recharging raises the available
    energy without undoing what was already spent, so this objective reads a
    cumulative consumption counter. Use
    :class:`~htn.strategy.reward.objectives.net_energy_balance_objective.NetEnergyBalanceObjective`
    when the intended objective really is the balance.
    """

    def __init__(self, consumed_energy_key: str = CONSUMED_ENERGY_KEY) -> None:
        """Initialize the objective with the fact holding cumulative consumption."""
        self._consumed_energy_key = consumed_energy_key

    def calculate(self, current_state: WorldState, next_state: WorldState) -> float:
        """Return the negative energy consumed during the transition.

        Raises:
            ValueError: If the consumption fact is missing, is not numeric, or
                decreased, which would mean recovery was recorded as negative
                consumption.
        """
        before = read_numeric_fact(current_state, self._consumed_energy_key)
        after = read_numeric_fact(next_state, self._consumed_energy_key)

        if after < before:
            raise ValueError(
                f"Cumulative energy consumption decreased from {before} to {after}; "
                "recovery must not be recorded as negative consumption."
            )

        return -(after - before)
