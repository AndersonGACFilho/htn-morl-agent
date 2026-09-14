"""Symbolic facts published by the multi-objective scenario."""

from __future__ import annotations

from htn._examples.grid_world.core.world import GridWorld
from htn._examples.grid_world.multi_objective.config import MultiObjectiveGridConfig
from htn._examples.grid_world.multi_objective.env import MultiObjectiveGridWorldEnv
from htn._examples.grid_world.single_objective.sensors import KeyDoorSensor
from htn.world.state import WorldState


class MultiObjectiveGridSensor(KeyDoorSensor):
    """Publish the key-door facts plus the quantities the objectives read.

    ``energy_is_low`` and ``route_is_dangerous`` are derived here because HTN
    preconditions compare a fact against a literal and cannot express a
    threshold. The thresholds themselves stay in the scenario configuration.
    """

    def __init__(self, config: MultiObjectiveGridConfig) -> None:
        """Initialize the sensor with the thresholds it derives facts from."""
        self._config = config

    def sense(self, world: GridWorld, world_state: WorldState) -> None:
        """Refresh every fact used by the domain and by the reward objectives."""
        super().sense(world, world_state)

        env = world.env
        assert isinstance(env, MultiObjectiveGridWorldEnv)

        state = env.multi_objective_state
        risk = env.current_risk()

        world_state.set_state("energy", state.energy)
        world_state.set_state("energy_consumed", state.consumed_energy)
        world_state.set_state("health", state.health)
        world_state.set_state("elapsed_time", state.elapsed_time)
        world_state.set_state("risk", risk)

        world_state.set_state(
            "at_recharge", state.agent_position in self._config.recharge_positions
        )
        world_state.set_state(
            "energy_is_low", state.energy < self._config.low_energy_threshold
        )
        world_state.set_state("route_is_dangerous", risk > self._config.risk_threshold)
