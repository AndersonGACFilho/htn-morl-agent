"""Movement and interaction semantics of the multi-objective scenario."""

from __future__ import annotations

from htn._examples.grid_world.core.geometry import translate
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.movement_profile import (
    InteractionCommand,
    MoveCommand,
)
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState
from htn._examples.grid_world.multi_objective.config import (
    RECHARGE,
    MultiObjectiveGridConfig,
)
from htn._examples.grid_world.multi_objective.state import MultiObjectiveGridState


class MultiObjectiveGridRules(GridRules):
    """Extend the key-door rules with energy, terrain, hazards, and time.

    A move the agent cannot afford is refused, but the clock still advances:
    failing to act takes time, which is what keeps the time objective honest.
    """

    def __init__(self, config: MultiObjectiveGridConfig) -> None:
        """Initialize the rules with the scenario parameters."""
        self._config = config

    def apply(
        self,
        layout: GridLayout,
        state: GridWorldState,
        command: MoveCommand | InteractionCommand,
    ) -> GridWorldState:
        """Return the state produced by executing one command."""
        assert isinstance(state, MultiObjectiveGridState)

        if isinstance(command, MoveCommand):
            advanced = self._move(layout, state, command)
        elif command.name == RECHARGE:
            advanced = self._recharge(state)
        else:
            advanced = super()._interact(layout, state, command)

        assert isinstance(advanced, MultiObjectiveGridState)
        timed = advanced.with_changes(
            elapsed_time=advanced.elapsed_time + self._duration(command)
        )

        return self._evaluate_done(layout, timed)

    def _move(
        self,
        layout: GridLayout,
        state: GridWorldState,
        command: MoveCommand,
    ) -> GridWorldState:
        """Return the state after attempting a move with a movement profile."""
        assert isinstance(state, MultiObjectiveGridState)
        profile = command.profile

        if state.energy < profile.energy_cost:
            return state

        destination = translate(state.agent_position, command.delta)

        if not self.is_passable(layout, state, destination):
            return self._spend(state, profile.energy_cost)

        if (
            destination in self._config.rough_terrain
            and not profile.traverses_rough_terrain
        ):
            return self._spend(state, profile.energy_cost)

        moved = self._spend(state, profile.energy_cost).with_changes(
            agent_position=destination
        )

        if destination in self._config.hazards and not profile.avoids_hazards:
            return self._damage(moved, self._config.hazard_damage)

        return moved

    def _recharge(self, state: MultiObjectiveGridState) -> MultiObjectiveGridState:
        """Recover energy when the agent stands on a recharge tile."""
        if state.agent_position not in self._config.recharge_positions:
            return state

        energy = min(
            state.energy + self._config.recharge_amount, self._config.max_energy
        )

        return state.with_changes(energy=energy)

    def _spend(
        self, state: MultiObjectiveGridState, amount: float
    ) -> MultiObjectiveGridState:
        """Charge energy to both the balance and the cumulative counter."""
        return state.with_changes(
            energy=max(state.energy - amount, 0.0),
            consumed_energy=state.consumed_energy + amount,
        )

    def _damage(
        self, state: MultiObjectiveGridState, amount: float
    ) -> MultiObjectiveGridState:
        """Apply hazard damage without letting health fall below zero."""
        return state.with_changes(health=max(state.health - amount, 0.0))

    def _duration(self, command: MoveCommand | InteractionCommand) -> float:
        """Return the episode time consumed by a command."""
        if isinstance(command, MoveCommand):
            return command.profile.time_cost

        return self._config.movement_profiles[0].time_cost
