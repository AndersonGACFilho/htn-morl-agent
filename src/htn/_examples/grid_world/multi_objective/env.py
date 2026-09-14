"""Gymnasium environment of the multi-objective scenario."""

from __future__ import annotations

from typing import Any

from htn._examples.grid_world.core.env import GridWorldEnv
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.movement_profile import MovementActionCodec
from htn._examples.grid_world.core.rules import OPEN_DOOR, PICKUP_KEY
from htn._examples.grid_world.core.state import GridWorldState
from htn._examples.grid_world.multi_objective.config import (
    RECHARGE,
    MultiObjectiveGridConfig,
    validate_multi_objective_config,
)
from htn._examples.grid_world.multi_objective.risk import ThreatRiskModel
from htn._examples.grid_world.multi_objective.rules import MultiObjectiveGridRules
from htn._examples.grid_world.multi_objective.state import MultiObjectiveGridState


class MultiObjectiveGridWorldEnv(GridWorldEnv):
    """Key-door GridWorld carrying energy, health, and a risk field.

    The environment reuses the key-door layout and terminal condition; what it
    adds is the state the objectives measure.
    """

    def __init__(self, config: MultiObjectiveGridConfig) -> None:
        """Initialize the environment.

        Raises:
            ValueError: If the configuration is inconsistent.
        """
        validate_multi_objective_config(config)

        self.multi_objective_config = config
        self.risk_model = ThreatRiskModel(
            threats=config.threats,
            decay=config.risk_decay,
            vulnerability_weight=config.vulnerability_weight,
            max_health=config.max_health,
            hazards=config.hazards,
            static_hazard_risk=config.static_hazard_risk,
        )

        super().__init__(
            config.grid,
            rules=MultiObjectiveGridRules(config),
            codec=MovementActionCodec(
                profiles=config.movement_profiles,
                interactions=(PICKUP_KEY, OPEN_DOOR, RECHARGE),
            ),
        )

    @property
    def multi_objective_state(self) -> MultiObjectiveGridState:
        """Return the current state with its multi-objective quantities."""
        state = self.state
        assert isinstance(state, MultiObjectiveGridState)

        return state

    def current_risk(self) -> float:
        """Return the risk perceived at the agent's current position."""
        state = self.multi_objective_state

        return self.risk_model.risk_at(state.agent_position, state.health)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve a new layout and restart the episode."""
        return super().reset(seed=seed, options=options)

    def _initial_state(self, layout: GridLayout) -> GridWorldState:
        """Build the starting state, including energy, health, and the clock."""
        config = self.multi_objective_config

        state = MultiObjectiveGridState(
            agent_position=layout.start_position,
            has_key=config.grid.initial_has_key,
            door_open=config.grid.initial_door_open,
            energy=config.initial_energy,
            consumed_energy=0.0,
            health=config.initial_health,
            elapsed_time=0.0,
        )

        return state.with_changes(
            done=state.agent_position == layout.goal_position and state.door_open
        )
