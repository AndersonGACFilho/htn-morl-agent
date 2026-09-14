"""Configuration of the multi-objective key-door scenario."""

from __future__ import annotations

from dataclasses import dataclass, field

from htn._examples.grid_world.core.config import (
    GridWorldConfig,
    validate_grid_world_config,
)
from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.movement_profile import WALK, MovementProfile

RUN = MovementProfile(name="run", time_cost=0.5, energy_cost=3.0)
SAFE_MOVE = MovementProfile(
    name="safe_move", time_cost=2.0, energy_cost=1.0, avoids_hazards=True
)
CLIMB = MovementProfile(
    name="climb", time_cost=2.0, energy_cost=4.0, traverses_rough_terrain=True
)

MOVEMENT_PROFILES: tuple[MovementProfile, ...] = (WALK, RUN, SAFE_MOVE, CLIMB)
RECHARGE = "recharge"


@dataclass(frozen=True, slots=True)
class Threat:
    """A perceived threat contributing to the risk field.

    Attributes:
        position: Where the threat is perceived.
        threat_weight: Non-negative weight of this threat type.
    """

    position: Position
    threat_weight: float


@dataclass(frozen=True, slots=True)
class MultiObjectiveGridConfig:
    """Scenario parameters that make time, energy, and safety distinguishable.

    The grid configuration is composed rather than inherited so each half keeps
    its own validation.

    Attributes:
        grid: Placement and dimensions of the underlying grid.
        movement_profiles: Ways the agent may traverse a tile.
        initial_energy: Energy available at the start of an episode.
        max_energy: Upper bound on available energy.
        initial_health: Health at the start of an episode.
        max_health: Upper bound on health; must be positive.
        recharge_positions: Tiles where the agent can recover energy.
        recharge_amount: Energy recovered by one recharge.
        rough_terrain: Tiles only a climbing profile may enter.
        hazards: Tiles that damage an unprotected agent.
        hazard_damage: Health lost when entering a hazard unprotected.
        threats: Perceived threats forming the risk field.
        risk_decay: Decay of threat influence with distance; must be positive.
        vulnerability_weight: How much low health amplifies risk.
        static_hazard_risk: Risk contributed by a hazard tile itself.
        low_energy_threshold: Below this, the agent considers energy low.
        risk_threshold: Above this, the agent considers a route dangerous.
    """

    grid: GridWorldConfig
    movement_profiles: tuple[MovementProfile, ...] = MOVEMENT_PROFILES

    initial_energy: float = 40.0
    max_energy: float = 40.0
    initial_health: float = 10.0
    max_health: float = 10.0

    recharge_positions: frozenset[Position] = field(default_factory=frozenset)
    recharge_amount: float = 15.0

    rough_terrain: frozenset[Position] = field(default_factory=frozenset)
    hazards: frozenset[Position] = field(default_factory=frozenset)
    hazard_damage: float = 2.0

    threats: tuple[Threat, ...] = ()
    risk_decay: float = 0.6
    vulnerability_weight: float = 1.0
    static_hazard_risk: float = 1.5

    low_energy_threshold: float = 12.0
    risk_threshold: float = 0.35


def validate_multi_objective_config(config: MultiObjectiveGridConfig) -> None:
    """Validate the multi-objective parameters and the grid they extend.

    Raises:
        ValueError: If any documented parameter constraint is violated.
    """
    validate_grid_world_config(config.grid)

    if config.risk_decay <= 0:
        raise ValueError("risk_decay must be greater than zero.")

    if config.vulnerability_weight < 0:
        raise ValueError("vulnerability_weight cannot be negative.")

    if config.max_health <= 0:
        raise ValueError("max_health must be greater than zero.")

    if config.max_energy <= 0:
        raise ValueError("max_energy must be greater than zero.")

    if config.hazard_damage < 0:
        raise ValueError("hazard_damage cannot be negative.")

    if config.static_hazard_risk < 0:
        raise ValueError("static_hazard_risk cannot be negative.")

    for threat in config.threats:
        if threat.threat_weight < 0:
            raise ValueError(f"Threat at {threat.position} has a negative weight.")
