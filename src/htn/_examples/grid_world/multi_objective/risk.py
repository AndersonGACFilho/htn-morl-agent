"""Perceived-risk model of the multi-objective scenario."""

from __future__ import annotations

import math
from dataclasses import dataclass

from htn._examples.grid_world.core.geometry import Position, manhattan_distance
from htn._examples.grid_world.multi_objective.config import Threat


@dataclass(frozen=True, slots=True)
class ThreatRiskModel:
    """Score the risk perceived at a tile.

    The model implements

    ``Risk(S) = (sum_e w_e * exp(-lambda * d_e)) * [1 + alpha_v * (1 - h/h_max)]``

    and then adds the contribution of static hazards. Adding rather than
    multiplying matters: with no perceived threat the product would be zero,
    which reads as "safe", whereas an empty threat set only means nothing was
    *perceived*. A hazard tile stays costly either way.
    """

    threats: tuple[Threat, ...]
    decay: float
    vulnerability_weight: float
    max_health: float
    hazards: frozenset[Position] = frozenset()
    static_hazard_risk: float = 0.0

    def risk_at(self, position: Position, health: float) -> float:
        """Return the perceived risk at a tile for an agent of given health.

        Args:
            position: Tile being evaluated.
            health: Current agent health.

        Returns:
            A non-negative risk score.
        """
        perceived = sum(
            threat.threat_weight
            * math.exp(-self.decay * manhattan_distance(position, threat.position))
            for threat in self.threats
        )

        risk = perceived * self._vulnerability(health)

        if position in self.hazards:
            risk += self.static_hazard_risk

        return risk

    def _vulnerability(self, health: float) -> float:
        """Return the factor by which low health amplifies perceived risk."""
        bounded = min(max(health, 0.0), self.max_health)

        return 1.0 + self.vulnerability_weight * (1.0 - bounded / self.max_health)


class HealthAwareRiskField:
    """Adapt the risk model to the renderer's risk-overlay contract.

    The overlay needs a value per tile at the agent's current health, and it
    expects a normalized value so the heat map stays comparable across frames.
    """

    def __init__(self, model: ThreatRiskModel, health: float, scale: float) -> None:
        """Initialize the field.

        Raises:
            ValueError: If the normalization scale is not positive.
        """
        if scale <= 0:
            raise ValueError("Risk normalization scale must be greater than zero.")

        self._model = model
        self._health = health
        self._scale = scale

    def risk_at(self, position: Position) -> float:
        """Return the normalized risk of a tile, clamped to ``[0, 1]``."""
        risk = self._model.risk_at(position, self._health)

        return min(risk / self._scale, 1.0)
