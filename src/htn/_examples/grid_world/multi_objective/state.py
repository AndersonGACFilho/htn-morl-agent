"""Episode state of the multi-objective scenario."""

from __future__ import annotations

from dataclasses import dataclass

from htn._examples.grid_world.core.state import GridWorldState


@dataclass(frozen=True, slots=True)
class MultiObjectiveGridState(GridWorldState):
    """Key-door state extended with the quantities the objectives measure.

    ``consumed_energy`` is cumulative and never decreases. Recharging raises
    ``energy`` without lowering it, so recovery can never be mistaken for
    negative consumption — the consumption-versus-balance distinction is
    enforced by the data model rather than by convention.

    Attributes:
        energy: Energy currently available.
        consumed_energy: Total energy spent so far this episode.
        health: Current health.
        elapsed_time: Episode time consumed so far.
    """

    energy: float = 0.0
    consumed_energy: float = 0.0
    health: float = 0.0
    elapsed_time: float = 0.0
