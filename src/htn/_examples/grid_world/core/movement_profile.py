"""Movement profiles and the discrete action encoding derived from them.

A movement profile is the seam that separates *how* the agent moves from
*where* it moves. The single-objective example declares one profile, so its
action space matches the original six-action layout. The multi-objective
example declares several profiles with different time and energy costs, which
is what makes time and energy distinguishable objectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from htn._examples.grid_world.core.geometry import DIRECTIONS, Position


@dataclass(frozen=True, slots=True)
class MovementProfile:
    """One way of traversing a tile, with its own time and energy cost.

    Attributes:
        name: Identifier used in facts, logs, and the rendered panels.
        time_cost: Episode time consumed by one step.
        energy_cost: Energy consumed by one step; never negative.
        avoids_hazards: Whether the profile prevents hazard damage.
        traverses_rough_terrain: Whether the profile can enter rough terrain.
    """

    name: str
    time_cost: float
    energy_cost: float
    avoids_hazards: bool = False
    traverses_rough_terrain: bool = False


WALK = MovementProfile(name="walk", time_cost=1.0, energy_cost=1.0)


@dataclass(frozen=True, slots=True)
class MoveCommand:
    """A request to move one tile using a specific movement profile."""

    profile: MovementProfile
    delta: Position


@dataclass(frozen=True, slots=True)
class InteractionCommand:
    """A request to interact with the tile the agent occupies."""

    name: str


class MovementActionCodec:
    """Map movement profiles and interactions onto discrete action ids.

    Move actions occupy the leading ids, ordered by profile and then by
    direction, so a single-profile codec reproduces the conventional
    up/right/down/left encoding. Interactions follow in declaration order.
    """

    def __init__(
        self,
        profiles: Sequence[MovementProfile],
        interactions: Sequence[str],
    ) -> None:
        """Initialize the codec.

        Args:
            profiles: Movement profiles available to the agent.
            interactions: Names of the non-movement actions.

        Raises:
            ValueError: If no movement profile is supplied, or if a profile
                declares a negative cost.
        """
        if not profiles:
            raise ValueError("At least one movement profile is required.")

        for profile in profiles:
            if profile.energy_cost < 0:
                raise ValueError(
                    f"Movement profile {profile.name!r} has negative energy cost."
                )
            if profile.time_cost < 0:
                raise ValueError(
                    f"Movement profile {profile.name!r} has negative time cost."
                )

        self._profiles = tuple(profiles)
        self._interactions = tuple(interactions)

    @property
    def profiles(self) -> tuple[MovementProfile, ...]:
        """Return the movement profiles known to this codec."""
        return self._profiles

    @property
    def action_count(self) -> int:
        """Return the size of the discrete action space."""
        return len(self._profiles) * len(DIRECTIONS) + len(self._interactions)

    def profile_index(self, name: str) -> int:
        """Return the index of a movement profile by name.

        Raises:
            ValueError: If no profile carries that name.
        """
        for index, profile in enumerate(self._profiles):
            if profile.name == name:
                return index
        raise ValueError(f"Unknown movement profile: {name!r}")

    def encode_move(self, profile_index: int, direction_index: int) -> int:
        """Return the action id for a profile and direction pair.

        Raises:
            ValueError: If either index is out of range.
        """
        if not 0 <= profile_index < len(self._profiles):
            raise ValueError(f"Invalid movement profile index: {profile_index}")

        if not 0 <= direction_index < len(DIRECTIONS):
            raise ValueError(f"Invalid direction index: {direction_index}")

        return profile_index * len(DIRECTIONS) + direction_index

    def encode_interaction(self, name: str) -> int:
        """Return the action id for a named interaction.

        Raises:
            ValueError: If the interaction is not declared.
        """
        if name not in self._interactions:
            raise ValueError(f"Unknown interaction: {name!r}")

        return len(self._profiles) * len(DIRECTIONS) + self._interactions.index(name)

    def decode(self, action: int) -> MoveCommand | InteractionCommand:
        """Return the command represented by an action id.

        Raises:
            ValueError: If the action id is outside the action space.
        """
        move_action_count = len(self._profiles) * len(DIRECTIONS)

        if not 0 <= action < self.action_count:
            raise ValueError(f"Invalid action: {action}")

        if action < move_action_count:
            profile_index, direction_index = divmod(action, len(DIRECTIONS))
            return MoveCommand(
                profile=self._profiles[profile_index],
                delta=DIRECTIONS[direction_index],
            )

        return InteractionCommand(name=self._interactions[action - move_action_count])
