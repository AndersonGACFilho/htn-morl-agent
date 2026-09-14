"""Construction of the multi-objective view.

The builder converts arrays into plain tuples so that no panel depends on
numpy, and leaves weights and utility unset unless a caller supplies a
preference vector. There is no learner in this baseline: an unset field is the
honest representation, not a zero.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from htn._examples.grid_world.core.view.models import OBJECTIVE_NAMES, ObjectiveView


class ObjectiveViewBuilder:
    """Assemble the objective panel data for one frame."""

    def __init__(self, names: Sequence[str] = OBJECTIVE_NAMES) -> None:
        """Initialize the builder with the fixed objective order.

        Raises:
            ValueError: If no objective name is supplied.
        """
        if not names:
            raise ValueError("At least one objective name is required.")

        self._names = tuple(names)

    def build(
        self,
        step_reward: np.ndarray | None,
        episode_return: np.ndarray | None,
        *,
        weights: Sequence[float] | None = None,
    ) -> ObjectiveView:
        """Build the objective view.

        Args:
            step_reward: Reward vector of the last transition.
            episode_return: Undiscounted accumulation over the episode.
            weights: Preference vector supplied for display only.

        Returns:
            The objective view, with unset fields where no data exists.

        Raises:
            ValueError: If ``weights`` does not match the objective count.
        """
        if weights is not None and len(weights) != len(self._names):
            raise ValueError(
                f"Expected {len(self._names)} weights, received {len(weights)}."
            )

        weight_values = tuple(float(weight) for weight in weights) if weights else None

        return ObjectiveView(
            names=self._names,
            step_reward=self._to_tuple(step_reward),
            episode_return=self._to_tuple(episode_return),
            weights=weight_values,
            utility=self._utility(episode_return, weight_values),
        )

    def _to_tuple(self, values: np.ndarray | None) -> tuple[float, ...] | None:
        """Convert an array into a tuple of floats, preserving ``None``."""
        if values is None:
            return None

        return tuple(float(value) for value in values)

    def _utility(
        self,
        episode_return: np.ndarray | None,
        weights: tuple[float, ...] | None,
    ) -> float | None:
        """Return the scalarized utility when both inputs are available."""
        if episode_return is None or weights is None:
            return None

        return float(np.dot(np.asarray(weights, dtype=np.float64), episode_return))
