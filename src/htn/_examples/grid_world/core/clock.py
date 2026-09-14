"""Per-episode time accounting.

Elapsed time is not the action count: movement profiles have different
durations. Each episode owns its clock, and planning advances a copy so a
hypothetical decomposition never moves the live episode forward.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EpisodeClock:
    """Elapsed simulated time owned by a single episode."""

    elapsed_time: float = 0.0

    def advance(self, duration: float) -> None:
        """Advance the clock by a non-negative duration.

        Raises:
            ValueError: If the duration is negative.
        """
        if duration < 0:
            raise ValueError("Duration cannot be negative.")

        self.elapsed_time += duration

    def reset(self) -> None:
        """Return the clock to the start of an episode."""
        self.elapsed_time = 0.0

    def copy(self) -> "EpisodeClock":
        """Return an independent clock for hypothetical planning."""
        return EpisodeClock(elapsed_time=self.elapsed_time)
