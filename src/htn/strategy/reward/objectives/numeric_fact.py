"""Shared reading of numeric facts from a symbolic state.

Objectives raise when a fact they need is absent rather than defaulting to
zero. A silently zero objective — safety above all — would be indistinguishable
from a genuinely safe transition and would quietly corrupt every comparison
built on it.
"""

from __future__ import annotations

from htn.utils import is_number
from htn.world.state import WorldState


def read_numeric_fact(state: WorldState, key: str) -> float:
    """Return a numeric fact from a symbolic state.

    Args:
        state: State to read from.
        key: Name of the fact.

    Returns:
        The fact value as a float.

    Raises:
        ValueError: If the fact is missing or is not numeric.
    """
    if key not in state.state_space:
        raise ValueError(f"Reward objective requires the missing fact {key!r}.")

    value = state.state_space[key]

    if not is_number(value):
        raise ValueError(f"Reward objective requires {key!r} to be numeric.")

    return float(value)
