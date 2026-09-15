from dataclasses import dataclass

from htn.tasks.types.compound_task import CompoundTask
from htn.tasks.types.method import Method


@dataclass(frozen=True, slots=True)
class MethodDecomposition:
    """Records that a method decomposed a compound task over a plan span.

    Attributes:
        method: The method the planner applied.
        compound_task: The task the method decomposed.
        start_index: First plan position produced by this method.
        end_index: Position after the last one; equals start_index for a
            method that expands to no primitive tasks.
        depth: Recursion depth, used to check outer methods before inner ones.
    """

    method: Method
    compound_task: CompoundTask
    start_index: int
    end_index: int
    depth: int
