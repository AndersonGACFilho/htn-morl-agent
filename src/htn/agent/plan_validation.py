from dataclasses import dataclass
from enum import Enum, auto

from htn.planner.method_decomposition import MethodDecomposition
from htn.tasks.types.compound_task import CompoundTask, Task
from htn.tasks.types.preconditions import are_preconditions_satisfied
from htn.tasks.types.primitive_task import PrimitiveTask
from htn.world import WorldState


class PlanViolationKind(Enum):
    """Kind of the planning problem."""

    INFEASIBLE_TASK = auto()
    UNJUSTIFIED_METHOD = auto()


@dataclass(frozen=True, slots=True)
class PlanViolation:
    """First point at which the remaining plan stopped being valid.

    Attributes:
        kind: Whether a task cannot execute or a method no longer applies.
        plan_index: Position in the remaining plan where validation stopped.
        state_before_violation: Simulated state reaching that position, used
            as the starting point for a partial replan.
        replan_target: Compound task to decompose again, or None when the
            violation is not contained in any compound task.
    """

    kind: PlanViolationKind
    plan_index: int
    state_before_violation: WorldState
    replan_target: CompoundTask | None


@dataclass(frozen=True, slots=True)
class PlanValidation:
    """Outcome of validating a remain plan."""

    violation: PlanViolation | None

    @property
    def is_valid(self) -> bool:
        """Whether the plan is valid."""
        return self.violation is None


def _group_openings_by_index(
    decompositions: list[MethodDecomposition],
) -> dict[int, list[MethodDecomposition]]:
    """Index decompositions by the plan position where they open.

    Args:
        decompositions: Method decompositions, in execution order.
    Returns:
        Decompositions per opening position, outermost first.
    """
    openings: dict[int, list[MethodDecomposition]] = {}
    for decomposition in sorted(decompositions, key=lambda item: item.depth):
        openings.setdefault(decomposition.start_index, []).append(decomposition)

    return openings


def _find_unjustified_method(
    decompositions: list[MethodDecomposition], plan_index: int, world_state: WorldState
) -> PlanViolation | None:
    """Return the first method opening here whose preconditions no longer hold."""
    for decomposition in decompositions:
        if not are_preconditions_satisfied(
            decomposition.method.get_preconditions(),
            world_state,
        ):
            return PlanViolation(
                kind=PlanViolationKind.UNJUSTIFIED_METHOD,
                plan_index=plan_index,
                state_before_violation=world_state.copy(),
                replan_target=decomposition.compound_task,
            )

    return None


def _infeasible_task(
    task: PrimitiveTask, plan_index: int, world_state: WorldState
) -> PlanViolation:
    """Build the violation for a task that can no longer execute."""
    return PlanViolation(
        kind=PlanViolationKind.INFEASIBLE_TASK,
        plan_index=plan_index,
        state_before_violation=world_state.copy(),
        replan_target=task.parent_task,
    )


def _state_after(task: PrimitiveTask, world_state: WorldState) -> WorldState:
    """Return a new state with the task's effects applied."""
    next_state = world_state.copy()
    task.apply_effects(next_state)

    return next_state


def validate_plan(
    plan: list[Task],
    decompositions: list[MethodDecomposition],
    world_state: WorldState,
) -> PlanValidation:
    """Validate a remaining plan and the decomposition that produced it.

    Task preconditions are checked where each task executes, and method
    preconditions where their compound task started decomposing, both against
    a simulated state that carries the effects of the preceding tasks.

    Args:
        plan: Remaining tasks, in execution order.
        decompositions: Methods applied to build the plan, with their spans.
        world_state: State the remaining plan starts from.

    Returns:
        The first violation found, or a valid result.
    """
    openings = _group_openings_by_index(decompositions)
    simulated_state = world_state.copy()

    for index, task in enumerate(plan):
        violation = _find_unjustified_method(
            openings.get(index, []),
            index,
            simulated_state,
        )

        if violation is not None:
            return PlanValidation(violation)

        if not isinstance(task, PrimitiveTask):
            continue

        if not task.check_preconditions(simulated_state):
            return PlanValidation(
                _infeasible_task(
                    task,
                    index,
                    simulated_state,
                )
            )

        simulated_state = _state_after(task, simulated_state)

    trailing = _find_unjustified_method(
        openings.get(len(plan), []),
        len(plan),
        simulated_state,
    )
    if trailing is not None:
        return PlanValidation(trailing)

    return PlanValidation(violation=None)
