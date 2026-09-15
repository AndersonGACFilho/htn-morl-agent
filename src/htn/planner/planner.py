from dataclasses import dataclass

from htn.planner.method_decomposition import MethodDecomposition
from htn.strategy.method_selection_strategy import MethodSelectionStrategy
from htn.tasks.domains.domain import Domain
from htn.tasks.types.compound_task import CompoundTask
from htn.tasks.types.method import Method
from htn.tasks.types.primitive_task import PrimitiveTask
from htn.tasks.types.task import Task
from htn.world.state import WorldState


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """Tasks planned so far, the state they lead to, and how they were decomposed."""

    tasks: list[Task]
    world_state: WorldState
    decompositions: list[MethodDecomposition]


class Planner:
    """Build primitive-task plans by recursively decomposing HTN tasks.

    A :class:`MethodSelectionStrategy` ranks methods after feasibility checks
    and before recursive decomposition. A strategy influences exploration
    order, never the planner's symbolic validity checks or backtracking.
    """

    domain: Domain
    _current_plan: PlanningResult | None
    _strategy: MethodSelectionStrategy
    world_state_copy: WorldState

    def __init__(
        self, domain: Domain, world_state: WorldState, strategy: MethodSelectionStrategy
    ):
        """
        Initialize the planner with a domain, state snapshot, and strategy.

        Args:
            domain: Domain that defines the available root tasks.
            world_state: Initial symbolic state to copy for planning.
            strategy: Policy that orders feasible methods during decomposition.
        """
        self.domain = domain
        self._current_plan = None
        self.world_state_copy = world_state.copy()
        self._strategy = strategy

    def build_plan(self, tasks: list[Task]) -> PlanningResult | None:
        """
        Builds an executable plan for the given task sequence.

        Tasks are planned in order. If a later task cannot currently be
        planned, the successfully planned prefix is returned so execution
        can proceed and planning can be resumed from the updated world state.

        Args:
            tasks: Root tasks to plan in order. The caller owns this sequence.

        Returns:
            The planned prefix with its decomposition, or ``None`` when the
            first task cannot be planned.
        """
        result = PlanningResult(
            tasks=[],
            world_state=self.world_state_copy.copy(),
            decompositions=[],
        )

        for task in tasks:
            extended = self.recursive_planning(result, task)

            if not extended:
                break

            result = extended

        if not result.tasks:
            return None

        self._current_plan = result

        return result

    def recursive_planning(
        self,
        branch: PlanningResult,
        task: Task,
        depth: int = 0,
    ) -> PlanningResult | None:
        """
        Extend a planning branch for one task using simulated state.

        Args:
            branch: Branch planned so far.
            task: Task to decompose or validate.
            depth: Decomposition depth, recorded so the validator can check
                outer methods before the ones nested in them.

        Returns:
            The extended branch, or ``None`` when no valid decomposition exists.
        """
        if isinstance(task, PrimitiveTask):
            return self._plan_primitive_task(branch, task)

        if isinstance(task, CompoundTask):
            return self._plan_compound_task(branch, task, depth)

        return None

    def _plan_primitive_task(
        self,
        branch: PlanningResult,
        task: PrimitiveTask,
    ) -> PlanningResult | None:
        """
        Append a planning task to the current planning result.

        Args:
            branch: Branch planned so far.
            task: Task to decompose or validate.
        Returns:
            The planned task, or ``None`` when no valid decomposition exists.
        """
        if not task.check_preconditions(branch.world_state):
            return None

        planned_world_state = branch.world_state.copy()
        task.apply_effects(planned_world_state)

        return PlanningResult(
            tasks=branch.tasks + [task],
            world_state=planned_world_state,
            decompositions=branch.decompositions,
        )

    def _plan_compound_task(
        self,
        branch: PlanningResult,
        task: CompoundTask,
        depth: int = 0,
    ) -> PlanningResult | None:
        """
        Decompose a compound task based on the current strategy

        Args:
            branch: Branch planned so far.
            task: Task to decompose or validate.
            depth: Decomposition depth, recorded so the validator can check
            outer methods before the ones nested in them.
        Returns:
            The planned task, or ``None`` when no valid decomposition exists.
        """
        start_index = len(branch.tasks)
        feasible_methods = task.get_feasible_methods(branch.world_state)

        ordered_methods = self._strategy.order_methods(
            feasible_methods,
            branch.world_state,
        )
        for method in ordered_methods:
            decomposed = self._decompose_with_method(branch, method, depth)

            if not decomposed:
                continue

            return PlanningResult(
                tasks=decomposed.tasks,
                world_state=decomposed.world_state,
                decompositions=decomposed.decompositions
                + [
                    MethodDecomposition(
                        method=method,
                        compound_task=task,
                        start_index=start_index,
                        end_index=len(branch.tasks),
                        depth=depth,
                    )
                ],
            )

        return None

    def _decompose_with_method(
        self,
        branch: PlanningResult,
        method: Method,
        depth: int = 0,
    ) -> PlanningResult | None:
        """
        Plan every subtask of a method, or fail as a whole.

        Args:
            branch: Branch planned so far.
            method: Method to decompose.
            depth: Decomposition depth, recorded so the validator can check

        Returns:
            The planned task, or ``None`` when no valid decomposition exists.
        """
        decomposed = branch
        for subtask in method.tasks:
            extended = self.recursive_planning(decomposed, subtask, depth + 1)

            if not extended:
                return None

            decomposed = extended

        return decomposed

    def update_world_state(self, world_state: WorldState) -> None:
        """
        Updates the world state of the planner.
        Args:
            world_state: The new world state.
        Returns:
            None
        """
        self.world_state_copy = world_state.copy()
        self._current_plan = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the planner.
        Returns:
            A string representation of the planner.
        """
        return f"Planner(domain={self.domain}, plan={self._current_plan}, world_state_copy={self.world_state_copy})"
