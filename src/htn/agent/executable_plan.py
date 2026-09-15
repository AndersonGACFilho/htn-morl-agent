from dataclasses import dataclass, replace

from htn.planner.method_decomposition import MethodDecomposition
from htn.tasks.types.task import Task


@dataclass(frozen=True, slots=True)
class ExecutablePlan:
    """A planned task sequence together with the decomposition that produced it.

    Executed tasks are tracked by a cursor instead of being removed, so the
    recorded decomposition spans stay aligned with the task list.
    """

    tasks: list[Task]
    decompositions: list[MethodDecomposition]
    executed_count: int = 0

    @classmethod
    def empty(cls) -> "ExecutablePlan":
        """
        Return an empty plan.
        """
        return cls(tasks=[], decompositions=[], executed_count=0)

    @property
    def remaining_tasks(self) -> list[Task]:
        """Tasks that have not yet been executed."""
        return self.tasks[self.executed_count :]

    @property
    def remaining_decompositions(self) -> list[MethodDecomposition]:
        """Decompositions still open, with spans rebased on the remaining tasks."""
        return [
            replace(
                decomposition,
                start_index=max(decomposition.start_index - self.executed_count, 0),
                end_index=decomposition.end_index - self.executed_count,
            )
            for decomposition in self.decompositions
            if decomposition.end_index > self.executed_count
        ]

    def advance(self) -> "ExecutablePlan":
        """Return the plan with the current task marked as executed."""
        return replace(self, executed_count=self.executed_count + 1)
