"""Construction of the HTN decomposition view.

Method applicability is evaluated against the *observed* world state, not the
hypothetical planning state that the planner holds at each node. The planner
filters methods before handing them to a selection strategy, so a rejected
method is never observable from outside; reporting applicability at the
observed state is the closest faithful approximation without a planner hook.
Panels must label the result accordingly.
"""

from __future__ import annotations

from typing import Sequence

from htn._examples.grid_world.core.view.models import (
    DecompositionView,
    MethodView,
    TaskNodeView,
)
from htn.tasks.domains.domain import Domain
from htn.tasks.types.compound_task import CompoundTask
from htn.tasks.types.method import Method
from htn.tasks.types.task import Task
from htn.utils import check_condition
from htn.world.state import WorldState

MAX_DEPTH = 6


class DecompositionViewBuilder:
    """Flatten a domain into an annotated tree for the current state."""

    def __init__(self, domain: Domain) -> None:
        """Initialize the builder for a fixed domain."""
        self._domain = domain

    def build(
        self,
        world_state: WorldState,
        *,
        current_task: str | None = None,
        remaining_tasks: Sequence[str] = (),
    ) -> DecompositionView:
        """Build the decomposition view for the observed state.

        Args:
            world_state: State used to evaluate method preconditions.
            current_task: Task executing this tick, highlighted in the tree.
            remaining_tasks: Tasks still queued, used to mark pending leaves.

        Returns:
            The annotated decomposition of the domain.
        """
        pending = frozenset(remaining_tasks)
        nodes: list[TaskNodeView] = []

        for task in self._domain.tasks:
            self._collect(task, world_state, current_task, pending, 0, nodes)

        return DecompositionView(nodes=tuple(nodes))

    def _collect(
        self,
        task: Task,
        world_state: WorldState,
        current_task: str | None,
        pending: frozenset[str],
        depth: int,
        nodes: list[TaskNodeView],
    ) -> None:
        """Append ``task`` and its descendants to ``nodes`` in display order."""
        if depth > MAX_DEPTH:
            return

        if not isinstance(task, CompoundTask):
            nodes.append(
                TaskNodeView(
                    name=task.name,
                    depth=depth,
                    is_compound=False,
                    is_current=task.name == current_task,
                )
            )
            return

        nodes.append(
            TaskNodeView(
                name=task.name,
                depth=depth,
                is_compound=True,
                is_current=task.name == current_task,
                methods=self._methods(task, world_state),
            )
        )

        for method in task.get_methods():
            for subtask in method.get_tasks():
                self._collect(
                    subtask, world_state, current_task, pending, depth + 1, nodes
                )

    def _methods(
        self, task: CompoundTask, world_state: WorldState
    ) -> tuple[MethodView, ...]:
        """Build the method views of a compound task."""
        return tuple(
            self._method_view(method, world_state) for method in task.get_methods()
        )

    def _method_view(self, method: Method, world_state: WorldState) -> MethodView:
        """Build one method view, listing the preconditions that fail."""
        failed = self._failed_preconditions(method, world_state)

        return MethodView(
            id=method.id,
            name=method.name,
            applicable=not failed,
            failed_preconditions=failed,
        )

    def _failed_preconditions(
        self, method: Method, world_state: WorldState
    ) -> tuple[str, ...]:
        """Return the precondition keys that do not hold in the observed state.

        A key absent from the state counts as failed, matching the planner's own
        treatment of missing facts.
        """
        failed: list[str] = []

        for key, (operator, expected) in method.get_preconditions().items():
            if key not in world_state.state_space:
                failed.append(key)
                continue

            if not check_condition(world_state.state_space[key], operator, expected):
                failed.append(key)

        return tuple(failed)
