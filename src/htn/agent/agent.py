from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from htn.actions.action_status import ActionStatus
from htn.agent.agent_base import AgentBase
from htn.agent.executable_plan import ExecutablePlan
from htn.agent.plan_validation import PlanViolationKind, validate_plan
from htn.planner.planner import Planner
from htn.tasks.types.primitive_task import PrimitiveTask
from htn.tasks.types.task import Task
from htn.world.state import WorldState

if TYPE_CHECKING:
    from htn.world.world import World


@dataclass(frozen=True, slots=True)
class AgentTickResult:
    """
    Result produced by one agent execution tick.

    Attributes:
        task_name: Name of the task executed in this tick, if any.
        status: Status returned by the executed action.
        replanned: Whether the agent built a new plan before executing.
        planned_tasks: Names of the tasks produced by the new plan.
        remaining_plan: Names of the tasks still pending after this tick.
        message: Optional execution message.
    """

    task_name: str | None
    status: ActionStatus | None
    replanned: bool
    planned_tasks: list[str]
    remaining_plan: list[str]
    message: str | None = None


class Agent(AgentBase):
    """
    Agent that owns, validates, replans, and executes an HTN plan.

    The planner only builds plans. The agent owns an independent copy of the
    ordered root-task sequence and passes it to the planner whenever it needs
    a new plan.
    """

    planner: Planner
    world_state: WorldState
    _plan: ExecutablePlan
    tasks: list[Task]

    @property
    def plan(self) -> list[Task]:
        """Tasks of the current plan that have not executed yet."""
        return self._plan.remaining_tasks

    def __init__(
        self, planner: Planner, world_state: WorldState, tasks: list[Task]
    ) -> None:
        """
        Initialize the agent.

        Args:
            planner: Planner used to build symbolic plans.
            world_state: Initial symbolic world state observed by the agent.
            tasks: Ordered root tasks used for every replanning attempt. The
                agent copies this list, so later caller-side changes do not
                alter its planning objective.
        """
        self.planner = planner
        self.world_state = world_state.copy()
        self.tasks = tasks.copy()
        self._plan = ExecutablePlan.empty()

    def tick(self, world: World) -> AgentTickResult:
        """
        Execute one agent tick.

        The agent:
        1. replans if there is no plan or the current plan became invalid;
        2. executes the current primitive task;
        3. keeps running tasks in the plan;
        4. removes successful tasks from the plan;
        5. clears the plan on failure.

        Args:
            world: Runtime world where actions are executed.
        Returns:
            Execution result for this tick.
        """
        replanned = False
        planned_tasks: list[str] = []

        if self._should_replan():
            planning_result = self.planner.build_plan(self.tasks)
            replanned = True

            if planning_result is None:
                self._plan = ExecutablePlan.empty()
                return AgentTickResult(
                    task_name=None,
                    status=None,
                    replanned=True,
                    planned_tasks=[],
                    remaining_plan=[],
                    message="HTN: No valid plan.",
                )

            self._plan = ExecutablePlan(
                tasks=planning_result.tasks,
                decompositions=planning_result.decompositions,
            )
            planned_tasks = self.get_plan_names()

        current_task = self.plan[0]

        if not isinstance(current_task, PrimitiveTask):
            self._plan = self._plan.advance()

            return AgentTickResult(
                task_name=current_task.name,
                status=None,
                replanned=replanned,
                planned_tasks=planned_tasks,
                remaining_plan=self.get_plan_names(),
                message=f"Skipped non-primitive task: {current_task.name}",
            )

        status = current_task.action.execute(world)

        if status == ActionStatus.SUCCESS:
            self._plan = self._plan.advance()

        elif status == ActionStatus.FAILURE:
            self._plan = ExecutablePlan.empty()

        elif status == ActionStatus.RUNNING:
            pass

        else:
            raise ValueError(f"Unknown action status: {status}")

        return AgentTickResult(
            task_name=current_task.name,
            status=status,
            replanned=replanned,
            planned_tasks=planned_tasks,
            remaining_plan=self.get_plan_names(),
        )

    def handle_world_state_change(self, world_state: WorldState) -> None:
        """
        Handle a sensor/world-state update.

        The agent receives the updated symbolic state and forwards it to the
        planner. The plan is not immediately discarded; it is validated on the
        next tick, allowing still-valid plans to continue.

        Args:
            world_state: Updated symbolic world state.
        Returns:
            None
        """
        self.world_state = world_state.copy()
        self.planner.update_world_state(world_state)

    def get_plan_names(self) -> list[str]:
        """
        Return the names of the current remaining plan tasks.

        Returns:
            List of task names.
        """
        return [task.name for task in self.plan]

    def _should_replan(self) -> bool:
        """
        Decide whether the agent should request a new plan.

        Returns:
            True when the agent has no plan or the current plan became invalid after a world-state change.
        """
        if not self.plan:
            return True

        validation = validate_plan(
            self.plan,
            self._plan.remaining_decompositions,
            self.world_state,
        )
        violation = validation.violation

        if violation is None:
            return False

        if violation.kind is PlanViolationKind.INFEASIBLE_TASK:
            return True

        return True
