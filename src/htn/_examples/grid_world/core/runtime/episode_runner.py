"""Execution of one GridWorld episode with rendering and optional rewards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from htn._examples.grid_world.core.geometry import Position
from htn._examples.grid_world.core.rendering.export import SvgFrameExporter
from htn._examples.grid_world.core.rendering.renderer import GridWorldRenderer
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.core.runtime.presenter import TickPresenter
from htn._examples.grid_world.core.view.decomposition_view_builder import (
    DecompositionViewBuilder,
)
from htn._examples.grid_world.core.view.grid_view_builder import GridViewBuilder
from htn._examples.grid_world.core.view.models import (
    FrameView,
    MessageRole,
    ObjectiveView,
    PlanEntryStatus,
    PlanEntryView,
    PlanView,
)
from htn._examples.grid_world.core.view.objective_view_builder import (
    ObjectiveViewBuilder,
)
from htn._examples.grid_world.core.world import GridWorld
from htn.actions.action_status import ActionStatus
from htn.agent.agent import Agent, AgentTickResult
from htn.sensors import SensorSystem
from htn.strategy.reward.reward_function import RewardFunction
from htn.world.state import WorldState

DEFAULT_MAX_TICKS = 100


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Outcome of one executed episode.

    Attributes:
        ticks: Number of ticks executed.
        succeeded: Whether the environment reached its terminal condition.
        frames: Paths of the exported frames, in order.
        episode_return: Accumulated reward vector, when rewards were evaluated.
    """

    ticks: int
    succeeded: bool
    frames: tuple[Path, ...]
    episode_return: np.ndarray | None = None


class EpisodeRunner:
    """Drive the tick loop, refresh sensors, render, and accumulate rewards.

    Accumulating the reward vector belongs here rather than in the reward
    function: a reward function reports one transition and must not apply a
    discount, a preference, or a running total.
    """

    def __init__(
        self,
        agent: Agent,
        world: GridWorld,
        world_state: WorldState,
        sensor_system: SensorSystem[GridWorld],
        renderer: GridWorldRenderer,
        grid_view_builder: GridViewBuilder,
        decomposition_view_builder: DecompositionViewBuilder,
        *,
        route_planner: RoutePlanner | None = None,
        exporter: SvgFrameExporter | None = None,
        reward_function: RewardFunction[np.ndarray] | None = None,
        objective_view_builder: ObjectiveViewBuilder | None = None,
        display_weights: Sequence[float] | None = None,
        max_ticks: int = DEFAULT_MAX_TICKS,
    ) -> None:
        """Initialize the runner with everything one episode needs."""
        self._agent = agent
        self._world = world
        self._world_state = world_state
        self._sensor_system = sensor_system
        self._renderer = renderer
        self._grid_view_builder = grid_view_builder
        self._decomposition_view_builder = decomposition_view_builder
        self._route_planner = route_planner
        self._exporter = exporter
        self._reward_function = reward_function
        self._objective_view_builder = objective_view_builder
        self._display_weights = display_weights
        self._max_ticks = max_ticks
        self._presenter = TickPresenter()
        self._plan_tasks: tuple[str, ...] = ()
        self._completed = 0

    def run(self) -> EpisodeResult:
        """Execute the episode until it terminates or the tick budget runs out."""
        frames: list[Path] = []
        episode_return: np.ndarray | None = None
        step_reward: np.ndarray | None = None
        tick = 0

        self._draw(PlanView(tick=tick))
        self._capture(frames, tick)

        while not self._world.done and tick < self._max_ticks:
            tick += 1
            previous_state = self._world_state.copy()

            result = self._agent.tick(self._world)

            if result.replanned and not result.planned_tasks:
                self._renderer.print_message(
                    "HTN: no valid plan.", role=MessageRole.FAILURE
                )
                break

            if result.replanned:
                self._plan_tasks = tuple(result.planned_tasks)
                self._completed = 0

            self._sensor_system.update(self._world, self._world_state)

            if self._reward_function is not None:
                step_reward = self._reward_function.calculate(
                    previous_state, self._world_state
                )
                episode_return = (
                    step_reward.copy()
                    if episode_return is None
                    else episode_return + step_reward
                )

            self._draw(
                self._plan_view(tick, result),
                step_reward=step_reward,
                episode_return=episode_return,
                current_task=result.task_name,
            )
            self._capture(frames, tick)

        return EpisodeResult(
            ticks=tick,
            succeeded=self._world.done,
            frames=tuple(frames),
            episode_return=episode_return,
        )

    def _plan_view(self, tick: int, result: AgentTickResult) -> PlanView:
        """Build the plan view, keeping finished tasks visible as done.

        Progress is tracked as the agent reports it rather than derived from the
        shrinking plan: a succeeded task advances the cursor, while a running or
        failed one leaves it on the task the agent is still attempting.
        """
        if result.status is ActionStatus.SUCCESS:
            self._completed += 1

        entries = tuple(
            PlanEntryView(name=name, status=self._entry_status(index))
            for index, name in enumerate(self._plan_tasks)
        )

        return PlanView(
            tick=tick,
            entries=entries,
            current_role=self._presenter.role_of(result),
            replanned=result.replanned,
        )

    def _entry_status(self, index: int) -> PlanEntryStatus:
        """Return how far the agent has progressed through one plan entry."""
        if index < self._completed:
            return PlanEntryStatus.DONE

        if index == self._completed:
            return PlanEntryStatus.CURRENT

        return PlanEntryStatus.PENDING

    def _draw(
        self,
        plan: PlanView,
        *,
        step_reward: np.ndarray | None = None,
        episode_return: np.ndarray | None = None,
        current_task: str | None = None,
    ) -> None:
        """Build the frame for this tick and render it."""
        env = self._world.env
        target = self._current_target()
        route = self._route(target)

        frame = FrameView(
            grid=self._grid_view_builder.build(
                env.layout, env.state, route=route, target=target
            ),
            plan=plan,
            decomposition=self._decomposition_view_builder.build(
                self._world_state, current_task=current_task
            ),
            objectives=self._objectives(step_reward, episode_return),
        )

        self._renderer.render(frame)

    def _objectives(
        self,
        step_reward: np.ndarray | None,
        episode_return: np.ndarray | None,
    ) -> ObjectiveView | None:
        """Return the objective view when this episode evaluates rewards."""
        if self._objective_view_builder is None:
            return None

        return self._objective_view_builder.build(
            step_reward, episode_return, weights=self._display_weights
        )

    def _current_target(self) -> Position | None:
        """Return the tile the executing navigation action aims at, if any."""
        for task in self._agent.plan:
            action = getattr(task, "action", None)
            target = getattr(action, "target", None)

            if target is not None:
                return target

        return None

    def _route(self, target: Position | None) -> tuple[Position, ...] | list[Position]:
        """Return the planned route to the current target, if computable."""
        if self._route_planner is None or target is None:
            return ()

        env = self._world.env
        return self._route_planner.find_route(env.layout, env.state, target)

    def _capture(self, frames: list[Path], tick: int) -> None:
        """Export the current frame when an exporter is configured."""
        if self._exporter is None:
            return

        frames.append(self._exporter.export(self._renderer.console, tick))
