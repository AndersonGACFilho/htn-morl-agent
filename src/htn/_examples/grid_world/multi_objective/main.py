"""Composition root of the multi-objective GridWorld example."""

from __future__ import annotations

import argparse
from pathlib import Path

from htn._examples.grid_world.core.config import GridWorldConfig
from htn._examples.grid_world.core.pathfinder import GridPathfinder
from htn._examples.grid_world.core.rendering.export import GifBuilder, SvgFrameExporter
from htn._examples.grid_world.core.rendering.renderer import GridWorldRenderer
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.core.runtime.episode_runner import EpisodeRunner
from htn._examples.grid_world.core.theme import GridWorldTheme
from htn._examples.grid_world.core.view.decomposition_view_builder import (
    DecompositionViewBuilder,
)
from htn._examples.grid_world.core.view.grid_view_builder import GridViewBuilder
from htn._examples.grid_world.core.view.models import MessageRole
from htn._examples.grid_world.core.view.objective_view_builder import (
    ObjectiveViewBuilder,
)
from htn._examples.grid_world.core.world import GridWorld
from htn._examples.grid_world.multi_objective.config import (
    MultiObjectiveGridConfig,
    Threat,
)
from htn._examples.grid_world.multi_objective.domain import (
    build_multi_objective_domain,
)
from htn._examples.grid_world.multi_objective.env import MultiObjectiveGridWorldEnv
from htn._examples.grid_world.multi_objective.reward import build_reward_function
from htn._examples.grid_world.multi_objective.risk import HealthAwareRiskField
from htn._examples.grid_world.multi_objective.sensors import MultiObjectiveGridSensor
from htn.agent.agent import Agent
from htn.planner.planner import Planner
from htn.sensors import SensorSystem
from htn.strategy.depth_first_search_strategy import DepthFirstSearchStrategy
from htn.world.state import WorldState

OUTPUT_ROOT = Path("out") / "grid_world" / "multi_objective"
SEED = 42
RISK_OVERLAY_SCALE = 2.0

DEMO_CONFIG = MultiObjectiveGridConfig(
    grid=GridWorldConfig(
        width=10,
        height=10,
        start_position=(0, 0),
        key_position=(9, 0),
        door_position=(5, 5),
        goal_position=(9, 9),
        fixed_obstacles=frozenset(
            {(2, 2), (3, 2), (6, 6), (6, 7), (2, 5), (3, 5), (7, 1), (8, 5)}
        ),
        random_obstacle_count=0,
        initial_has_key=False,
        initial_door_open=False,
    ),
    recharge_positions=frozenset({(4, 6)}),
    rough_terrain=frozenset({(4, 4), (5, 4)}),
    hazards=frozenset({(7, 3), (7, 4)}),
    threats=(
        Threat(position=(7, 3), threat_weight=1.2),
        Threat(position=(3, 7), threat_weight=0.8),
    ),
    initial_energy=24.0,
    max_energy=40.0,
    recharge_amount=20.0,
    low_energy_threshold=25.0,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options of the example."""
    parser = argparse.ArgumentParser(
        description="Run the multi-objective GridWorld example."
    )
    parser.add_argument(
        "--theme",
        default="dissertation",
        choices=["dissertation", "eramia"],
        help="Color palette matching one of the manuscripts.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Seed used to resolve the layout."
    )
    parser.add_argument(
        "--display-weights",
        type=float,
        nargs=3,
        metavar=("TIME", "ENERGY", "SAFETY"),
        default=None,
        help=(
            "Preference vector shown in the objective panel. Display only: no "
            "learner uses it and no method selection depends on it."
        ),
    )
    parser.add_argument(
        "--no-gif", action="store_true", help="Skip assembling the animated GIF."
    )

    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    """Run one episode of the multi-objective GridWorld and export its frames."""
    arguments = parse_arguments(argv)
    theme = GridWorldTheme.by_name(arguments.theme)

    env = MultiObjectiveGridWorldEnv(DEMO_CONFIG)
    env.reset(seed=arguments.seed)

    route_planner = RoutePlanner(GridPathfinder(), env.rules)
    domain = build_multi_objective_domain(env.layout, route_planner, DEMO_CONFIG)

    world_state = WorldState()
    planner = Planner(domain, world_state, DepthFirstSearchStrategy())
    agent = Agent(planner, world_state, domain.tasks.copy())
    world = GridWorld(env, world_state, agent)

    sensor_system: SensorSystem[GridWorld] = SensorSystem()
    sensor_system.add_sensor(MultiObjectiveGridSensor(DEMO_CONFIG))
    sensor_system.on_world_state_changed.add_handler(agent.handle_world_state_change)
    sensor_system.update(world, world_state)

    renderer = GridWorldRenderer(theme, show_objectives=True)
    risk_field = HealthAwareRiskField(
        env.risk_model, env.multi_objective_state.health, RISK_OVERLAY_SCALE
    )

    runner = EpisodeRunner(
        agent=agent,
        world=world,
        world_state=world_state,
        sensor_system=sensor_system,
        renderer=renderer,
        grid_view_builder=GridViewBuilder(
            env.rules,
            risk_field=risk_field,
            hazards=DEMO_CONFIG.hazards,
            recharge_positions=DEMO_CONFIG.recharge_positions,
        ),
        decomposition_view_builder=DecompositionViewBuilder(domain),
        route_planner=route_planner,
        exporter=SvgFrameExporter(OUTPUT_ROOT / "frames", theme, title_prefix="morl"),
        reward_function=build_reward_function(),
        objective_view_builder=ObjectiveViewBuilder(),
        display_weights=arguments.display_weights,
    )

    result = runner.run()

    if result.succeeded:
        renderer.print_message(
            f"Episode succeeded in {result.ticks} ticks.", role=MessageRole.SUCCESS
        )
    else:
        renderer.print_message(
            f"Episode did not finish within {result.ticks} ticks.",
            role=MessageRole.FAILURE,
        )

    if result.episode_return is not None:
        components = ", ".join(
            f"{name}={value:.2f}"
            for name, value in zip(("time", "energy", "safety"), result.episode_return)
        )
        renderer.print_message(f"Episode return: [{components}]")

    if arguments.no_gif or not result.frames:
        return

    gif_path = GifBuilder().build(
        result.frames,
        OUTPUT_ROOT / "episode.gif",
        progress=lambda index, total: renderer.print_message(
            f"Rendering frame {index}/{total}..."
        ),
    )
    renderer.print_message(f"GIF saved at {gif_path}", role=MessageRole.SUCCESS)


if __name__ == "__main__":
    run()
