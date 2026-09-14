"""Composition root of the single-objective GridWorld example."""

from __future__ import annotations

import argparse
from pathlib import Path

from htn._examples.grid_world.core.config import GridWorldConfig
from htn._examples.grid_world.core.env import GridWorldEnv
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
from htn._examples.grid_world.core.world import GridWorld
from htn._examples.grid_world.single_objective.domain import build_key_door_domain
from htn._examples.grid_world.single_objective.sensors import KeyDoorSensor
from htn.agent.agent import Agent
from htn.planner.planner import Planner
from htn.sensors import SensorSystem
from htn.strategy.depth_first_search_strategy import DepthFirstSearchStrategy
from htn.world.state import WorldState

OUTPUT_ROOT = Path("out") / "grid_world" / "single_objective"
SEED = 42

DEMO_CONFIG = GridWorldConfig(
    width=10,
    height=10,
    start_position=None,
    key_position=None,
    door_position=None,
    goal_position=None,
    fixed_obstacles=frozenset({(2, 2), (3, 2)}),
    random_obstacle_count=10,
    initial_has_key=False,
    initial_door_open=False,
)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line options of the example."""
    parser = argparse.ArgumentParser(description="Run the key-door GridWorld example.")
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
        "--no-gif", action="store_true", help="Skip assembling the animated GIF."
    )

    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    """Run one episode of the key-door GridWorld and export its frames."""
    arguments = parse_arguments(argv)
    theme = GridWorldTheme.by_name(arguments.theme)

    env = GridWorldEnv(DEMO_CONFIG)
    env.reset(seed=arguments.seed)

    route_planner = RoutePlanner(GridPathfinder(), env.rules)
    domain = build_key_door_domain(env.layout, route_planner)

    world_state = WorldState()
    planner = Planner(domain, world_state, DepthFirstSearchStrategy())
    agent = Agent(planner, world_state, domain.tasks.copy())
    world = GridWorld(env, world_state, agent)

    sensor_system: SensorSystem[GridWorld] = SensorSystem()
    sensor_system.add_sensor(KeyDoorSensor())
    sensor_system.on_world_state_changed.add_handler(agent.handle_world_state_change)
    sensor_system.update(world, world_state)

    renderer = GridWorldRenderer(theme)
    exporter = SvgFrameExporter(
        OUTPUT_ROOT / "frames",
        theme,
        title_prefix=f"{env.layout.width}x{env.layout.height}",
    )

    runner = EpisodeRunner(
        agent=agent,
        world=world,
        world_state=world_state,
        sensor_system=sensor_system,
        renderer=renderer,
        grid_view_builder=GridViewBuilder(env.rules),
        decomposition_view_builder=DecompositionViewBuilder(domain),
        route_planner=route_planner,
        exporter=exporter,
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

    if arguments.no_gif or not result.frames:
        return

    gif_path = GifBuilder().build(
        result.frames,
        OUTPUT_ROOT / "episode.gif",
        progress=lambda index, total: renderer.print_message(
            f"Rendering frame {index}/{total}...", role=MessageRole.NEUTRAL
        ),
    )
    renderer.print_message(f"GIF saved at {gif_path}", role=MessageRole.SUCCESS)


if __name__ == "__main__":
    run()
