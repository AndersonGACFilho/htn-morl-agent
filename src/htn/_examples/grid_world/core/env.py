"""Gymnasium facade composing the GridWorld layout, rules, and observations."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from htn._examples.grid_world.core.clock import EpisodeClock
from htn._examples.grid_world.core.config import (
    GridWorldConfig,
    validate_grid_world_config,
)
from htn._examples.grid_world.core.layout import GridLayout, LayoutResolver
from htn._examples.grid_world.core.movement_profile import (
    WALK,
    MoveCommand,
    MovementActionCodec,
)
from htn._examples.grid_world.core.observation import GridObservationBuilder
from htn._examples.grid_world.core.rules import OPEN_DOOR, PICKUP_KEY, GridRules
from htn._examples.grid_world.core.state import GridWorldState

GOAL_REWARD = 1.0


def build_default_codec() -> MovementActionCodec:
    """Return the single-profile codec used by the key-door scenario."""
    return MovementActionCodec(profiles=(WALK,), interactions=(PICKUP_KEY, OPEN_DOOR))


class GridWorldEnv(gym.Env):
    """Configurable deterministic GridWorld used as a simulation sandbox.

    The environment does not decide what to do. The HTN planner chooses the
    symbolic intention and the action classes translate tasks into steps. This
    class only sequences its collaborators: it resolves a layout, applies the
    rules, advances the episode clock, and reports an observation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: GridWorldConfig | None = None,
        *,
        rules: GridRules | None = None,
        codec: MovementActionCodec | None = None,
    ) -> None:
        """Initialize the environment.

        Args:
            config: Scenario configuration; the 3x3 example layout by default.
            rules: Movement and interaction semantics.
            codec: Encoding between discrete action ids and commands.

        Raises:
            ValueError: If the configuration is inconsistent.
        """
        super().__init__()

        self.config = config or GridWorldConfig()
        validate_grid_world_config(self.config)

        self._rules = rules or GridRules()
        self._codec = codec or build_default_codec()
        self._resolver = LayoutResolver(self.config)
        self._observations = GridObservationBuilder(
            self.config.width, self.config.height
        )
        self._clock = EpisodeClock()

        self._layout = self._resolver.resolve(self.np_random)
        self._state = self._initial_state(self._layout)

        self.action_space = gym.spaces.Discrete(self._codec.action_count)
        self.observation_space = self._observations.space()

    @property
    def layout(self) -> GridLayout:
        """Return the entity placement of the current episode."""
        return self._layout

    @property
    def state(self) -> GridWorldState:
        """Return the current episode state."""
        return self._state

    @property
    def rules(self) -> GridRules:
        """Return the rules governing this environment."""
        return self._rules

    @property
    def codec(self) -> MovementActionCodec:
        """Return the action encoding of this environment."""
        return self._codec

    @property
    def clock(self) -> EpisodeClock:
        """Return the clock owned by the current episode."""
        return self._clock

    @property
    def done(self) -> bool:
        """Return whether the episode reached its terminal condition."""
        return self._state.done

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve a new layout and restart the episode.

        Random positions are resolved here rather than in ``__init__`` so each
        reset can produce a new layout; passing a seed makes it reproducible.
        """
        super().reset(seed=seed)

        self._layout = self._resolver.resolve(self.np_random)
        self._state = self._initial_state(self._layout)
        self._clock.reset()

        return self._observations.build(self._layout, self._state), {}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Execute one action and return the Gymnasium five-tuple.

        Raises:
            ValueError: If the action is outside the action space.
        """
        if self._state.done:
            return (
                self._observations.build(self._layout, self._state),
                0.0,
                True,
                False,
                {},
            )

        command = self._codec.decode(int(action))
        self._state = self._rules.apply(self._layout, self._state, command)
        self._clock.advance(self._duration_of(command))

        reward = GOAL_REWARD if self._state.done else 0.0
        observation = self._observations.build(self._layout, self._state)

        return observation, reward, self._state.done, False, {}

    def render(self) -> str:
        """Return an ANSI representation of the current grid."""
        from htn._examples.grid_world.core.theme import GridWorldTheme
        from htn._examples.grid_world.core.view.grid_view_builder import (
            GridViewBuilder,
        )

        theme = GridWorldTheme.dissertation()
        grid = GridViewBuilder(self._rules).build(self._layout, self._state)

        rows = [
            " ".join(theme.glyph(cell.role).strip() for cell in row)
            for row in grid.rows
        ]
        rows.append(
            f"agent={self._state.agent_position}, "
            f"has_key={self._state.has_key}, "
            f"door_open={self._state.door_open}, "
            f"done={self._state.done}"
        )

        return "\n".join(rows)

    def _initial_state(self, layout: GridLayout) -> GridWorldState:
        """Build the starting state for a freshly resolved layout."""
        state = GridWorldState(
            agent_position=layout.start_position,
            has_key=self.config.initial_has_key,
            door_open=self.config.initial_door_open,
        )

        return state.with_changes(
            done=state.agent_position == layout.goal_position and state.door_open
        )

    def _duration_of(self, command: object) -> float:
        """Return the episode time consumed by a command."""
        if isinstance(command, MoveCommand):
            return command.profile.time_cost

        return WALK.time_cost
