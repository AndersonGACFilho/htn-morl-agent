"""Adapter between the HTN world abstraction and the GridWorld environment."""

from __future__ import annotations

from typing import Any

from htn._examples.grid_world.core.env import GridWorldEnv
from htn.agent.agent import Agent
from htn.world.gym.gym_world import GymWorld
from htn.world.state import WorldState


class GridWorld(GymWorld):
    """Expose the GridWorld environment to actions and record each step.

    Actions call :meth:`step` rather than reaching into the environment, so the
    latest observation and reward are always captured in one place.
    """

    env: GridWorldEnv

    def __init__(
        self,
        env: GridWorldEnv,
        world_state: WorldState,
        agent: Agent,
    ) -> None:
        """Initialize the adapter and seed it with the current observation."""
        super().__init__(env, world_state, agent)
        self.done = env.done

    def step(self, action: int) -> None:
        """Execute one environment action and refresh the cached observation."""
        observation, reward, terminated, _, _ = self.env.step(action)
        self.last_reward = float(reward)
        self.update_from_obs(observation)
        self.done = terminated

    def update_from_obs(self, obs: object) -> None:
        """Store the latest observation and mirror its terminal flag."""
        self.last_obs = obs

        if isinstance(obs, dict):
            self.done = bool(obs.get("done", self.done))

    def observation(self) -> dict[str, Any] | None:
        """Return the last observation when one has been recorded."""
        if isinstance(self.last_obs, dict):
            return self.last_obs

        return None
