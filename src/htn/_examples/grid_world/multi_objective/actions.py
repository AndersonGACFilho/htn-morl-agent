"""Actions specific to the multi-objective scenario."""

from __future__ import annotations

from typing import cast

from htn._examples.grid_world.core.world import GridWorld
from htn._examples.grid_world.multi_objective.config import RECHARGE
from htn._examples.grid_world.multi_objective.env import MultiObjectiveGridWorldEnv
from htn.actions.action import Action
from htn.actions.action_status import ActionStatus
from htn.world.world import World


class RechargeAction(Action):
    """Recover energy while the agent stands on a recharge tile."""

    def execute(self, world: World) -> ActionStatus:
        """Recharge once and report whether any energy was recovered."""
        grid_world = cast(GridWorld, world)
        env = grid_world.env
        assert isinstance(env, MultiObjectiveGridWorldEnv)

        energy_before = env.multi_objective_state.energy
        grid_world.step(env.codec.encode_interaction(RECHARGE))

        if env.multi_objective_state.energy > energy_before:
            return ActionStatus.SUCCESS

        return ActionStatus.FAILURE
