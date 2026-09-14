"""Translation of tick outcomes into presentable messages."""

from __future__ import annotations

from htn._examples.grid_world.core.view.models import MessageRole
from htn.actions.action_status import ActionStatus
from htn.agent.agent import AgentTickResult


class TickPresenter:
    """Turn an agent tick result into the role that styles the plan panel.

    Keeping this mapping out of the composition root stops presentation
    decisions from leaking into the wiring, and out of the renderer, which
    should not know what an ``ActionStatus`` is.
    """

    def role_of(self, result: AgentTickResult) -> MessageRole:
        """Return the role conveying the outcome of one tick.

        Args:
            result: Outcome reported by the agent for this tick.

        Returns:
            The role used to colour the executing task.
        """
        roles = {
            ActionStatus.SUCCESS: MessageRole.SUCCESS,
            ActionStatus.RUNNING: MessageRole.RUNNING,
            ActionStatus.FAILURE: MessageRole.FAILURE,
        }

        if result.status is None:
            return MessageRole.NEUTRAL

        return roles[result.status]
