"""End-to-end tests of the two example episodes."""

from __future__ import annotations

import pytest
from htn._examples.grid_world.core.runtime.presenter import TickPresenter
from htn._examples.grid_world.core.view.models import MessageRole
from htn._examples.grid_world.multi_objective.main import run as run_multi_objective
from htn._examples.grid_world.single_objective.main import run as run_single_objective
from htn.actions.action_status import ActionStatus
from htn.agent.agent import AgentTickResult


def _result(status: ActionStatus | None, message: str | None = None) -> AgentTickResult:
    """Return an agent tick result carrying the given status."""
    return AgentTickResult(
        task_name="go_to_key",
        status=status,
        replanned=False,
        planned_tasks=[],
        remaining_plan=[],
        message=message,
    )


def test_presenter_marks_a_succeeded_task_as_success() -> None:
    assert TickPresenter().role_of(_result(ActionStatus.SUCCESS)) is MessageRole.SUCCESS


def test_presenter_marks_a_failed_task_as_failure() -> None:
    assert TickPresenter().role_of(_result(ActionStatus.FAILURE)) is MessageRole.FAILURE


def test_presenter_marks_a_running_task_as_running() -> None:
    assert TickPresenter().role_of(_result(ActionStatus.RUNNING)) is MessageRole.RUNNING


def test_presenter_falls_back_to_neutral_without_a_status() -> None:
    assert TickPresenter().role_of(_result(None)) is MessageRole.NEUTRAL


def test_single_objective_example_runs_to_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_single_objective(["--no-gif"])


def test_multi_objective_example_runs_to_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    run_multi_objective(["--no-gif", "--display-weights", "0.2", "0.2", "0.6"])


def test_multi_objective_example_rejects_a_weight_vector_of_the_wrong_length(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        run_multi_objective(["--no-gif", "--display-weights", "0.5", "0.5"])
