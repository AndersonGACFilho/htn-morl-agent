"""Tests for the view-model builders."""

from __future__ import annotations

import numpy as np
import pytest
from htn._examples.grid_world.core.layout import GridLayout
from htn._examples.grid_world.core.pathfinder import GridPathfinder
from htn._examples.grid_world.core.route import RoutePlanner
from htn._examples.grid_world.core.rules import GridRules
from htn._examples.grid_world.core.state import GridWorldState
from htn._examples.grid_world.core.view.decomposition_view_builder import (
    DecompositionViewBuilder,
)
from htn._examples.grid_world.core.view.grid_view_builder import GridViewBuilder
from htn._examples.grid_world.core.view.models import (
    CellRole,
    LegendEntryView,
    LegendKey,
)
from htn._examples.grid_world.core.view.objective_view_builder import (
    ObjectiveViewBuilder,
)
from htn._examples.grid_world.single_objective.domain import build_key_door_domain

from tests.conftest import build_world_state


def _cell(view, position):
    """Return the cell of a grid view at a position."""
    return view.rows[position[1]][position[0]]


def test_grid_view_marks_the_agent_tile(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    view = GridViewBuilder(rules).build(layout, state)

    assert _cell(view, state.agent_position).role is CellRole.AGENT


def test_grid_view_hides_the_key_once_it_is_held(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 0), has_key=True)

    view = GridViewBuilder(rules).build(layout, state)

    assert _cell(view, layout.key_position).role is not CellRole.KEY


def test_grid_view_marks_a_closed_door(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    view = GridViewBuilder(rules).build(layout, state)

    assert _cell(view, layout.door_position).role is CellRole.DOOR_CLOSED


def test_grid_view_marks_cells_on_the_planned_route(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    route = RoutePlanner(GridPathfinder(), rules).find_route(layout, state, (2, 0))

    view = GridViewBuilder(rules).build(layout, state, route=route)

    assert _cell(view, (1, 0)).on_route


def test_grid_view_marks_the_current_task_target_cell(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    view = GridViewBuilder(rules).build(layout, state, target=layout.key_position)

    assert _cell(view, layout.key_position).is_target


def _entry(view, key: LegendKey) -> LegendEntryView:
    """Return the legend entry carrying the given key."""
    return next(entry for entry in view.legend if entry.key is key)


def test_legend_marks_the_key_as_collected_once_it_is_held(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 0), has_key=True)

    view = GridViewBuilder(rules).build(layout, state)

    assert _entry(view, LegendKey.KEY).collected


def test_legend_keeps_the_key_entry_after_it_is_collected(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=(0, 0), has_key=True)

    view = GridViewBuilder(rules).build(layout, state)

    assert LegendKey.KEY in {entry.key for entry in view.legend}


def test_legend_keeps_the_goal_entry_while_the_agent_occludes_it(
    layout: GridLayout, rules: GridRules
) -> None:
    state = GridWorldState(agent_position=layout.goal_position, door_open=True)

    view = GridViewBuilder(rules).build(layout, state)

    assert LegendKey.GOAL in {entry.key for entry in view.legend}


def test_legend_updates_the_door_entry_instead_of_replacing_it(
    layout: GridLayout, rules: GridRules
) -> None:
    builder = GridViewBuilder(rules)
    closed = builder.build(layout, GridWorldState(agent_position=(0, 0)))
    opened = builder.build(
        layout, GridWorldState(agent_position=(0, 0), door_open=True)
    )

    assert (
        _entry(closed, LegendKey.DOOR).role is not _entry(opened, LegendKey.DOOR).role
    )


def test_legend_keeps_the_same_entries_when_the_door_opens(
    layout: GridLayout, rules: GridRules
) -> None:
    builder = GridViewBuilder(rules)
    closed = builder.build(layout, GridWorldState(agent_position=(0, 0)))
    opened = builder.build(
        layout, GridWorldState(agent_position=(0, 0), door_open=True)
    )

    assert [entry.key for entry in closed.legend] == [
        entry.key for entry in opened.legend
    ]


def test_legend_omits_terrain_the_scenario_does_not_declare(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    view = GridViewBuilder(rules).build(layout, state)

    assert LegendKey.HAZARD not in {entry.key for entry in view.legend}


def test_legend_includes_terrain_the_scenario_declares(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    builder = GridViewBuilder(rules, hazards=frozenset({(2, 1)}))

    view = builder.build(layout, state)

    assert LegendKey.HAZARD in {entry.key for entry in view.legend}


def test_grid_view_leaves_risk_unset_without_a_risk_field(
    layout: GridLayout, rules: GridRules, state: GridWorldState
) -> None:
    view = GridViewBuilder(rules).build(layout, state)

    assert _cell(view, (1, 0)).risk is None


def test_decomposition_view_marks_an_applicable_method(
    layout: GridLayout, rules: GridRules
) -> None:
    domain = build_key_door_domain(layout, RoutePlanner(GridPathfinder(), rules))
    world_state = build_world_state(done=False, door_open=False, has_key=False)

    view = DecompositionViewBuilder(domain).build(world_state)
    methods = {method.id: method for method in view.nodes[0].methods}

    assert methods["escape_grid.prepare_and_exit"].applicable


def test_decomposition_view_lists_the_failing_precondition_key(
    layout: GridLayout, rules: GridRules
) -> None:
    domain = build_key_door_domain(layout, RoutePlanner(GridPathfinder(), rules))
    world_state = build_world_state(done=False, door_open=False, has_key=False)

    view = DecompositionViewBuilder(domain).build(world_state)
    methods = {method.id: method for method in view.nodes[0].methods}

    assert methods["escape_grid.exit_through_open_door"].failed_preconditions == (
        "door_open",
    )


def test_decomposition_view_treats_a_missing_fact_as_a_failed_precondition(
    layout: GridLayout, rules: GridRules
) -> None:
    domain = build_key_door_domain(layout, RoutePlanner(GridPathfinder(), rules))

    view = DecompositionViewBuilder(domain).build(build_world_state())
    methods = {method.id: method for method in view.nodes[0].methods}

    assert "done" in methods["escape_grid.prepare_and_exit"].failed_preconditions


def test_objective_view_leaves_weights_unset_without_a_learner() -> None:
    view = ObjectiveViewBuilder().build(np.array([-1.0, -2.0, -3.0]), None)

    assert view.weights is None


def test_objective_view_leaves_utility_unset_without_weights() -> None:
    view = ObjectiveViewBuilder().build(None, np.array([-1.0, -2.0, -3.0]))

    assert view.utility is None


def test_objective_view_scalarizes_the_return_with_supplied_weights() -> None:
    view = ObjectiveViewBuilder().build(
        None, np.array([-2.0, -3.0, -1.0]), weights=[0.7, 0.2, 0.1]
    )

    assert view.utility == pytest.approx(-2.1)


def test_objective_view_keeps_the_declared_objective_order() -> None:
    view = ObjectiveViewBuilder().build(None, None)

    assert view.names == ("time", "energy", "safety")


def test_objective_view_rejects_a_weight_vector_of_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="Expected 3 weights"):
        ObjectiveViewBuilder().build(None, None, weights=[1.0])
