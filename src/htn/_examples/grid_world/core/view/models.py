"""View models describing one rendered frame.

Nothing in this module imports the environment, the agent, the domain, or
numpy. Panels consume these structures alone, which is what lets the renderer
show multi-objective information before a learner exists and keeps it unchanged
once one does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from htn._examples.grid_world.core.geometry import Position

OBJECTIVE_NAMES: tuple[str, ...] = ("time", "energy", "safety")


class CellRole(StrEnum):
    """What a tile represents, independently of how it is drawn."""

    AGENT = "agent"
    OBSTACLE = "obstacle"
    KEY = "key"
    DOOR_CLOSED = "door_closed"
    DOOR_OPEN = "door_open"
    GOAL = "goal"
    HAZARD = "hazard"
    RECHARGE = "recharge"
    EMPTY = "empty"


class MessageRole(StrEnum):
    """Meaning of a status message, so the theme can choose its style."""

    NEUTRAL = "neutral"
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class CellView:
    """One tile of the rendered grid.

    Attributes:
        position: Grid coordinate of the tile.
        role: What the tile represents.
        cost: Traversal cost, when the variant models one.
        risk: Perceived risk, when the variant models one.
        on_route: Whether the planned route crosses this tile.
        is_target: Whether the current task aims at this tile.
    """

    position: Position
    role: CellRole
    cost: float | None = None
    risk: float | None = None
    on_route: bool = False
    is_target: bool = False


class LegendKey(StrEnum):
    """Stable identity of a legend entry, independent of its current state."""

    AGENT = "agent"
    KEY = "key"
    DOOR = "door"
    GOAL = "goal"
    OBSTACLE = "obstacle"
    HAZARD = "hazard"
    RECHARGE = "recharge"


@dataclass(frozen=True, slots=True)
class LegendEntryView:
    """One legend row, describing what a scenario contains.

    Entries come from the scenario rather than from what is currently visible,
    so a tile the agent is standing on — or an item it collected — updates its
    row instead of dropping out of the legend.

    Attributes:
        key: Stable identity of the row.
        role: Current role, which selects the glyph and its colors.
        collected: Whether the agent already took this item off the grid.
    """

    key: LegendKey
    role: CellRole
    collected: bool = False


@dataclass(frozen=True, slots=True)
class GridView:
    """The grid as a table of tiles, in row-major order.

    Attributes:
        width: Grid width in tiles.
        height: Grid height in tiles.
        rows: Tiles, row by row.
        legend: One entry per element the scenario declares.
    """

    width: int
    height: int
    rows: tuple[tuple[CellView, ...], ...]
    legend: tuple[LegendEntryView, ...] = ()


class PlanEntryStatus(StrEnum):
    """How far the agent has progressed through one task of the plan."""

    DONE = "done"
    CURRENT = "current"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class PlanEntryView:
    """One task of the plan and its progress."""

    name: str
    status: PlanEntryStatus


@dataclass(frozen=True, slots=True)
class PlanView:
    """Progress through the plan the agent is executing.

    The whole plan is carried, finished tasks included, so the panel can show
    what was already accomplished instead of dropping it. A panel too short for
    the plan scrolls rather than truncating.

    Attributes:
        tick: Number of simulation ticks elapsed.
        entries: Every task of the current plan, in execution order.
        current_role: Outcome of the executing task, used to style it.
        replanned: Whether the plan was rebuilt on this tick.
    """

    tick: int
    entries: tuple[PlanEntryView, ...] = ()
    current_role: MessageRole = MessageRole.NEUTRAL
    replanned: bool = False


@dataclass(frozen=True, slots=True)
class MethodView:
    """One decomposition method and its applicability in the observed state."""

    id: str
    name: str
    applicable: bool
    failed_preconditions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskNodeView:
    """One node of the domain's decomposition structure."""

    name: str
    depth: int
    is_compound: bool
    is_current: bool = False
    methods: tuple[MethodView, ...] = ()


@dataclass(frozen=True, slots=True)
class DecompositionView:
    """The domain decomposition annotated with the current status."""

    nodes: tuple[TaskNodeView, ...] = ()


@dataclass(frozen=True, slots=True)
class ObjectiveView:
    """Multi-objective quantities for the panel.

    Every numeric field is optional. A field left as ``None`` renders as a
    placeholder, which is how the panel shows the space reserved for a learner
    that does not exist yet.

    Attributes:
        names: Objective order, shared with rewards, weights, and logs.
        step_reward: Reward vector of the last transition.
        episode_return: Undiscounted accumulation over the episode.
        weights: Preference vector, when one was supplied.
        utility: Scalarized utility, when weights were supplied.
    """

    names: tuple[str, ...] = OBJECTIVE_NAMES
    step_reward: tuple[float, ...] | None = None
    episode_return: tuple[float, ...] | None = None
    weights: tuple[float, ...] | None = None
    utility: float | None = None


@dataclass(frozen=True, slots=True)
class FrameView:
    """Everything one rendered frame needs."""

    grid: GridView
    plan: PlanView
    decomposition: DecompositionView = field(default_factory=DecompositionView)
    objectives: ObjectiveView | None = None
