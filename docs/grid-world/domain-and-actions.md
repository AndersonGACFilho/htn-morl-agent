# Domain, actions, and navigation

## HTN domain

`build_grid_world_domain(env)` creates the domain from the environment's effective positions. Movement effects are built by `_position_effects()`, keeping coordinates and location predicates consistent during planner simulation.

```mermaid
flowchart TD
    Escape[escape_grid] --> Open{door_open?}
    Open -->|yes, done=False| Goal[go_to_goal]
    Open -->|no, done=False| Key[ensure_has_key]
    Key --> Have{has_key?}
    Have -->|yes| Door[ensure_door_open]
    Have -->|no| GoKey[go_to_key] --> Pick[pickup_key] --> Door
    Door --> IsOpen{door_open?}
    IsOpen -->|yes| Goal
    IsOpen -->|no and has_key| GoDoor[go_to_door] --> Unlock[open_door] --> Goal
```

The textual equivalent of the decomposition is:

```text
escape_grid
├── [done=False, door_open=True]  go_to_goal
└── [done=False, door_open=False]
    ├── ensure_has_key
    │   ├── [has_key=True]         (no-op)
    │   └── [has_key=False]        go_to_key → pickup_key
    ├── ensure_door_open
    │   ├── [door_open=True]       (no-op)
    │   └── [door_open=False, has_key=True] go_to_door → open_door
    └── go_to_goal
```

The *no-op* branches are deliberate: they express that a subgoal has already been satisfied without inserting an artificial action.

## Concrete actions

### `NavigateToPositionAction`

It receives a target position and a `GridPathfinder`. On each tick, it:

1. builds or uses the context with dimensions and blocked cells;
2. calculates a BFS route from the current position;
3. returns `FAILURE` if no next step exists;
4. converts the next adjacent step with `action_from_step()`;
5. calls `env.step(action_id)` to move one cell;
6. returns `RUNNING` until it reaches the target and `SUCCESS` once it does.

Recalculating the route on every tick makes navigation reactive to obstacles and positions that may change, at the cost of repeating BFS.

### `PickupKeyAction` e `OpenDoorAction`

Both call `env.step()` with the respective constant. The first succeeds when `env.has_key` becomes true; the second succeeds when `env.door_open` becomes true. If the environment does not accept the operation under current conditions, they return `FAILURE`.

## BFS and movement

`GridContext` is an immutable dataclass with `width`, `height`, and `blocked: frozenset[Position]`. `GridPathfinder.find_path()` uses breadth-first search, so it finds a path with the fewest steps in an unweighted grid.

```mermaid
flowchart LR
    S[Start position] --> Q[BFS queue]
    Q --> N[Expands valid neighbors\nin deterministic order]
    N -->|new| V[Marks predecessor]
    V --> Q
    N -->|goal| R[Reconstructs route\nfrom end to start]
```

The returned path includes the start and goal; if they are equal, the route contains only the start position. If the goal cannot be reached, it returns an empty list. `action_from_step()` accepts only orthogonal neighbors and fails on an invalid jump, protecting the contract between the pathfinder and environment.

!!! note "Door and pathfinding"
    The domain decides when to open the door. The blocked-cell configuration supplied to navigation must remain consistent with the door's concrete state so the calculated path is executable.

## Unweighted routing and the heuristic baseline

`RoutePlanner` derives the blocked set from `GridRules` and searches it with
BFS, which minimizes the **number of steps** and nothing else. Cost and risk do
not influence the route.

This has a consequence worth stating plainly, because it limits what the
multi-objective example currently demonstrates: every method that navigates to
the same target follows the **same route**. `move_safely_to_goal`,
`walk_to_goal`, and `run_to_goal` differ only in the movement profile used to
traverse that one route — their time and energy costs differ, and the safe
profile avoids hazard *damage*, but none of them routes around a dangerous
region. A method named "safe route" that crosses the same tiles as the direct
one is therefore safe in a narrower sense than its name suggests.

Making the route itself risk-aware means replacing breadth-first search with a
weighted search — Dijkstra or A\* over a per-tile cost derived from
`ThreatRiskModel` — so that a longer route around a threat can outrank a short
one through it. That is a *declared heuristic over an explicit cost model*, not
a learned policy, which places it squarely at **Baseline 2 (HTN heurístico)** of
the evaluation plan rather than in the MORL contribution. The framework already
accommodates it: `Pathfinder[NodeT, ContextT]` makes no assumption about the
search algorithm, and `HeuristicBasedSearchStrategy` provides the matching
lower-is-better ordering hook at the method-selection layer.

Keeping the two layers distinct matters for the experiment. A weighted route
changes *where the agent walks* for a fixed method; a heuristic or learned
selector changes *which method is chosen*. Reporting a gain without separating
them would attribute to method selection an improvement that came from routing.

Neither the weighted pathfinder nor a concrete heuristic strategy is
implemented; both are recorded here as the next step for the symbolic baseline.
