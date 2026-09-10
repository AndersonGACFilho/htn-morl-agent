# HTN framework overview

The `htn` module separates three responsibilities: the **Planner** builds a symbolic sequence of primitive tasks; the **Agent** decides when to reuse or rebuild it and executes one step per tick; the environment and **Sensors** provide observable truth.

## Components

| Layer              | Responsibility                                                   | Main implementations                 |
|--------------------|------------------------------------------------------------------|--------------------------------------|
| Symbolic state     | Stores named facts                                               | `WorldState`                         |
| Domain             | Declares high-level tasks                                        | `Domain`, `CompoundTask`, `Method`   |
| Executable leaves  | Connect conditions and effects to code                           | `PrimitiveTask`, `Action`            |
| Planning           | Orders feasible methods, decomposes tasks, and simulates effects | `Planner`, `MethodSelectionStrategy` |
| Execution          | Keeps the plan alive and executes ticks                          | `Agent`                              |
| Observation        | Converts the concrete world into facts                           | `Sensor`, `SensorSystem`             |
| Integration        | Provides execution context                                       | `World`, `GymWorld`, `Pathfinder`    |

## Control cycle

```mermaid
sequenceDiagram
    participant Loop as Application loop
    participant Sensors as SensorSystem
    participant Agent as Agent
    participant Planner as Planner
    participant Action as Action
    participant World as World/Env

    Loop->>Sensors: update(world, world_state)
    Sensors->>World: reads concrete state
    Sensors->>Agent: on_world_state_changed
    Agent->>Planner: update_world_state(copy)
    Loop->>Agent: tick(world)
    alt no plan or invalid plan
        Agent->>Planner: build_plan(root tasks)
        Planner-->>Agent: PrimitiveTask[]
    end
    Agent->>Action: execute(world)
    Action->>World: applies an operation
    Action-->>Agent: returns status (RUNNING, SUCCESS, or FAILURE)
```

The planner **does not execute actions** and does not modify the live `WorldState`. It applies effects only to copies. After an action changes the concrete environment, a sensor updates the symbolic state in the next cycle.

### Proposed learning cycle

The research extension adds **prediction → planning learning → execution → empirical correction** to this runtime. Only the initial planning state comes from sensors; subsequent hypothetical states come from primitive-task symbolic effects. A domain reward function evaluates these predicted transitions, and method-level vector returns support a planning update of Q. Execution records the corresponding observed outcomes for correction at the next planning boundary, once the next applicable methods are known. Prior training remains an optional complement.

This learning cycle is proposed, not provided by the current strategy hook. Its target is a complete primitive plan; the current planner can return a successfully planned prefix when a later root task fails. Local method applicability also does not guarantee a complete decomposition or successful execution in the real environment. See the [symbolic MORL architecture](../architecture/symbolic-morl.md) and [proposed extension contracts](extensions.md#proposed-reward-and-experience-contracts).

## Code organization

```text
src/htn/
├── actions/       # Action and ActionStatus contract
├── agent/         # execution and replanning
├── delegates/     # event multicast
├── pathfinding/   # generic Pathfinder contract
├── planner/       # recursive HTN decomposition
├── strategy/      # method-ordering policies
├── sensors/       # observation and coordination
├── tasks/         # domain, tasks, methods, conditions, and effects
├── world/         # WorldState, World e GymWorld
└── _examples/grid_world/
```

## Important invariants

1. A normal execution plan contains only `PrimitiveTask` instances.
2. Every planner branch receives a copy of the simulated state.
3. A primitive task enters the plan only if its preconditions are true.
4. Actions return an `ActionStatus`; `RUNNING` keeps the same task at the front of the plan.
5. Sensors observe; they do not decide or execute actions.
6. A strategy ranks only methods already feasible in the symbolic state; the
   planner preserves method-level backtracking after a ranked branch fails.

See the [domain and state model](model.md) for the symbolic language and the [planner and agent guide](planning-runtime.md) for runtime behavior.
