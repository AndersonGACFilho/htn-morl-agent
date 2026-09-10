# Concepts and glossary

| Term                      | Meaning in this project                                                                                |
|---------------------------|--------------------------------------------------------------------------------------------------------|
| HTN                       | A planning technique that decomposes abstract tasks into subtasks until primitive actions are reached. |
| Domain                    | An ordered collection of root tasks that the planner must satisfy.                                     |
| Primitive task            | An executable leaf with an action, preconditions, and effects.                                         |
| Compound task             | An abstract goal decomposed by alternative methods.                                                    |
| Method                    | A decomposition rule: preconditions plus an ordered list of subtasks.                                  |
| Precondition              | A fact that must be true in the symbolic state for a branch to be applicable.                          |
| Effect                    | A symbolic transformation used only to predict state during planning or validation.                    |
| Symbolic state            | `WorldState`, the planner's view of the world.                                                         |
| Concrete environment      | The system that actually changes, such as `GridWorldEnv`.                                              |
| Sensor                    | An adapter that observes the concrete world and updates the symbolic state.                            |
| Tick                      | One iteration of agent execution; at most one current action advances.                                 |
| Backtracking              | Trying the next method when the current decomposition fails.                                           |
| Method-selection strategy | A policy that orders feasible methods before the planner recursively decomposes them.                  |
| Lazy replanning           | Rebuilding only when no plan exists or validation of the remaining plan fails.                         |
| BFS                       | Breadth-first search; in GridWorld, it finds routes with the fewest moves.                             |

## Proposed learning concepts

These terms describe the research extension rather than additional implemented runtime APIs.

| Term                                   | Meaning in the proposal                                                                                 |
|----------------------------------------|---------------------------------------------------------------------------------------------------------|
| Observed state                         | Sensor-produced symbolic snapshot of the concrete environment.                                          |
| Hypothetical state                     | A planning copy advanced by symbolic effects, not by new observations.                                  |
| Reward vector                          | Objective consequences of one transition, initially ordered `[time, energy, safety]`.                   |
| Method return                          | Discounted primitive rewards accumulated over a defined method interval.                                |
| Vector Q                               | Expected return of a method choice in its state, task, and continuation-policy/preference context.      |
| Preference vector                      | Weights used by the strategy to compare objective values; not an input to objective reward calculation. |
| Planning update                        | Model-assisted Q update using predicted return, next state, and duration.                               |
| Empirical correction                   | Q update using linked execution evidence, pending until the next decision boundary when needed.         |
| PlannedTransition / ExecutedTransition | Proposed paired records linking prediction and observation by method decision identity.                 |
| Episode clock                          | Time owned by one environment episode; distinct from action count and copied for planning.              |
| Objective reward                       | A signal measuring an outcome the task actually seeks to optimize.                                      |
| Reward shaping                         | An additional designed learning incentive, distinct from the measured objective return.                 |

See [extension contracts](../framework/extensions.md#proposed-reward-and-experience-contracts) and the [symbolic MORL architecture](../architecture/symbolic-morl.md).

## Module reference

| Path                       | Contents                              |
|----------------------------|---------------------------------------|
| `htn.world`                | Symbolic state and world abstractions |
| `htn.tasks`                | HTN domain language                   |
| `htn.planner`              | Recursive planning                    |
| `htn.strategy`             | Feasible-method ordering policies     |
| `htn.agent`                | Running plan and replanning           |
| `htn.sensors`              | Observation and notification          |
| `htn.actions`              | Action and status contract            |
| `htn.pathfinding`          | Generic pathfinding contract          |
| `htn._examples.grid_world` | End-to-end demonstration              |
