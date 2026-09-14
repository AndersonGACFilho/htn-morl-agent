# Environment and configuration

## `GridWorldConfig`

`GridWorldConfig` is an immutable dataclass (`frozen=True`, `slots=True`) that supports deterministic and random scenarios.

| Group         | Parameters                                                         |
|---------------|--------------------------------------------------------------------|
| Dimensions    | `width`, `height`                                                  |
| Entities      | `start_position`, `key_position`, `door_position`, `goal_position` |
| Obstacles     | `fixed_obstacles`, `random_obstacle_count`                         |
| Initial state | `initial_has_key`, `initial_door_open`                             |

A `None` position is resolved randomly in `reset()`. Fixed positions are appropriate for tests and reproducible demonstrations; use a seed in `reset(seed=...)` when the configuration includes randomness.

```python
config = GridWorldConfig(
    width=8,
    height=6,
    start_position=None,
    key_position=None,
    door_position=None,
    goal_position=None,
    fixed_obstacles=frozenset({(2, 2), (3, 2)}),
    random_obstacle_count=5,
    initial_has_key=False,
    initial_door_open=False,
)
env = GridWorldEnv(config)
obs, info = env.reset(seed=42)
```

## Gymnasium interface

`GridWorldEnv` is a `gym.Env`. Its `observation_space` is a `spaces.Dict`
representing the environment state, and its `action_space` is derived by
`MovementActionCodec` rather than fixed by hand: move actions come first,
ordered by movement profile and then by direction, followed by the declared
interactions.

The key–door scenario declares one profile and two interactions, reproducing
the conventional six-action layout:

|  Id | Operation                        |
|----:|----------------------------------|
|   0 | walk up                          |
|   1 | walk right                       |
|   2 | walk down                        |
|   3 | walk left                        |
|   4 | picks up the key when applicable |
|   5 | opens the door when applicable   |

The multi-objective scenario declares four profiles — walk, run, safe move, and
climb — and three interactions, giving `Discrete(19)`. Encode an action through
the codec instead of assuming an id:

```python
action = env.codec.encode_move(env.codec.profile_index("run"), 1)
```

`reset()` resolves the layout, restores `has_key` and `door_open` according to the configuration, and returns `(observation, {})`. `step(action)` returns the five-item Gymnasium contract: observation, reward, `terminated`, `truncated`, and `info`, and advances the episode clock by the command's duration.

!!! tip "Termination"
    The example uses `env.done` to control the loop. The goal is completing navigation to the objective, reflected first by the concrete state and then by the sensor.

## Layout validation

The environment places entities and obstacles without allowing invalid collisions. Obstacles are consulted both by the environment when processing movement and by the pathfinder when finding routes. A random configuration is reliable only if enough space is available for the selected entities and barriers.

## Observation versus HTN facts

The observation object is the environment interface; it does not populate `WorldState` directly. `GridWorldSensor` performs this translation and exposes the facts required by the domain. This separation lets the observation shape change without rewriting the planner, provided the sensor preserves the symbolic contract.
