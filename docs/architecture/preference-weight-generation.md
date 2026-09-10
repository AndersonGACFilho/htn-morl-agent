# Preference sources and optional weight generation

## Role in the architecture

The research focus is MORL-guided HTN method selection at planning time. A preference vector $\mathbf{w}_t$ is an input that expresses the current trade-off among objectives. It may be supplied directly by the game or AI director, or produced by a **replaceable experimental component**; it is not a separate planner and not the central contribution.

In the proposed prediction–learning–execution–correction cycle, the reward function reports objective consequences without receiving $\mathbf w_t$. The strategy uses weights to compare vector Q estimates. Those estimates still depend on the compound-task context and continuation policy, which may itself depend on preferences; separating rewards from weights does not make Q universally preference-independent. The initial GridWorld objective order is `[time, energy, safety]` and must be shared by weights, rewards, values, and logs.

At each HTN decision point, a preference source supplies a normalized vector:

$$
\mathbf{w}_t = [w_1, w_2, \ldots, w_d]\in\mathbb{R}^d
$$

Here, $t$ identifies the current HTN decision point, $\mathbf{w}_t$ is the preference vector supplied at that point, $w_j$ is the weight of objective $j$, and $d$ is the number of objectives. When linear scalarization is used, weights are normally constrained by $w_j \geq 0$ and $\sum_{j=1}^{d} w_j = 1$.

The MORL policy receives $\mathbf{w}_t$ together with the feasible-method mask supplied by HTN. A preference source cannot override the HTN validity mask; an optional learned encoder is subject to the same constraint. In the latency-sensitive online path, $\mathbf{w}_t$ may come directly from a game profile, AI director, or explicit rule; a graph or deep encoder is optional and must not be required for every decision.

## Preference-source families

| Source             | Input                                    | Output                    | Purpose                                |
|--------------------|------------------------------------------|---------------------------|----------------------------------------|
| Fixed profile      | Designer-selected profile                | Constant $\mathbf{w}$     | Minimum MORL baseline.                 |
| Game / AI director | Runtime intent or scenario configuration | Contextual $\mathbf{w}_t$ | Direct low-latency source.             |
| Explicit rules     | State and context                        | Documented $\mathbf{w}_t$ | Interpretable dynamic baseline.        |
| Relational graph   | Typed state-and-goal graph               | Contextual $\mathbf{w}_t$ | Main structural candidate.             |
| Deep learning      | Features or learned state representation | Contextual $\mathbf{w}_t$ | Candidate for learned generalization.  |

## Optional relational graph encoder

The graph represents the relationships that matter for preferences. Candidate node types are goals, symbolic predicates, resources, threats, actors, HTN methods, and reward components. Candidate edge types include:

- a method requires a predicate or resource;
- a method changes a predicate or resource;
- a method contributes to a goal or reward component;
- a threat endangers an actor or goal; and
- an objective conflicts with, constrains, or supports another objective.

The representation is inspired by GOAP's explicit treatment of goals, conditions, effects, and costs. It does **not** add a GOAP search loop, calculate a GOAP plan, or introduce a second source of planning decisions. A graph neural network or message-passing aggregation can map the graph to $\mathbf{w}_t$.

## Optional deep-learning encoder

For a fixed-size observation, an MLP can infer $\mathbf{w}_t$ from state features, the current goal, context, and agent profile. When relationships and entity counts vary, a graph neural network is more appropriate. The output should be normalized, for example with softmax, and logged with each decision.

The deep-learning encoder is an extension after the structural baseline. It should be added only when available data, training budget, and the experimental question justify the additional complexity.

## Experimental comparison

Every encoder is evaluated under the same HTN domain, feasible-method mask, MORL policy interface, reward scales, scenario set, seeds, and training budget. This isolates the effect of preference inference from the effect of symbolic filtering and direct method selection.

| Comparison                         | Question                                                        |
|------------------------------------|-----------------------------------------------------------------|
| Fixed profile vs. rules            | Do dynamic preferences help over a stable profile?              |
| Rules vs. graph                    | Does relational structure improve adaptation or generalization? |
| Graph vs. deep learning            | Does learned representation justify its additional cost?        |
| Any encoder vs. fixed method order | Does preference-conditioned MORL improve HTN planning?          |

## Constraints and observability

- Normalize weights and record the normalization rule.
- Apply smoothing, hysteresis, commitment periods, or switch costs if weights cause repeated method changes. A small change in $\mathbf{w}_t$ should normally affect the next HTN decision point, not interrupt an already committed method.
- Log input context, $\mathbf{w}_t$, feasible methods, objective values, selected method, and observed outcome.
- Measure preference stability, adaptation latency, held-out-context performance, and inference cost.

These constraints keep the encoder auditable and prevent it from being confused with the policy that selects a method.

See [MORL](../annotations/reinforcement-learning/morl.md) for the decision model and [the symbolic MORL architecture](symbolic-morl.md) for the full planning flow.
