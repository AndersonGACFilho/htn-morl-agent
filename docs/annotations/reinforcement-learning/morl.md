# Multi-Objective Reinforcement Learning (MORL)

Multi-Objective Reinforcement Learning studies sequential decisions with several objectives that may conflict.
The environment provides a **reward vector** instead of a single scalar reward.

This page introduces the general concepts. The project's HTN integration, transition-based reward interpretation, and proposed planning updates with empirical correction are described in the [symbolic MORL architecture](../../architecture/symbolic-morl.md).

## Vector rewards

For $d$ objectives, the reward at step $t$ is:

$$
\mathbf r_t=[r_{1,t},r_{2,t},\ldots,r_{d,t}]^{\mathsf T}.
$$

Each component measures one objective. For example:

- Mission progress.
- Damage avoidance.
- Resource conservation.
- Time efficiency.

Keeping these signals separate makes the trade-offs explicit.
Signs and scales must be defined consistently: a larger numerical component can dominate a weighted comparison even with a small weight.

## Rewards, returns, and values

| Concept                    | Meaning                                                                  |
|----------------------------|--------------------------------------------------------------------------|
| Reward $\mathbf r_t$       | Feedback for one interaction step.                                       |
| Return $\mathbf G_t$       | Accumulation of rewards over time.                                       |
| Value $\mathbf Q^\pi(s,a)$ | Expected return after action $a$ in state $s$, followed by policy $\pi$. |

The discounted vector return is:

$$
\mathbf G_t=\sum_{k=0}^{\infty}\gamma^k\mathbf r_{t+k}.
$$

- $\gamma$ controls the importance of later rewards.
- Episodic returns stop at the episode boundary.
- A value estimate accounts for future consequences, not only immediate feedback.

## Preferences and scalarization

A preference vector expresses the relative importance of the objectives.
For normalized non-negative linear weights:

$$
0\leq w_i\leq1,\qquad\sum_iw_i=1.
$$

Linear scalarization converts a vector return into scalar utility:

$$
u_{\mathbf w}(\mathbf G)=\mathbf w^{\mathsf T}\mathbf G=\sum_iw_iG_i.
$$

### Example

For the project's proposed order `[time, energy, safety]`:

| Weights           | Interpretation                                                                |
|-------------------|-------------------------------------------------------------------------------|
| `[0.7, 0.2, 0.1]` | Give time efficiency the largest weight while retaining the other objectives. |
| `[1, 0, 0]`       | Consider time efficiency alone: a one-hot vector.                             |

For $\mathbf r=[-2,-3,-1]^{\mathsf T}$:

$$
\mathbf w^{\mathsf T}\mathbf r=0.7(-2)+0.2(-3)+0.1(-1)=-2.1.
$$

The result is a **scalar dot product**. The reward vector remains available for evaluating other preferences.

### Scope of linear weights

- Intermediate weights express mixed preferences.
- Linear weights are a modeling choice, not a requirement of all MORL methods.
- Nonlinear utilities, lexicographic preferences, and hard constraints require their own formulations.
- For fixed weights and linear utility, scalarizing an expected return and taking the expectation of scalarized return coincide.

## Preference-conditioned policies

A policy can receive preferences as an input:

$$
\pi(a\mid s,\mathbf w).
$$

Changing $\mathbf w$ can change behavior without retraining a separate policy for each preference.
Whether this works for unseen weights must be evaluated experimentally.

A vector value function may also depend on preferences:

$$
\mathbf Q(s,a,\mathbf w).
$$

The dependence can arise because the continuation policy changes with $\mathbf w$, producing different trajectories and expected vector returns.
The weights need not change the underlying reward components.

### Objective reward and shaping

The proposed reward function interprets a transition as $\mathbf r=R(S,S')$; a raw state delta is not automatically a reward. Time, energy consumption, and safety are objective rewards when they are the outcomes being optimized. An extra incentive for approaching a key solely to accelerate learning is reward shaping. Potential-based shaping uses $F(s,s')=\gamma\Phi(s')-\Phi(s)$ for unit-step transitions; elapsed-duration discounting requires the corresponding duration in the exponent. Shaping and objective returns should be logged separately.

The planned method learner uses hypothetical transitions from the HTN effects and later corrects values with linked observed experience. The shorthand $\mathbf Q(S,M)$ suppresses compound-task and continuation-policy/preference context; it must not erase that context from a learner's representation.

## Pareto trade-offs and coverage

For maximization, one return vector dominates another when it is:

- At least as good in every objective.
- Strictly better in at least one objective.

The **Pareto frontier** contains nondominated achievable return vectors.
A **coverage set** contains policies sufficient for the preference class being considered.

Linear scalarization can select supported trade-offs; it need not recover every Pareto-optimal solution.
The appropriate solution set therefore depends on the utility model.

## Evaluation

Useful evaluation dimensions include:

- Vector returns for each objective.
- Utility under specified preferences.
- Performance on preferences not used in training.
- Coverage and hypervolume when the evaluated set and reference point are defined.
- Training cost and decision latency.

Comparisons should use consistent reward scales, preference distributions, and evaluation budgets.

## Further reading

- [RL basics](basics.md): states, actions, rewards, and temporal learning.
- [Bibliography](../../reference/bibliography.md): MORL surveys, preference specification, GPI, and evaluation references, including Hayes et al. and Felten et al.
- [Symbolic MORL architecture](../../architecture/symbolic-morl.md): how this project applies these concepts to HTN methods.
