# Reinforcement Learning with Options

Options are a formal model of **temporal abstraction** in reinforcement learning. Instead of selecting a primitive action every time step, a high-level policy can select an Option that controls behavior for several steps. Sutton, Precup, and Singh define an Option as:

$$
o = \langle \mathcal{I}, \pi, \beta \rangle
$$

Here, $o$ denotes one Option; $\mathcal{I}$ is its initiation set; $\pi$ is its internal policy; and $\beta$ specifies its termination behavior.

| Component     | Meaning                                                             |
|---------------|---------------------------------------------------------------------|
| $\mathcal{I}$ | Initiation set: the states in which the Option may start.           |
| $\pi$         | Internal policy: the behavior followed while the Option is active.  |
| $\beta$       | Termination condition: the probability or rule for ending control.  |

The resulting decision process is a semi-Markov decision process (semi-MDP): one high-level decision can cover a variable number of primitive environment steps. See [Sutton, Precup, and Singh](../../reference/bibliography.md) for the original formulation.

## Why Options matter here

The project does not turn every HTN method into a learned Option. The correspondence is conceptual and helps define the right decision boundary for MORL-guided planning.

| HTN concept                            | Closest Option concept       | Shared idea                                           | Important difference                                                                                          |
|----------------------------------------|------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Method preconditions                   | Initiation set $\mathcal{I}$ | Both restrict when a high-level choice can start.     | HTN preconditions are symbolic hard checks, not a learned initiation classifier.                              |
| Method decomposition                   | Internal policy $\pi$        | Both specify behavior over multiple steps.            | A method is a task structure; it may contain further planning rather than a single learned policy.            |
| Decomposition completion or replanning | Termination $\beta$          | Both return control to the high level.                | HTN termination follows task semantics and runtime validity, not necessarily a stochastic termination model.  |
| Symbolic effects and observed rewards  | Option outcome               | Both describe consequences of the high-level choice.  | HTN effects are planning predictions; environment rewards are empirical feedback.                             |

The distinction matters: a method should not be called an Option merely because it is hierarchical. The HTN method remains an explicit, designer-authored decomposition checked against the symbolic model. Local applicability guarantees neither successful full decomposition nor success in the real environment.

## Call-and-return interpretation

The proposed MORL-HTN architecture can use a call-and-return-like execution cycle:

```text
1. HTN identifies the current CompoundTask.
2. HTN filters methods whose preconditions do not hold.
3. MORL selects one remaining method.
4. HTN simulates the primitive span and evaluates predicted rewards during planning.
5. Execution supplies the corresponding observed primitive trace.
6. The chosen interval ends at method completion or a recorded interruption.
```

This is the method-completion interpretation of credit assignment. The next
nested child selection does not itself complete the parent method. A protocol
using successive decision points instead must define its own continuation
context. If primitive rewards are $\mathbf{r}_t$ through $\mathbf{r}_{t+k-1}$,
the method-selection return is:

$$
\mathbf{R}_{\mathrm{method}} = \sum_{i=0}^{k-1}\gamma^i\mathbf{r}_{t+i}
$$

Here, $\mathbf{R}_{\mathrm{method}}$ is the vector return assigned to the selected HTN method, $\mathbf{r}_{t+i}$ is the primitive-step reward vector at offset $i$, $k$ counts primitive transitions, and $\gamma \in [0,1]$ discounts delayed outcomes. This expression assumes a unit duration per transition. With elapsed-time discounting, use $\sum_{i=0}^{k-1}\gamma^{u_i}\mathbf r_i$, where $u_i$ is elapsed time before transition $i$, and bootstrap with $\gamma^{\tau}$ for total elapsed duration $\tau$. Predicted and actual traces have their own rewards, end states, and durations.

In the proposed learner, this same credit interval is first evaluated through symbolic simulation during planning, then linked to its observed execution for empirical correction. Nested methods need explicit interval bookkeeping: a parent may aggregate the primitive trace, but must not also add child returns covering those same rewards. Interruptions and returned plan prefixes are partial evidence, not completed-method outcomes. The current runtime does not implement this learning bookkeeping.

For MORL, every term is a reward vector. A preference-conditioned utility is applied only when comparing valid method choices; it must not weaken the HTN precondition mask.

## Relation to Hierarchical Programmatic Options

The Hierarchical Programmatic Option Framework (HIPO) is useful related work because it retrieves interpretable programmatic Options and learns a high-level policy that selects among them. Its high-level decision process also acts over temporally extended, reusable behaviors. [Lin et al.](../../reference/bibliography.md)

The proposed HTN-MORL architecture differs in three key ways:

| HIPO                                                  | Proposed HTN-MORL architecture                                |
|-------------------------------------------------------|---------------------------------------------------------------|
| Retrieves programmatic Options as low-level policies. | Uses designer-authored HTN methods and decompositions.        |
| Learns a high-level policy over retrieved programs.   | Uses MORL to select one symbolically feasible method.         |
| Optimizes a task return.                              | Optimizes a vector return conditioned on current preferences. |

HIPO therefore supports the value of interpretable temporal structure, while this project studies how symbolic feasibility and multi-objective preferences constrain a planning-time choice.

## Design implications

- Define the initiation condition from `Method.preconditions`, not from a separate learned gate.
- Keep the HTN feasible-method mask in training and inference.
- Log the start time, end condition, primitive trace, vector return, and replanning events for each selected method.
- Do not assume a fixed duration: method execution can be interrupted by failure or replanning.
- Evaluate whether longer method intervals improve or worsen credit assignment and adaptation.

See [MORL](morl.md) for the preference-conditioned selection rule and [the symbolic MORL architecture](../../architecture/symbolic-morl.md) for the complete system flow.
