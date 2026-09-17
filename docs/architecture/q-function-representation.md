# Representing $Q_\theta$: options beyond a tabular baseline

!!! warning "Research notes — not implemented, not scheduled"
    This page compares possible parameterizations of the method-value
    function $\mathbf{Q}_\theta$ introduced in
    [MORL-guided HTN method selection](symbolic-morl.md). None of the
    options below should be built before that architecture's tabular or MLP
    baseline has been implemented and evaluated, and its limitations
    measured empirically.

## Role in the architecture

[MORL-guided HTN method selection](symbolic-morl.md) defines the decision
rule

$$
M_t^* = \underset{M \in \mathcal{M}_{\mathrm{valid}}(C_t, WS_t)}{\operatorname{arg\,max}}\;
u_{\mathbf{w}_t}\!\left(\mathbf{Q}_{\theta}(C_t, WS_t, M,\mathbf{w}_t)\right)
$$

without committing to a specific parameterization of $\mathbf{Q}_\theta$. A
tabular representation (one entry per discretized $(WS_t, M)$ pair) is the
most direct baseline. It shares two limitations with any fixed-output-head
representation, worth naming explicitly, because every option below is a
different answer to one or both of them:

1. **Fixed action cardinality.** $\mathcal{M}(C_t)$ — the methods declared
   for a `CompoundTask` — has a different size for each `CompoundTask`. A
   tabular or fixed-output-head representation of $\mathbf{Q}_\theta$ needs
   either one table/network per `CompoundTask`, or a padded/masked global
   action space sized to the largest $\mathcal{M}(C_t)$ in the domain.
2. **No generalization across symbolic states.** A tabular
   $\mathbf{Q}_\theta$ only has a value for a $WS_t$ it has seen exactly
   before. Two symbolic states that differ by one irrelevant fact do not
   share anything learned.

## Options

### Masked or method-conditioned ordinary network

An otherwise ordinary MLP — the same kind of network already used by
`Envelope`'s `QNet` — addresses the fixed-action-cardinality limitation
without attention, in one of two algebraically equivalent ways (see
Qi, below):

- **Padded output with a feasibility mask.** Keep a fixed-size output over
  a domain-wide action space (the union of every `method.id` declared
  anywhere in the domain). At each decision point, build a binary mask —
  1 at the indices of $\mathcal{M}_{\mathrm{valid}}(C_t, WS_t)$, 0
  elsewhere — and add a large negative value to the masked-out entries of
  $\mathbf{Q}_\theta$ before $\arg\max$/scalarization.
- **Method identity as input.** Instead of one fixed output per action,
  make the network compute $\mathbf{Q}_\theta(WS_t, M, \mathbf{w}_t)$ from
  $[WS_t, \operatorname{one\_hot}(M)]$ concatenated as input, and call it
  once per $M \in \mathcal{M}_{\mathrm{valid}}(C_t, WS_t)$. The output size
  no longer depends on the size of the domain's action space at all.

Either form reuses an ordinary feed-forward network, is far cheaper to
build than the attention encoder below, and already gives cross-`WorldState`
generalization (the network is shared, so learning at one $(WS_t, M)$
transfers to similar $WS_t$). Neither gives cross-`CompoundTask` interaction
between candidates (a method's value cannot depend on which other methods
happen to be offered alongside it), and, as long as $M$ is a plain one-hot
identity, neither generalizes to a method never seen in training — see
[Deriving a non-opaque method representation](#deriving-a-non-opaque-method-representation)
below for how to lift that specific restriction.

**Precedent:**

- **Padded output with masking — Huang and Ontañón (item 28).** Their study
  of invalid action masking shows two things. First, masking invalid logits
  to a large negative value before softmax/argmax corresponds to a
  mathematically valid policy-gradient update over the restricted set of
  valid actions — it is not an ad hoc correction. Second, they compare it
  empirically against the two obvious alternatives: penalizing an invalid
  choice with negative reward (the agent spends training signal learning
  "not X" instead of the actual task), and resampling until a valid action
  is drawn (wasteful). Masking scales far better than either as the number
  of invalid actions grows, which is the same regime this proposal is in —
  most methods declared anywhere in the domain are infeasible at any given
  decision point.
- **Method identity as input — Schaul, Horgan, Gregor, and Silver (item
  29).** Universal Value Function Approximators solve a differently
  labeled but structurally identical problem: instead of training one value
  function per goal, they condition a single shared network on
  $(state, goal)$, factoring the representation into separate state and
  goal embeddings combined to produce the value. Sharing structure this way
  lets the network generalize to goals seen only partially during
  training. Substituting "method" for "goal" is exactly the
  input-concatenation form above: one network conditioned on
  $[WS_t, \operatorname{one\_hot}(M)]$ instead of one output slot per
  method. This is the historical origin of that pattern, not an
  improvisation invented for this page.
- **Formal equivalence of the two forms — Qi (item 30).** This result shows
  a standard DQN can be written as $Q_\theta(s,a) = a^\top \varphi_\theta(s)$
  when $a$ is a one-hot vector — algebraically, "one output per action" and
  "dot product of a one-hot action with a state embedding" are the same
  computation, and the paper proves this form has universal approximation
  capacity. Choosing the input-concatenation form for engineering
  convenience (no need to size the output to the domain's largest action
  set) costs nothing in representational power relative to the
  padded-output form.

**A masking mistake that silently corrupts training.** The mask must be
applied not only to the action actually selected at inference, but also to
the bootstrap term of the TD target,
$\max_{a'} \mathbf{Q}_\theta(WS_{t+1}, a', \mathbf{w})$. If the next
state's infeasible methods are left unmasked there, the target is computed
from actions that could never actually be chosen at $WS_{t+1}$, and the
learned values are biased upward by however good those phantom actions
look. Masking only the acting/inference policy and forgetting the
bootstrap term is a common and easy-to-miss version of this bug.

### Token-based attention encoder

Represent both $WS_t$ and $\mathcal{M}_{\mathrm{valid}}(C_t, WS_t)$ as
unordered token sets, and use an attention encoder to produce one vector
value per candidate method in a single forward pass:

- **State tokens.** Each fact in $WS_t$ becomes one token: an embedding of
  the fact's key (from the domain's known predicate vocabulary) combined
  with an encoding of its value (a lookup for categorical/boolean facts, a
  linear projection for numeric ones).
- **Method tokens.** Each $M \in \mathcal{M}_{\mathrm{valid}}(C_t, WS_t)$
  becomes one token: an embedding of the method's declared identifier, from
  the domain's known method vocabulary — or the structured representation
  from [Deriving a non-opaque method representation](#deriving-a-non-opaque-method-representation)
  below, in place of that opaque lookup.
- **Encoder.** Self-attention among state tokens, and cross-attention of
  method tokens over state tokens — a small transformer encoder, not a
  pretrained language model.
- **Output head.** A shared head (tied weights) applied to each method
  token's encoded representation, producing
  $\mathbf{Q}_\theta(C_t, WS_t, M, \mathbf{w}_t) \in \mathbb{R}^d$ for that
  method, where $d$ is the number of objectives.

Because the encoder consumes a variable-length set of method tokens and
emits one output per token, $\mathcal{M}_{\mathrm{valid}}(C_t, WS_t)$ can
have any size without changing the architecture, and a single shared model
can in principle serve every `CompoundTask` in the domain. Unlike the
masked/conditioned MLP above, method tokens can attend to each other and to
the full state, so a candidate's value can in principle depend on which
other methods are offered alongside it — the one thing the cheaper option
cannot do.

**Precedent:**

- **Multi-objective Decision Transformers.** Ghanem, Ciblat, and Ghogho's
  multi-objective Decision Transformer for offline RL, and Wang,
  Karatzoglou, Arapakis, and Jose's reward-driven multi-objective Decision
  Transformer for recommendation, both replace a fixed-input value function
  with attention over a tokenized trajectory, conditioned on multiple
  objectives (items 21-22). Neither targets a per-step action set whose
  size changes with the decision point.
- **Permutation-invariant attention over sets.** Lee et al.'s Set
  Transformer is the general architecture behind treating $WS_t$ and
  $\mathcal{M}_{\mathrm{valid}}(C_t, WS_t)$ as unordered token sets rather
  than an artificially ordered sequence (item 23) — this is what would
  justify attention over CNN/RNN for this specific input shape.
- **One value output per action token.** Itaya et al.'s Action
  Q-Transformer is the closest existing precedent for the output head
  above: an encoder-decoder where each action has its own query token, and
  the decoder produces one value per action query (item 24). This page's
  "one method token in, one $\mathbf{Q}_\theta$ vector out" head is the
  same pattern, applied to HTN methods instead of a fixed low-level action
  set.
- **Variable/large discrete action spaces.** SAINT (Landers et al., item
  25) applies attention policies directly to discrete combinatorial action
  spaces; Q-Transformer (Chebotar et al., item 26) instead discretizes and
  handles a large action space autoregressively per dimension; Dulac-Arnold
  et al.'s classic Wolpertinger approach (item 27) embeds actions and
  searches among them with nearest-neighbor lookup instead of attention.
  These are three different answers to the same fixed-action-cardinality
  problem, worth comparing against this design before committing to it.

No existing work applies any of the above to HTN method selection with a
variable, per-decision-point candidate set; that adaptation is the open
part of this option, not something reused from prior work.

### Deriving a non-opaque method representation

Both options above default to treating $M$ as an opaque identity — a
one-hot vector for the masked/conditioned MLP, an embedding looked up from
the method's declared id for the attention encoder — and neither
generalizes to a method never seen in training as a result. Both can be
extended the same way, because `Method` already carries structured,
formally defined data that does not depend on how the domain's author
happened to name anything:

- **Parent-task token.** `method.parent_task` (the owning `CompoundTask`)
  as a categorical embedding.
- **Precondition tokens.** Each entry of `method.preconditions`
  (`dict[str, tuple[ConditionOperator, WorldValue]]`,
  `src/htn/tasks/types/preconditions.py`) becomes one token: the fact key
  embedded with the **same table used for $WS_t$'s fact tokens**, the
  operator with its own small embedding (six possible values), and the
  value encoded the same way state-fact values are. Aggregate by sum or
  mean — preconditions form a set, not a sequence, so nothing should depend
  on the order `Method.preconditions` happens to iterate in.
- **Subtask tokens.** Each entry of `Method.tasks`, in declaration order —
  unlike preconditions, this order is semantically meaningful, since it is
  the decomposition sequence.

Sharing the fact-key embedding table between $WS_t$ and precondition tokens
is the important part: it lets $\mathbf{Q}_\theta$ relate "this method
requires $energy\_is\_low = \mathrm{True}$" to "the current state has
$energy\_is\_low = \mathrm{True}$" directly, instead of learning that
association from scratch behind an opaque id. The resulting vector can
replace the one-hot input to the masked/conditioned MLP, or the looked-up
embedding of a method token in the attention encoder — it is a
representation choice, not tied to either option above.

This only captures **applicability** — when the method is chosen, not what
it does once decomposed. Two methods can share identical preconditions and
have unrelated `tasks`. A behavioral signal would need the effects of the
`PrimitiveTask`s actually reached by decomposing the method, which live on
those primitives, not on `Method` itself — gathering them means the same
recursive walk down `Method.tasks` (expanding any `CompoundTask` through
its own methods) that method-level reward accumulation already needs (see
[MORL-guided HTN method selection](symbolic-morl.md)). Building the
representation and accumulating the reward could share that traversal
instead of implementing it twice.

**A cheaper, weaker fallback:** tokenizing the method's id/name string
itself (e.g. splitting `reach_goal.recharge_first` on `.`/`_` and embedding
the pieces) costs almost nothing and gives partial overlap between
similarly-named methods for free. Unlike the structured tokens above, it
depends entirely on the domain author's naming discipline: `recharge_first`
and `recharge_last` share a token by luck of vocabulary, not because
anything about their formal definition was compared, and nothing prevents
unrelated methods from colliding on a shared word. Prefer the
preconditions/parent-task/subtask tokens above when the encoder for $WS_t$
already exists, since the marginal cost of reusing its embedding table for
preconditions is low; treat id tokenization as a fallback only if that
encoder does not exist yet.

**Precedent:** Jain, Szot, and Lim (item 31) train a policy over actions
represented by embeddings inferred from each action's own observed effect
on the environment — rather than an opaque action id — so that a
never-seen action can be plugged in at test time once its representation is
inferred, with no retraining. Alchihabi, Zhang, and Guo (item 32) refine
this to require far fewer observations of the new action to infer its
representation. Neither paper uses a formally declared definition the way
`Method`'s preconditions and subtasks are used above — they infer the
representation from interaction data because their actions have no such
declaration — but the underlying claim is the same one this section relies
on: an action/method representation derived from what it *does or requires*,
rather than from an arbitrary identity, is what allows generalization to
one never seen in training. The relational graph encoder already proposed
in [preference-weight generation](preference-weight-generation.md) — nodes
for predicates, resources, and methods; edges for "a method requires a
predicate" or "a method changes a predicate" — is the same idea again,
proposed independently for a different purpose (inferring preference
weights) before this page connected it to method representation.

## Comparison

| | Tabular baseline | Masked / conditioned MLP | Token-based attention |
|---|---|---|---|
| Fixed action cardinality per `CompoundTask` | No — needs one table per `CompoundTask` | Yes | Yes |
| Generalization across similar $WS_t$ | No | Yes | Yes, plus generalization across facts |
| Cross-candidate interaction | No | No | Yes |
| Generalization to methods never seen in training | No | Not by default ($M$ is an opaque one-hot identity) — possible if the input is replaced with the structured token below | Not by default (same reason) — possible if the method token is built the same way |
| Engineering cost | Lowest | Low — reuses an ordinary network | Highest — new architecture and training loop |

## Sequencing

Do not implement any option beyond the tabular baseline before
[MORL-guided HTN method selection](symbolic-morl.md)'s baseline is running
and has produced evidence, from real training scenarios, that:

- the number of methods per `CompoundTask` varies enough, and often enough,
  that one table/network per `CompoundTask` is a real engineering burden;
  or
- $WS_t$ recurs too rarely across episodes for the tabular representation
  to accumulate reusable experience for most decision points visited.

If neither holds, moving past the tabular baseline has no measured
justification. If one does, try the masked/conditioned MLP first — it
removes the same limitations at a fraction of the engineering cost of
attention. Reach for token-based attention only once there is a concrete
need for candidates to influence each other's value, or for method tokens
to carry more than an opaque identity — neither of which the masked/
conditioned MLP can do, and neither of which this page currently specifies
how to build.

## Related pages

See [MORL-guided HTN method selection](symbolic-morl.md) for the value
function these options parameterize, and
[preference-weight generation](preference-weight-generation.md) for how
$\mathbf{w}_t$ is supplied to $u_{\mathbf{w}_t}$. See the
[bibliography](../reference/bibliography.md) for full citations of the
numbered items above.
