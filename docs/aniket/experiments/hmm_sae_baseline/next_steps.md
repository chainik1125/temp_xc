---
author: aniket
date: 2026-03-22
tags:
  - design
  - in-progress
---

## What "finalize the HMM data generator" means

Dmitry's directive is to make the HMM data generator a reusable piece of shared
infrastructure that the whole team can build on. Right now the generator lives in
`src/v5_hmm_sae_baseline/hmm_data.py` as a self-contained experiment module. It works
correctly — the v5 experiment validated the math across the full $(\lambda, \gamma)$
grid — but it is not consumable by other codebases in the project. Andre's
`temporal_crosscoders/data.py` and Han's `src/v2_temporal_schemeC/markov_data_generation.py`
each have their own data generation code with different interfaces, shape conventions,
and parameterizations. The HMM extension needs to land in a place where all three
pipelines can use it without duplicating the emission logic.

There are three concrete things that need to happen, in order of priority.

---

## 1. Upstream the emission step into `src/data_generation/`

The existing pipeline in `src/data_generation/` produces deterministic emissions:
`generate_support` calls `generate_support_markov` from `src/shared/temporal_support.py`,
which returns the hidden chain states directly as the observed support. The HMM extension
adds one step between the hidden chain and the observed support: sample
$s_t \sim \text{Bernoulli}(p_{z_t})$.

The cleanest way to do this is:

**Add `EmissionConfig` to `src/data_generation/configs.py`.** Two fields: `p_A` (emission
probability in state A, default 0.0) and `p_B` (emission probability in state B,
default 1.0). The defaults recover the current MC behavior, so existing code that
doesn't pass an emission config sees no change.

**Add `emit` to `src/data_generation/support.py`** (or a new `src/data_generation/emission.py`).
This function takes the hidden chain output `(k, T)` and the emission config, samples
Bernoulli emissions, and returns the observed support. The existing `generate_support`
function would be renamed or wrapped: `generate_hidden_states` produces the Markov chain,
`apply_emission` produces the observations.

**Thread the emission config through `DataGenerationConfig` and `generate_dataset`.** The
dataset output dict gains a `hidden_states` key alongside the existing `support` key.
`support` becomes the observed (emitted) support, and `hidden_states` is the raw Markov
chain. This is exactly the convention v5 already uses.

**Update `TransitionConfig.from_reset_process`** to accept a parameter named `q` (stationary
probability of state B) instead of `p`, to avoid collision with $p_A$/$p_B$. The old `p`
parameter should still work but emit a deprecation warning. The mapping is trivial: $q$
replaces $p$ everywhere. The `stationary_on_prob` field stays as-is; it just means "the
probability of being in the emitting state."

This is a backwards-compatible change. The `from_reset_process(lam, p)` call with
default emission config produces exactly the same data as before. The only visible
difference is that `dataset["support"]` is now technically the *emitted* support, but
with $p_A = 0, p_B = 1$ this equals the hidden chain.

**Add theoretical helpers.** `hmm_marginal_sparsity`, `hmm_autocorrelation_amplitude`, and
`hmm_theoretical_autocorrelation` should move from v5 into `src/data_generation/transition.py`
(or a new `src/data_generation/hmm.py`). These are the formulas downstream experiments
need for validation.

**Validation.** The existing `src/data_generation/test.py` smoke test should be extended with
an HMM case: generate data with $p_A = 0.0, p_B = 0.5, q = 0.1$, verify empirical
$\mu$ and empirical autocorrelation match theory. The v5 experiment already does this at
scale; the smoke test just needs a fast version (small $n$, few sequences).

---

## 2. Support per-feature heterogeneity

Han's generator parameterizes each feature independently with its own $(\pi_i, \rho_i)$.
Our current pipeline uses a single `TransitionConfig` (one matrix $P$) for all $k$
features. This is a limitation — real SAE features at different layers have different
persistence timescales, and the experimental designs that matter (Han's spread of
$\rho \in \{0.0, 0.3, 0.5, 0.7, 0.9\}$, testing whether an architecture
differentially exploits high-$\rho$ features) require per-feature control.

The implementation path: `generate_support_markov` in `src/shared/temporal_support.py`
already runs $k$ independent chains, but they all share the same $(\alpha, \beta)$. The
change is to accept vectors $\alpha_i, \beta_i$ of length $k$ instead of scalars. The
inner loop becomes:

```python
support[:, t] = torch.where(
    prev == 1,
    (u < alpha).float(),   # alpha is now (k,), broadcasts against u which is (k,)
    (u < beta).float(),
)
```

This is a one-line change. The `TransitionConfig` dataclass would accept either a 2x2
matrix (broadcast to all features) or a tensor of shape `(k, 2, 2)` (per-feature). A
convenience constructor `TransitionConfig.per_feature(pi, rho)` would take vectors and
build the per-feature transition probabilities using Han's parameterization:
$p_{01}^{(i)} = \pi_i(1 - \rho_i)$, $p_{10}^{(i)} = (1 - \pi_i)(1 - \rho_i)$.

Similarly, `EmissionConfig` could accept per-feature $(p_A^{(i)}, p_B^{(i)})$ vectors.
This is less urgent — no experiment currently needs per-feature emission probabilities —
but the interface should allow it.

The mapping between Han's $(\pi, \rho)$ parameterization and our $(\lambda, p)$ reset
process is: $\rho = 1 - \lambda$, $\pi = p$. They are the same Markov chain, just
different names. The per-feature extension breaks the reset process structure (which
assumes a single $\lambda$ for all features), but that is intentional — heterogeneous
temporal persistence is the point.

---

## 3. Make the generator work with Andre's codebase

Andre's `temporal_crosscoders/data.py` has a different interface from our pipeline. His
`generate_markov_sequences` returns `(num_seqs, T, num_feats)` activations directly
(features as the last dimension, magnitudes already applied). His `CachedDataSource`
pre-generates long chains and caches the result of mapping through a `ToyModel` (the
orthogonal embedding). The SAE and crosscoder draw sliding windows and single tokens
from this cache.

For the crosscoder comparison experiment, there are two options:

**Option A: Add `generate_hmm_sequences` to Andre's `data.py`.** This is the minimal
change. Copy the emission step into a new function that wraps `generate_markov_sequences`,
adding the Bernoulli emission sampling. Register it in `DATASET_GENERATORS` so the
sweep can select it by name. This keeps Andre's codebase self-contained but duplicates
the emission logic.

**Option B: Make Andre's `CachedDataSource` call our pipeline.** Replace
`generate_markov_sequences` with a call to `generate_dataset` from
`src/data_generation/dataset.py` (after the upstream in step 1), reshaping the output
to match Andre's `(num_seqs, T, num_feats)` convention. This is cleaner but requires
Andre's code to import from `src/data_generation/`, which it currently does not
(his codebase is a standalone directory with its own `config.py` and relative imports).

**Recommendation: Option A first, Option B later.** The crosscoder comparison is the
immediate experiment. Getting HMM data into Andre's sweep quickly matters more than
architectural purity. Once the comparison is done and we know the HMM generator is
the right abstraction, we can refactor Andre's code to use the shared pipeline.

The key function to add is:

```python
def generate_hmm_sequences(
    num_seqs: int,
    T: int,
    alpha: float = MARKOV_ALPHA,
    beta: float = MARKOV_BETA,
    p_A: float = 0.0,
    p_B: float = 1.0,
    mu: float = FEAT_MEAN,
    sigma: float = FEAT_STD,
    num_feats: int = NUM_FEATS,
    device: torch.device = DEVICE,
) -> torch.Tensor:
```

This generates the hidden chain via the existing Markov logic, applies the emission step,
multiplies by magnitudes, and returns `(num_seqs, T, num_feats)`. The sweep config would
add emission parameters to the grid alongside the existing `DATASETS` and `SWEEP_K` axes.

---

## What comes after finalization

Once the generator is upstreamed, the next experiment is the one the whole project is
building toward: **SAE vs crosscoder on HMM data across the $(\lambda, \gamma)$ grid.**

The v5 experiment established that the SAE baseline is flat at AUC = 0.944 across all
conditions — it cannot exploit temporal structure because it sees one position at a time.
The prediction for the crosscoder:

- At $\gamma = 0$ (i.i.d. observations), the crosscoder should match the SAE. There is
  no temporal information to exploit regardless of $\lambda$.
- At $\gamma = 1$ (MC case, observations = hidden states), the crosscoder has maximum
  temporal signal. If $\lambda$ is small (high persistence), adjacent positions are
  highly correlated, and the crosscoder should be able to use this — but whether it
  actually does depends on the architecture. Andre's result (SAE dominates TXCDR) suggests
  the shared-latent bottleneck prevents this.
- The interesting regime is intermediate $\gamma$. As $\gamma$ decreases from 1 toward 0,
  the observations carry less information about the hidden state, and the crosscoder's
  potential advantage shrinks. The question is whether the crosscoder's AUC curve as a
  function of $\gamma$ shows a monotonic relationship, and at what $\gamma$ the advantage
  (if any) disappears.

Han's shuffle diagnostic is the methodological template: if you want to know whether an
architecture exploits temporal structure, train the same architecture on shuffled data
and compare. The $\gamma$ sweep is a continuous version of this — $\gamma = 0$ is the
"shuffled" limit where temporal information is absent from the observations, and $\gamma = 1$
is the "unshuffled" limit where observations perfectly reveal the hidden state.

The $\gamma$ parameter is something neither Han nor Andre had. Han varied $\rho$ (our
$1 - \lambda$), which controls how fast the hidden chain forgets. Andre swept $k$ and $T$.
But neither could independently control how much the observations reveal about the hidden
state. This is what $\gamma$ adds, and it is the axis most relevant to Dmitry's question
about whether temporal crosscoders can exploit real LM temporal structure — because in
real models, the "emission noise" (the gap between the latent context and the observed
activation) is substantial and feature-dependent.

## Concrete deliverables

1. PR upstreaming emission config into `src/data_generation/`. Backwards-compatible, with
   smoke test.
2. Per-feature heterogeneity support in `src/shared/temporal_support.py`. This can be a
   separate PR.
3. `generate_hmm_sequences` added to Andre's `temporal_crosscoders/data.py` (on my
   branch, not Andre's). Enough to run the crosscoder comparison.
4. The crosscoder comparison experiment itself: v6, SAE vs TXCDR on HMM data across
   $(\lambda, \gamma)$.
