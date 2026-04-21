# Why R²_max is the ceiling — detailed notes

This doc explains, carefully, what `R²_max` means in the
`separation_scaling` plots and *why* a probe can't beat it, no matter how
powerful.

The confusion is worth unpacking because the usual intuition — "if I had
a perfect classifier I could get R² = 1" — is **wrong for this setup**,
and the reason is not obvious until you write down the loss.

---

## 1. What we're trying to do

Per sweep cell:

- The generator commits each sequence to one of 3 components C ∈ {0, 1, 2},
  sampled uniformly at the start.
- Given C, it emits a token sequence X_1, ..., X_T via the corresponding
  mess3 HMM.
- We train a transformer next-token-predict on mixtures of these sequences.
- At each position t of each sequence we get a residual activation
  `f_t ∈ R^{d_model}` (d_model = 64).
- We train a **probe** (linear or MLP or logistic) to recover C from f_t.

The target is `one-hot(C) ∈ {0, 1}^3` — a one-hot vector. The probe outputs
a vector `g(f_t) ∈ R^3`.

We measure recovery with **R²**:

```text
R² = 1 − MSE(g, one-hot(C)) / Var(one-hot(C))
```

where both MSE and Var are computed per-component and then averaged.

---

## 2. The key fact: the target is noisy

Here's the crucial — and easily missed — point:

> **The target one-hot(C) is random given X, and its conditional
> distribution is exactly the Bayes posterior P(C | X).**

In words: "If I tell you the activation X, you still don't know C with
certainty. You know a distribution P(C | X) over components, and the
observed one-hot is a *single draw* from that distribution."

Formally:

```text
one-hot(C) | X = x  ~  Categorical(P(C | X = x))
```

so

```text
E[one-hot(C) | X = x]  =  P(C | X = x)         (the posterior, element-wise)
Var(one-hot(C)_c | X)  =  P(C=c|X) · (1 − P(C=c|X))    (Bernoulli variance)
```

This is the source of the irreducible noise. Even if you knew the
*exact posterior* P(C | X), you would still make errors predicting each
individual one-hot sample, because the target is stochastic.

---

## 3. The decomposition

For any probe function `g` (linear, MLP, anything):

```text
MSE(g, one-hot(C))  =  E_X E_C|X [(g(X) - one-hot(C))²]
                   =  E_X [(g(X) - P(C|X))²]          ← bias² (choice of g)
                    + E_X [Var(one-hot(C) | X)]         ← irreducible
```

The first term depends on `g`. The second term **does not**. You can't
make the second term smaller by choosing a better probe.

This is the standard bias/variance decomposition but applied to a
regression with a **stochastic** target.

### The Bayes-optimal probe

If `g(X) = P(C | X)` (exactly the posterior), the bias term is zero
and the MSE equals the irreducible piece:

```text
MSE* = E_X [Var(one-hot(C) | X)]
     = Σ_c E_X [P(C=c|X) · (1 − P(C=c|X))]     (per-component)
```

This is the **smallest** MSE any function can achieve. No network, no
sparse code, no window width changes it — it's a property of the data
distribution, not the probe.

---

## 4. Converting to R²

R² is defined by normalizing MSE against the variance of the target:

```text
R² = 1 − MSE / Var(one-hot(C))
```

For a uniform prior over 3 components, `Var(one-hot(C)_c) = (1/3)(2/3) = 2/9`
for each component independently. So the ceiling is:

```text
R²_max  =  1  −  MSE* / Var(one-hot(C))
        =  1  −  E_X [P(C|X)(1 − P(C|X))] / [p·(1−p)]       (p = 1/3)
```

**R²_max is a function of the generator and X alone.** It's the best
any probe can achieve on this data under an MSE loss against one-hots.

If a probe hits R²_max, it has exactly recovered the posterior in L2
geometry. If it falls short, the gap `R²_max − R²_probe` tells you how
much posterior information the probe is leaving on the table.

---

## 5. Why this breaks the usual "perfect classifier = R² 1" intuition

The usual intuition comes from regression with a *deterministic* target
— y = f(x) + ε where ε has mean zero and variance σ². There, a perfect
regressor gets MSE = σ², and R² → 1 only if the noise is zero.

Here, the target *isn't* `y = f(x) + noise`. It's a **categorical
sample**, and its conditional distribution is directly what we want to
recover. There's no separate "noise" to drive to zero; the noise
structure IS the posterior.

So:

- **If the posterior is a delta** (`P(C|X)` one-hot on the true C for
  every X), then `P(C|X)(1−P(C|X)) = 0` everywhere → R²_max = 1. This
  happens only when X is perfectly class-diagnostic.
- **If the posterior is uniform** (`P(C|X) = (1/3, 1/3, 1/3)` for every
  X), then `P(C|X)(1−P(C|X)) = 2/9` everywhere → R²_max = 0. X contains
  zero information about C.
- **In between**, R²_max sits strictly in (0, 1) and depends on how
  concentrated the posterior is, averaged over X.

The key sentence: **R² can never reach 1 unless the posterior is
deterministic.** Component identity in our setup is only recoverable
probabilistically, so there's always some residual posterior spread →
R²_max < 1.

---

## 6. Worked example (2 classes, closed form)

To sanity-check the intuition, let's do a clean 2-class example.

Suppose `C ∈ {0, 1}` with uniform prior. X is a random variable such that
`P(C=1 | X) = q(X)` is known — a single scalar for each x.

Target: one-hot(C) ∈ {(1,0), (0,1)}. Write it as a scalar `y = C`.

```text
MSE*   =  E_X[q(X)(1 − q(X))]               (Bernoulli variance)
Var(y) =  p(1−p) = 1/4
R²_max =  1 − 4 · E_X[q(X)(1 − q(X))]
       =  1 − 4 · E[q(1−q)]
```

Check three cases:

| posterior                          | q(1−q) | R²_max |
|------------------------------------|-------:|-------:|
| q(X) = 1 or 0 always (deterministic) |    0.0 |   1.00 |
| q(X) = 0.9 always (confident)       |   0.09 |   0.64 |
| q(X) = 0.7 always (mild)            |   0.21 |   0.16 |
| q(X) = 0.5 always (uninformative)   |   0.25 |   0.00 |

Even when the probe *perfectly* knows q(X) = 0.7, it can only get R² =
0.16. Why? Because with q = 0.7, roughly 30% of the time C = 0 instead
of 1, so predicting 0.7 still misses by ≥ 0.3 systematically. The
irreducible Bernoulli variance caps you at R²_max.

---

## 7. Numerical values on our sweep

Computed from the exact Bayesian forward filter on each δ-cell generator
(no transformer, pure data distribution):

| δ    | τ (bits) | R²_max (mean over t) | R²_max (t=T−1) |
|------|---------:|--------------------:|---------------:|
| 0.00 |    0.00  |               0.00  |          0.00  |
| 0.05 |    0.12  |               0.07  |          0.12  |
| 0.10 |    0.37  |               0.22  |          0.36  |
| 0.15 |    0.56  |               0.36  |          0.51  |
| 0.20 |    0.60  |               0.45  |          0.54  |

Per-component at δ=0.20 and t=T−1:

- comp 0: R²_max = 0.998 (the near-deterministic-emission component; posterior almost a delta after 128 tokens)
- comp 1: R²_max = 0.309
- comp 2: R²_max = 0.310

Two different ceilings, two different interpretations:

- **Mean-over-t**: the average R²_max over all 128 token positions. This
  is the correct comparison for our probes, which ingest activations at
  every position with equal weight.
- **t=T−1**: the ceiling for a probe that *only* reads the final-position
  activation (where the Bayesian posterior is maximally concentrated).

Our best position-agnostic probe (window-30 linear, mean over positions)
gets R² = 0.42 at δ=0.20. This is **0.03 below** the correct ceiling of
0.45 — not 0.15 below, which is what the τ=0.60 line on the earlier plot
implied.

---

## 8. Why τ is not R²_max

τ and R²_max are both "fraction of something explained by X", but:

- τ uses entropy: `H(one-hot) = −Σ p log p`.
- R²_max uses variance: `Var(one-hot_c) = p(1−p)`.

Both are concave in p and peaked at p = 1/K (uniform). But entropy has a
steeper curvature than variance near the edges: a posterior like
(0.9, 0.05, 0.05) has:

- Entropy ≈ 0.57 nats (versus log 3 ≈ 1.10 → 52% resolved → τ ≈ 0.48)
- Per-component variance: 0.9·0.1 + 2·(0.05·0.95) = 0.185; comparing to
  uniform variance (2/9 ≈ 0.222), R² component = 1 − 0.185/0.222 = 0.17

So for this posterior, τ ≈ 0.48 but R² ≈ 0.17. They're not equal because
the two functions weight "how much the posterior deviates from uniform"
differently. τ rewards sharp peaks more aggressively than R² does.

This is why the earlier τ line on the plot (at 0.60 for δ=0.20) looked
so much higher than the probe R²s — it was the *entropy* ceiling, not
the R² ceiling. The R² ceiling is 0.45, and our probes are right against
it.

---

## 9. Practical consequences

### What the probe is actually telling us

When we train a linear probe on residuals and get R² = 0.42 at δ=0.20,
that means:

- The transformer's residual contains a *linear approximation of the Bayesian
  posterior* that captures 94% of the available variance (0.42 / 0.45).
- Training a better linear probe (bigger window, more features) can close
  the remaining 6% gap.
- Training a *nonlinear* probe (MLP, logistic) could push closer to
  R²_max by fitting the posterior's shape rather than just its linear
  projection.
- Training a **position-aware** probe could reach R² = 0.54 (the
  final-position ceiling) — the probe gets to use all 128 tokens of
  evidence at once, whereas mean-over-t averages across weak early
  positions and strong late positions.

### What it does **not** tell us

- "The probe isn't working well" — at R² = 0.42 vs ceiling 0.45 the
  probe is performing near optimally.
- "There's lots of information the transformer is hiding" — no, the
  residual has roughly as much linearly-extractable posterior info as
  exists in the sequence up to that point.
- "MatTXC is a stronger representation than raw residuals" — no; its
  linear-on-latents probe hits 0.45 precisely because the latents are a
  window-60 integration of the residuals, and R²_max integrates all
  positions.

### Why regression at all

Because `E[one-hot(C) | X] = P(C | X)` and MSE loss with that target
directly finds the L2-best approximation of the posterior. The Bayes
floor in R²-units (R²_max) has a clean closed form from the generator.
Logistic regression would measure the same posterior approximation
quality in KL-units (with floor H(C | X) in nats), which is a different
but equally valid measuring stick — see `plot_probe_window_sweep.py`'s
logistic panel.

---

## 10. The relationship in one sentence

> `R²_max` is the maximum fraction of `Var(one-hot(C))` that any function
> `g(X)` can explain, and it equals `1 − E[Var(one-hot(C) | X)] / Var(one-hot(C))`.
> It's reached exactly when `g(X) = P(C | X)` — i.e., when the probe
> reproduces the Bayes posterior — because the remaining variance is
> irreducible noise in the one-hot target given X.

Everything else in this doc is unpacking that one sentence.
