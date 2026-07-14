# Synthetic benchmark spec — assumption→consequence (directed-grammar recovery)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the loop) —
frozen by the expansion Cycle-1 PROCEED, staged for a *later, deliberately
blind* B×A evaluation. No architecture has seen or touched this spec.

**Provenance.** Expansion-loop Cycle 1 (autonomous, gated): prereg card
[`../expansion/prereg/assumption-then-consequence.md`](../expansion/prereg/assumption-then-consequence.md)
(frozen 2026-07-13 before any data), calibration record
[`../expansion/records/assumption-then-consequence/calibration.md`](../expansion/records/assumption-then-consequence/calibration.md)
— verdict **PROCEED**, survived the full N1/N2/N3 null battery, the labeler
noise floor, and all five adversarial skeptic kill-items. This is the mirror
of a *measured* property of real R1-Distill reasoning traces, exactly as
[`../backtracking/`](../backtracking/) mirrors measured self-excitation.

## 1. What it tests

Whether a dictionary code linearly exposes the **directed grammar** of
reasoning: assumptions/case-splits precede their consequences
(assume-before-derive, never the reverse). Measured on real traces:
P(consequence at t+1 | assumption at t) = **0.463** forward vs **0.353**
time-reversed — directed asymmetry **0.135**, ~7× above the N1 within-trace
permutation band (0.019 hi), noise-robust (0.091 at ε̂ = 0.165), with
higher-order structure (Markov order-2 ≫ 1 ≫ 0, both p ≈ 0).

The probe asks: can a linear reader of the code predict the *next-sentence
discourse state* — and specifically the A→C directed dependency — from the
window's history? A per-token code sees the current state's connectives; the
directed dependency across sentences is the architectural degree of freedom
under test. This is the **AC-order-sensitive** axis on a *grounded* mirror
(vs the abstract signed-motion construct), in the reasoning-trace domain.

## 2. Generative process (two layers)

### Layer 1 — 3-state Markov discourse chain (the fitted mirror)

Per sequence of length `L`, a state stream `s_i ∈ {N, A, C}` (neither /
assumption / consequence) from the transition matrix **fit on 210 train
traces and validated on 90 held-out traces** (see § 5 of the calibration
record; directed asym reproduced 0.129 vs 0.141, non-fitted moments: ACF(1)
0.397 vs 0.411, dwell mean err 0.055, MI err 0.039):

```
P =  [[0.669, 0.118, 0.213],     # from N     pi = [0.382, 0.139, 0.478]
      [0.207, 0.317, 0.476],     # from A  ← the directed A→C edge (0.476)
      [0.189, 0.108, 0.703]]     # from C
```

(`../expansion/records/assumption-then-consequence/calibration_stats.json`
carries full precision.) **Matched:** the forward transition matrix (hence
the directed asymmetry, the marginal, the dwell geometry). **Deliberately NOT
matched:** the semantic validity of the actual derivations; any order-2+
structure (the real stream has it — a documented fidelity limit of this
mirror, carried as a caveat, not silently ignored).

### Layer 2 — emission into activations

Sentence `i` → activation `x_i ∈ R^{d_in}` over a fixed orthonormal
dictionary, exactly the backtracking-bench emission pattern:

```
x_i = m · u_{s_i}  +  Σ_{j ∈ content_i} m_j · u_j  +  σ · ε_i
```

- `u_N, u_A, u_C`: 3 state features (one per discourse state).
- `content_i`: sparse random subset (size `n_c`) of `K_c` content features.
- `m, m_j`: folded-normal magnitudes; `σ`: noise. Defaults at run time follow
  the backtracking datasource conventions (`d_in = 64`, `K_c = 17` ⇒
  **`F = 20`**), to be pinned in the `configs/data.yaml` entry when the
  datasource plugin is added (a later, separate step — no plugin is added by
  this cycle).

## 3. Ground truth (stated exactly, Part II § 1)

1. **Feature directions (`F = 3 + K_c`):** the state features + content
   features. This is what `d_sae` is budgeted against.
2. **Hidden/dynamical latents (not directions):**
   - the categorical state `s_i` (chance = modal marginal ≈ 0.478; oracle =
     1.0 — the state feature is in the span);
   - the **directed next-state dependency**: predict `s_{i+1}` (or the A→C
     indicator) from the code at `≤ i` — chance = marginal transition
     freq; oracle = the Markov conditional (Layer 1 is the data-generating
     process, so its one-step conditional is the Bayes ceiling);
   - the asymmetry itself: a code that supports next-state prediction above
     the marginal must carry the direction.

Pattern/window counts are derived quantities and never size `d_sae`.

## 4. Capacity / windows / metrics (Part II conventions, uniform design)

- `d_sae ∈ {F//2, F, 2F}` anchored on `F`; `k_pos ∈ {1,2,4,8,16}`;
  dict-feasibility `d_sae ≥ k_pos·T` for pooled families; seeds {1,2,42};
  untrained-encoder control mandatory.
- `L = 32`, `T ∈ {2,4,8}` tiled (never slid), per-token sparsity normalized
  (`k_win = k_pos · T`).
- Metrics: cosine-AUC on the named direction sets (state vs content,
  reported separately); linear probe (logistic) over the `L`-window for the
  state latent and the next-state/directed latent, split by example;
  windowed NMSE. Normalized to the stated [chance, oracle].

## 5. Predictions (frozen in the card, before any run)

- **per-token SAE:** captures the A/C connective features per token; blind to
  the directed A→C dependency across sentences.
- **window families (TXC-pre/-post/Stacked/Spectral):** backward-looking /
  position-mixing families should expose the order-sensitivity; additive
  (pre-squash) families predicted weaker on the directed latent (cf. the
  changepoint equality-pattern result).

## 6. Caveats carried from calibration

- Labeler κ = 0.517 (moderate); heuristic cross-check weak (κ 0.21) — the
  judge labels lean on context, though the *definition* has no built-in
  ordering (skeptic item b cleared: content markers, not a required
  preceding label).
- Split-half stability 0.109 / 0.160 — the asymmetry varies ~50% across
  halves while staying ≫ nulls in both.
- Mirror is order-1; the real stream is order-2+. Weak validation only
  (matched + neighboring moments reproduced on held-out traces).

## 7. Reproduction

```bash
.venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate assumption-then-consequence
.venv/bin/python -m experiments.explorations.synthetic.expansion.render_records
```

_Frozen 2026-07-14 by the runpod agent (expansion Cycle 1). The stage-6 grid
must go through the canonical runner with a registered datasource plugin;
nothing here has been run against any architecture._
