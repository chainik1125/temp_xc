# Synthetic benchmark conventions

**Scope.** The rules every synthetic benchmark in this framework follows, so
that results are comparable *across architectures* and *across benchmarks*.
Read this before proposing or implementing a new synthetic task; the
checklist in § 8 is the template for a proposal.

These conventions exist because a synthetic benchmark's entire value is that
the ground truth is known exactly. If a design can't state its ground truth
cleanly, or can't be evaluated identically across architectures of different
shapes, it is not a good synthetic benchmark.

---

## 1. State the ground truth exactly

Every benchmark must name two things up front:

1. **Feature directions** — the set of unit directions in `R^{d_in}` the
   activations are built from (the dictionary the data actually uses). Call
   their count **`F`**. These are what a dictionary model is meant to
   recover, and what `d_sae` is budgeted against (§ 2).

2. **Hidden / dynamical latents** — quantities that govern the data's
   structure or dynamics but are **not directions**: discrete states, signs,
   phases, chain occupancies, continuous parameters, etc. List each, its type
   (categorical / continuous), and its chance and oracle baselines.

Keep these two categories distinct. A latent that is *not* a direction is
**not a feature** and `d_sae` is not "for" it — it is recovered (if at all)
as structure that an architecture's code exposes *on top of* the `F`
features. Conflating "number of features" with "number of distinct patterns
the dynamics produce" is the most common modelling error here: the latter is
a derived property of the process, not ground truth, and must not be used to
size `d_sae`.

---

## 2. Capacity: equal across architectures, anchored on `F`, swept

- **Equal across architectures.** `d_sae` (dictionary size) and `k_pos`
  (atoms allowed to fire per token) take the **same value for every
  architecture** at each grid point. Per-architecture capacity is forbidden —
  it hands the experimenter a knob to rig the comparison.

- **Anchored on `F`.** Sweep `d_sae` relative to the ground-truth feature
  count `F` — e.g. points below, at, and above `F` — and **mark `F` on the
  axis**. Do not anchor on pattern/window counts or any other derived
  quantity.

- **Swept, not pinned.** Neither `d_sae` nor `k_pos` is chosen as a single
  "fair" value — there is no canonical fair point, because architectures that
  target different-sized structures sit at different over-completeness ratios
  at any fixed width. Instead sweep both as axes and report recovery as a
  **function of capacity** (§ 6). A single operating point may be quoted only
  as one labeled slice of the sweep.

Rationale: any single capacity point privileges one regime; sweeping removes
that degree of freedom and turns "where does an architecture's recovery
switch on?" into a reported finding rather than a hidden choice.

---

## 3. Per-token sparsity normalization

- `k_pos` is *atoms fired per token* — the same unit for every architecture,
  so **equal `k_pos` is well-defined fairness** (unlike `d_sae`, which is only
  meaningful relative to `F`).

- For a window architecture with window length `T`, the window-level sparsity
  budget is **`k_win = k_pos · T`**. Holding `k_pos` equal therefore holds the
  per-token activation budget equal across all architectures and all window
  sizes; the window budget scales with `T` automatically. The only remaining
  difference is that a window architecture may *allocate* that budget jointly
  across time rather than per-token — which is exactly the architectural
  degree of freedom under test.

- Keep **`d_sae ≥ k_pos · max(T)`** so no cell clips. The budget is
  `k_win = min(k_pos·T, d_sae)`; a clipped cell silently drops below the
  intended per-token rate. If a clipped corner is unavoidable, label it.

---

## 4. Window size and the apples-to-apples evaluation window

Architectures consume different window lengths `T` (a per-token model is the
`T = 1` special case). To compare them on identical data:

- **Fix one evaluation window length `L = 2^k`.** Constrain architecture
  windows to **powers of two**, `T = 2^j ≤ L`.

- **Why powers of two.** Any `2^j ≤ 2^k` divides `2^k`, so every architecture
  tiles the *same* `L` with no remainder, and *any subset* of architectures is
  mutually comparable against one fixed `L` — no per-comparison least-common-
  multiple, no remainder handling.

- **Tile, don't slide.** Sample `L`-windows at **random offsets** from the
  data, then partition each into `L/T` **non-overlapping** sub-windows. Every
  metric (reconstruction, probes) is aggregated over the **identical `L`
  positions**. This secures the fairness invariant: *each evaluated position
  is encoded/reconstructed exactly once by each architecture.* Overlapping /
  sliding windows are not used — per-position coverage would scale with `T`,
  re-weighting the comparison.

- **Sweep `T`** for temporal architectures (e.g. `T ∈ {2, 4, 8}`) to expose
  how window size affects recovery. `T` is baked into a model's weights, so
  each `T` is a separately trained model evaluated against the common `L`.

---

## 5. Metrics

- **Feature recovery** (for feature *directions*): best-matching `|cosine|`
  of the decoder atoms against each ground-truth direction, thresholded to an
  AUC. Report each named direction set separately (e.g. local vs global
  directions) — never pool them.

- **Latent recovery** (for hidden/dynamical latents that are *not*
  directions): a **linear probe** on the model's codes over the `L`-window —
  logistic regression for categorical latents, linear regression for
  continuous. **Linearity is mandatory and load-bearing**: it measures what
  is *linearly decodable* from the code, which is the architecturally
  meaningful quantity. A nonlinear (e.g. MLP) probe measures the probe's
  capacity, not the representation's, and is permitted only as an explicit,
  separately-reported ablation. Split by example (train / held-out) so the
  score reflects generalization, not memorization.

- **Reconstruction**: the apples-to-apples windowed NMSE of § 4.

- **Normalize to [chance, oracle]** wherever the baselines are definable, and
  state them. Where an impossibility can be *proven* (e.g. an information-
  theoretic bound ruling out recovery for a class of architectures), state it
  — a provable floor is stronger than an empirical gap. Note the **empirical
  chance floor**: finite-sample linear probes rarely sit exactly at chance, so
  report the observed floor and check that a "win" sits well outside it.

---

## 6. Reporting

- Report recovery as **curves / frontiers** against capacity (`d_sae`,
  `k_pos`) and window size `T`, with the ground-truth feature count `F` marked
  on the `d_sae` axis.
- The headline is the **frontier** — e.g. "the minimum capacity beyond `F` at
  which an architecture recovers latent *X*" — not a single hand-picked cell.
- Because `d_sae`, `k_pos`, and the evaluation windows are identical across
  architectures, any residual difference is attributable to architecture.
  That attribution is the entire point of the benchmark; protect it.

---

## 7. Plumbing (inherited framework rules)

- Every result goes through the canonical runner; rows are code-version
  stamped; the evaluator re-materializes the ground truth with the **training
  seed** so feature directions and latents match what the model was trained
  on. See `CLAUDE.md` (hard rules) and `framework_v2.md`.
- A new benchmark is a **plugin**: a generator + a `configs/data.yaml`
  datasource entry + (if a new metric is needed) an evaluator addition.
  Never edit `temp_bench/core/`.

---

## 8. Checklist for proposing a new synthetic benchmark

1. **Ground truth.** State `F` (number of feature directions) and list every
   hidden/dynamical latent with its type and chance/oracle baselines. Confirm
   nothing is conflated with a derived pattern count.
2. **What it isolates.** Name the axis of behaviour it probes and why the
   existing benchmarks don't already cover it.
3. **Recoverability.** For each latent, state whether/which architectures can
   recover it in principle — ideally with a proof of the chance/oracle bounds.
4. **Capacity grid.** Give the `d_sae` sweep anchored on `F`, the `k_pos`
   sweep, and confirm `d_sae ≥ k_pos · max(T)`.
5. **Windows.** Give `L = 2^k` and the architecture `T ∈ {2^j ≤ L}` sweep.
6. **Metrics.** Which feature-direction sets get cosine-AUC; which latents get
   a linear probe; the reconstruction metric. All over the common `L` tiling.
7. **Predictions.** Preregister expected recovery per architecture across the
   capacity/`T` frontier before running.
