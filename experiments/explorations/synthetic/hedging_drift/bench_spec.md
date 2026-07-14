# Synthetic benchmark spec — hedging drift (DC confidence persistence)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the loop) —
frozen by the expansion Cycle-1 PROCEED, staged for a *later, deliberately
blind* B×A evaluation. No architecture has seen or touched this spec.

**Provenance.** Expansion-loop Cycle 1 (autonomous, gated): prereg card
[`../expansion/prereg/uncertainty-hedging-drift.md`](../expansion/prereg/uncertainty-hedging-drift.md)
(frozen 2026-07-13 before any data), calibration record
[`../expansion/records/uncertainty-hedging-drift/calibration.md`](../expansion/records/uncertainty-hedging-drift/calibration.md)
— verdict **PROCEED**: the strongest, most stable signal of the cycle
(split-half 0.319 / 0.313), survived the full null battery, the noise floor,
and all five skeptic kill-items. First grounded benchmark on the
**DC-slow-drift** axis (reasoning-trace domain).

## 1. What it tests

Whether a dictionary code linearly exposes a **slowly-drifting scalar state**
— the reasoner's expressed confidence, which persists across sentences and
trends from hedged (early) to committed (late). Measured on real R1-Distill
traces: ordinal-confidence ACF(1) = **0.316** [0.298, 0.338] vs N1
within-trace permutation hi **0.078** and N2 trend-preserving hi **0.030** —
i.e. the persistence is genuine within-trace order structure *on top of* the
(also real) position trend (per-position mean confidence 0.68 → 0.97).
Noise-robust at ε̂ = 0.113 (perturbed ACF(1) = 0.255). Inter-judge κ = 0.636
(the best-validated labeler of the cycle).

The probe asks: can a linear reader recover the *current confidence state*
(and its local drift) from the code — a per-token code sees hedge/commit
lexicon; carrying the persistent state across sentences is the architectural
degree of freedom under test. This anchors, in real language, the DC axis the
coupling/denoising synthetic benches probe abstractly.

## 2. Generative process (two layers)

### Layer 1 — AR(1) around a linear drift (the fitted mirror)

Per sequence of length `L`, a scalar confidence stream (fit on 210 train
traces, validated on 90 held-out):

```
c_i = mu + beta·(i/L) + r_i ,   r_i = rho·r_{i-1} + sigma·eta_i
mu = 0.781   beta = +0.236   rho = 0.299   sigma = 0.556
```

(full precision in [`mirror_params.json`](mirror_params.json)). **Matched:**
the lag-1 persistence and the drift trend of the ordinal. **Deliberately NOT
matched:** the semantic coupling between confidence and actual solution
correctness; and — a documented fidelity limit — the real stream's
**slow-decay tail** (real ACF stays ≈ 0.12–0.15 through lag 8; AR(1)
collapses by lag 2–3). The skeptic pass explicitly noted this under-fit means
a benchmark on this mirror tests more than the inserted lag-1 moment; a
Cycle-2 refinement may upgrade the mirror to a longer-memory process from the
menu if the tail matters to the eval.

### Layer 2 — emission into activations

Sentence `i` → activation `x_i ∈ R^{d_in}`, the standard emission pattern:

```
x_i = c_i · m · u_conf  +  Σ_{j ∈ content_i} m_j · u_j  +  σ · ε_i
```

- `u_conf`: one confidence feature whose *magnitude* carries `c_i`
  (continuous-loading, like the frequency bench's tone loading);
  alternatively a 3-level one-hot variant (`u_hedge/u_neutral/u_commit`,
  quantizing `c_i`) may be specified as an ablation — the run-time datasource
  plugin pins one as primary. Defaults follow the program conventions
  (`d_in = 64`; `K_c` content features sized so `F = 20`).
- `content_i`: sparse random content subset; `m, m_j`: folded-normal
  magnitudes; `σ`: noise.

## 3. Ground truth (stated exactly, Part II § 1)

1. **Feature directions (`F`):** the confidence feature(s) + content
   features — what `d_sae` is budgeted against.
2. **Hidden/dynamical latents (not directions):**
   - the confidence state `c_i` (continuous; linear-probe target; chance =
     predicting the pooled mean, oracle = the generating `c_i` itself, R²=1);
   - the drift rate / local trend (slow DC component);
   - persistence: next-state prediction above the marginal requires carrying
     `r_i` across sentences (the AR component, ρ = 0.3).

## 4. Capacity / windows / metrics (Part II conventions, uniform design)

- `d_sae ∈ {F//2, F, 2F}` anchored on `F`; `k_pos ∈ {1,2,4,8,16}`;
  dict-feasibility `d_sae ≥ k_pos·T` for pooled families; seeds {1,2,42};
  untrained-encoder control mandatory.
- `L = 32`, `T ∈ {2,4,8}` tiled; per-token sparsity normalized.
- Metrics: cosine-AUC on named direction sets; **linear** (ridge) probe for
  `c_i` over the `L`-window (R², example-split); windowed NMSE; normalized to
  the stated [chance, oracle].

## 5. Predictions (frozen in the card, before any run)

- **per-token SAE:** captures hedge/commit lexicon per token; misses the slow
  drift (no cross-sentence state).
- **window families:** medium windows (T = 8+) best capture the persistence;
  very short windows lose the drift — confirming the DC character (cf. the
  DC/AC lens, `docs/ideas/frequency_lens.md`).

## 6. Caveats carried from calibration

- Heuristic cross-check is asymmetric (hedge-class F1 0.82, commit-class
  0.26) — the judge's "committed" notion is broader than the lexicon's; the
  inter-judge floor (κ 0.636, ε̂ 0.113) is the binding validation.
- Mirror under-fits the ACF tail (above) — weak validation only.
- One model, one domain: R1-Distill math/logic traces.

## 7. Reproduction

```bash
.venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate uncertainty-hedging-drift
.venv/bin/python -m experiments.explorations.synthetic.expansion.render_records
```

_Frozen 2026-07-14 by the runpod agent (expansion Cycle 1). The stage-6 grid
must go through the canonical runner with a registered datasource plugin;
nothing here has been run against any architecture._

## Amendment 2026-07-14 — mirror INVALID under gate 8 (Cycle-2 rider; spec provisional)

The Cycle-1 review's **non-fitted-moment mirror gate** (guardrail 8), applied
retroactively with preregistered moment + tolerance (ACF(2), ±0.05 abs —
`expansion/gate8_recheck_c1.py`), **fails the fitted ar1+trend mirror**:
held-out ACF(2) real **0.155** vs synthetic **0.085** (|err| 0.071). The one
preregistered menu-upgrade attempt (`semi_markov`, Appendix-B "heavy-tailed
dwell" row — `expansion/mirror_upgrade_hedging.py`) reproduces ACF(1)
exactly (0.317/0.319) but **also fails ACF(2)** (0.149/0.079).

**Diagnosis (the informative part):** the real confidence stream's held-out
ACF is a **plateau**, not a decay — [0.32, 0.15, 0.13, 0.13, 0.12, 0.14,
0.14, 0.15] over lags 1–8. On top of the fast lag-1 persistence there is a
long-memory component: per-trace confidence *levels* differ (composition,
≈ the N1 band 0.067) **plus** genuine slow within-trace drift (≈ another
0.07). No current Appendix-B process generates either a per-sequence random
level or a slow regime — every menu mirror collapses geometrically by lag 2–3.

**Consequences:**
- The **measurement verdict (TEMPORAL, DC-slow-drift) is unaffected** — it
  gates on real-vs-null, not on any mirror.
- This spec is **provisional (`SPEC*`): not runnable at stage 6** until a
  mirror passes gate 8. Proposed Cycle-3 menu extension (justification
  above): **hierarchical AR(1)** — per-sequence level `μ_j ~ N(μ, τ²)` +
  linear trend + AR(1) residual; the plateau is then ≈ τ²/(τ² + var), which
  this family can match while keeping ρ for lag-1.
- Recorded in `expansion/results/gate8_recheck_cycle1.json` and
  `expansion/results/mirror_upgrade_hedging.json`.
