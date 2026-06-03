# Research record — autoresearch #1: backtracking is self-exciting

**Verdict: TEMPORAL (self-exciting / AC).** Investigation record per
[`autoresearch_spec.md`](../autoresearch_spec.md) § 4. Preregistration:
[`backtracking_prereg.md`](backtracking_prereg.md) (frozen, committed before
measurement). All four preregistered predictions were confirmed.

> **Headline.** In chain-of-thought reasoning traces, backtracking is strongly
> **self-exciting**: a backtracking sentence makes the next sentence ~3.6× more
> likely to also be backtracking (`P=0.44` vs base `0.12`), with autocorrelation
> `ACF(1)=0.36` — ~3.5× above both a position-trend null and a count-preserving
> shuffle. The clustering is genuine order-sensitive (AC) structure, not a
> by-product of the (real, but separate) rising-rate trend. A discrete
> self-exciting (logistic-AR / Hawkes-style) generator reproduces the temporal
> signature on held-out traces. This is a real-language anchor for the
> order-sensitive axis the signed-motion bench probes synthetically.

## 1. Labeler + noise floor

- **Data:** 300 R1-Distill-Llama-8B reasoning traces (math/logic), 25,528
  sentences, per-sentence `is_backtracking` from a Sonnet-4.6 judge
  (`results/c7_backtracking/stage_a/`, ported from `origin/final`). Base rate
  **0.124**. Unit of time = sentence index; an event = a backtracking sentence.
- **Noise:** `validation.json` reports keyword-vs-judge F1 ≈ 0.34 — the
  labeling is genuinely ambiguous. **Carried as a caveat and tested** (§ 4):
  the effect survives substantial independent label noise.

## 2. Measured temporal signature (vs null models)

Nulls: **N1** within-trace permutation (preserves count, kills order); **N2**
inhomogeneous Poisson at the empirical per-position rate (preserves the trend,
kills clustering); **N3** homogeneous Poisson at base rate.

| statistic | real | N1 (permute) | N2 (trend-only) | N3 (homog) |
|---|---|---|---|---|
| ACF(1) | **0.362** [0.31, 0.41] | 0.101 | 0.020 | 0.000 |
| Fano (w=10) | **2.74** [2.39, 3.08] | 1.66 | 1.03 | 0.87 |
| self-excitation ratio `P(1\|1)/base` | **3.58** [3.29, 3.86] | — | 1.15 | — |
| inter-event gap CV | **1.58** [1.48, 1.68] | — | — | 0.92 (geometric) |

- `P(backtrack | prev backtrack) = 0.445` vs `P(· | prev not) = 0.081` (base 0.124).
- **Position trend:** rate rises 0.015 → 0.144 (first → last bin) — real, but
  N2 isolates it, and real ≫ N2, so the self-excitation is *on top of* the trend.
- **Markov order ≥ 2:** order-1≫0 (`p≈0`), order-2≫1 (`p=1.6e-141`); MI(lag)
  decays from 0.047 nats over ~6 sentences.

Figure: [`figs/backtracking_signature.png`](figs/backtracking_signature.png)
(ACF real-vs-nulls; position trend; MI vs lag).
Raw numbers: `backtracking_stats.json`.

### 2b. Sub-finding: self-excitation is domain-dependent

The excitation is present in all 10 problem categories but varies ~5× across
them (30 traces each, so treat as indicative — no per-category CIs):

| category | excite ratio | ACF(1) | Fano |
|---|---|---|---|
| sequences | 7.02 | 0.54 | 3.82 |
| number_theory | 5.47 | 0.30 | 3.22 |
| geometry | 4.10 | 0.37 | 2.90 |
| set_theory | 3.58 | 0.51 | 3.18 |
| … | … | … | … |
| probability | 2.13 | 0.24 | 2.02 |
| **inequalities** | **1.37** | **0.04** | **0.91** |

Pattern-exploration domains (sequences, number theory, geometry) show the
strongest clustering; **`inequalities` is essentially memoryless** (Fano ≈ 1,
ACF(1) ≈ 0 — a near-Poisson process). So "backtracking is self-exciting" is a
property of *exploratory* reasoning, not a universal of CoT — a clean internal
contrast (and a hint at what the self-excitation tracks: abandoned/restarted
lines of search).

## 3. Temporal verdict

Real exceeds **N2 (trend-only)** by ~18× on ACF(1) (0.362 vs 0.020, N2 95% hi
0.032) and exceeds **N1 (count-preserving shuffle)** by ~3.5×. The clustering
is therefore **genuine self-excitation**, not explained by the position trend
*or* by trace-level rate heterogeneity. → proceed to mirror (spec stage 4).

## 4. Controls passed (spec § 3, real-side)

- **Shuffle control:** real ≫ N1, N2, N3 on every statistic (above).
- **Label-noise robustness:** under independent symmetric flips, ACF(1) =
  0.36 → 0.24 (ε=0.05) → 0.15 (ε=0.10) and excite-ratio 3.6 → 2.3 → 1.6 — both
  attenuate (as independent noise must) but stay well above the N2 baseline.
  The effect is not a label-noise artifact.
- **Held-out:** the mirror validation (§ 5) is on a 90-trace held-out split.
- **Trend-vs-excitation disentangled:** N2 is the dedicated control for this
  and the effect survives it.

## 5. Synthetic mirror (discrete self-exciting) + validation

Model fit on 210 train traces: `logit P(b_i=1) = a + c·pos_i + Σ_l w_l·b_{i-l}`
(K=8 lags). Position coefficient `+0.49` (rising rate); self-excitation kernel
`w = [1.69, 0.91, 0.00, 0.70, 0.13, 0.17, 0.05, 0.47]` — strongest at lag 1,
decaying with a multi-sentence tail (consistent with the order ≥ 2 finding).

Held-out validation (real eval traces vs synthetic draws of matched lengths):

| statistic | real (held-out) | synthetic mirror |
|---|---|---|
| base rate | 0.120 | 0.127 |
| self-excitation ratio | 3.20 | 3.50 |
| Fano | 2.41 | 2.91 |
| inter-event gap CV | 1.48 | 1.64 |
| ACF(1) | 0.297 | 0.360 |
| ACF(2) | 0.216 | 0.299 |

Mean `|ACF_real − ACF_syn|` over lags 1–5 = **0.053**. The mirror reproduces
the signature (weak validation, spec § 2.5 — passed); it is *slightly
over-excited* (independent sampling compounds the self-excitation). Figure:
[`figs/backtracking_mirror.png`](figs/backtracking_mirror.png).

## 6. Caveats (honest scope)

- **LLM-judge labels** (subjective; keyword-vs-judge F1 ≈ 0.34). Mitigated by
  the noise-robustness control, but the absolute rates carry this uncertainty.
- **One model, one domain:** R1-Distill-Llama-8B on math/logic prompts. This
  is a finding about *this* reasoning data, not language at large.
- **Mirror is weak-validated** (reproduces the matched statistics), not
  strong-validated (we did not show a dictionary trained on real vs synthetic
  behaves identically — spec § 2.5 strong form).
- **No architecture benchmark yet** (spec stage 6): we have *not* tested
  whether a temporal architecture exploits this self-excitation better than a
  per-token baseline. That is the downstream payoff.

## 7. What this buys us + next

This is the first real-language property the loop has shown to carry genuine
**order-sensitive** temporal structure — the same axis the signed-motion bench
([`ac_signed_motion_bench.md`](ac_signed_motion_bench.md)) probes synthetically,
now anchored in data. The fitted self-exciting generator is a ready
benchmark-datasource: the natural **stage-6** experiment is to embed it under
[`synthetic_benchmark_guidance.md`](synthetic_benchmark_guidance.md) and ask
whether a window/temporal encoder recovers the self-excitation (predict the
next-sentence event from history) better than a per-token encoder — with the
same memorization-free controls. That closes the loop from "real property →
synthetic mirror → architecture test".

## 8. Reproduction

```bash
cd purified/
.venv/bin/python -m experiments.autoresearch.backtracking         # measure + nulls + figure
.venv/bin/python -m experiments.autoresearch.backtracking_mirror  # fit + validate mirror
```
Outputs: `docs/autoresearch/backtracking_stats.json`,
`backtracking_mirror_stats.json`, `figs/backtracking_signature.*`,
`figs/backtracking_mirror.*`. Inputs: `results/c7_backtracking/stage_a/`
(read-only Ward Stage-A labels). Deterministic (`SEED=0`).
