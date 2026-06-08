# Preregistration — backtracking as a temporal property

**Frozen before measurement** per [`autoresearch_spec.md`](../autoresearch_spec.md)
§ 2.1. Committed prior to running any statistic; not edited afterward (only an
abort/verdict is appended in the record).

Investigation #1 of the temporal-property autoresearch loop.

## Property

**Backtracking / self-correction in chain-of-thought reasoning** — sentences
where the model reverses or reconsiders a prior step ("wait, that's wrong",
"let me reconsider", …).

## Data + labeler (version-pinned)

- **Source:** Ward Stage-A traces — 300 reasoning traces from
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` over 300 math/logic prompts
  (10 categories × 30). `results/c7_backtracking/stage_a/sentence_labels.json`,
  restored from `origin/final` (ported from
  `origin/aniket-ward-stage-b @ a62175ee`; see `ATTRIBUTION.md`).
- **Labeler:** per-sentence `is_backtracking` boolean from a **Sonnet-4.6
  judge**. The unit of time is the **sentence index** within a trace (discrete);
  an "event" is a backtracking sentence.
- **Label-noise indicator:** `validation.json` reports keyword-heuristic vs
  judge agreement at F1 ≈ 0.34 — i.e. the task is genuinely ambiguous. The
  `is_backtracking` labels used here are the judge's (the better of the two),
  but the substantial ambiguity is a **carried caveat**: any temporal effect
  must be large relative to plausible label noise, and we will run a label-
  perturbation robustness check (§ controls).
- **Scope honesty:** this is one reasoning model on math/logic prompts — a
  finding here is about *this* reasoning data, not language in general.

Dataset shape (from verification): 25,528 sentences; base rate p ≈ 0.124;
mean 10.6 backtracks/trace; 255/300 traces have ≥ 2 events (usable for
inter-event statistics).

## Statistics to measure (fixed)

On the per-trace binary event sequence `b_1 … b_L` (`b_i = is_backtracking`):

1. base rate; per-**position** rate (is there a trend through the trace?);
2. event-indicator autocorrelation `ACF(k)` vs lag `k`;
3. conditional intensity / self-excitation: `P(b_{i+1}=1 | b_i=1)` and
   `P(event | event within last w)` vs base rate;
4. inter-event gap distribution (sentences between consecutive events);
5. burstiness: Fano factor of windowed counts; inter-event-gap coefficient;
6. Markov order (0 vs 1 vs 2) via a likelihood-ratio / conditional-entropy test;
7. mutual information `I(b_i; b_{i+k})` vs `k`.

All with bootstrap CIs over traces; also a per-category breakdown.

## Null models (the temporal-ness gate)

- **N1 — within-trace permutation** (preserves per-trace count, destroys all
  order). Real vs N1 = *total* temporal structure.
- **N2 — inhomogeneous Poisson** with the empirical per-position rate
  (preserves the position-trend, destroys clustering). **Real vs N2 = genuine
  self-excitation beyond the trend** — the key discriminator.
- **N3 — homogeneous Poisson / geometric gaps** at the base rate (preserves
  nothing).

## Predictions (preregistered, with confidence + reasoning)

- **P1 (high):** backtracking is **not** a homogeneous Poisson process —
  `ACF(1) > 0`, `Fano > 1`, inter-event gaps over-dispersed vs geometric.
  *Reason:* corrections plausibly come in clusters (one reconsideration spawns
  another).
- **P2 (moderate-high):** a **position-trend** — backtracking rate rises
  through the trace. *Reason:* errors accumulate / get caught later in
  reasoning.
- **P3 (moderate — the open question):** **genuine self-excitation persists
  beyond the position-trend** (real exceeds N2 at short lags). *Reason:* this
  is the AC/order claim. It may instead turn out that clustering is fully
  explained by the rate drift (real ≈ N2), in which case backtracking is
  temporal only via a slow DC rate-drift, **not** self-excitation — itself a
  clean, reportable finding.
- **P4 (moderate):** Markov order ≥ 1 (`b_{i+1}` depends on `b_i` beyond base
  rate).

## Verdict criteria

- **Temporal (self-exciting / AC):** real departs from **N2** beyond bootstrap
  CI and beyond the label-noise robustness check → proceed to mirror (Hawkes /
  semi-Markov), per spec stages 4–5.
- **Temporal (rate-drift only / DC):** real departs from N1/N3 but **not** N2
  → report "structured by position-trend, not self-excitation"; mirror with an
  inhomogeneous-rate process, not Hawkes.
- **Not temporal:** real ≈ N1 → **abort** with a clean negative.

## Controls required (spec § 3, real-side)

- Shuffle control: the N1/N2/N3 comparisons above.
- Label-noise robustness: re-run the headline statistic under random label
  flips at a rate consistent with the labeling ambiguity; the effect must
  survive.
- Held-out: report on a held-out split of traces (and check stability across
  the split).

## Out of scope for this investigation

- The synthetic mirror's *architecture benchmark* (spec stage 6) is downstream;
  the **measurement + verdict is the primary deliverable**. A mirror is built
  only if the verdict is "temporal".
- No model inference / activations — this is a pure text-label temporal
  analysis.
