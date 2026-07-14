# Preregistration — `<property-name>`  (FROZEN <date>, before any data)

> Copy this card per candidate. **Commit it before touching data.** Later stages
> may only *abort*, never revise it (log dated amendments transparently, like
> backtracking's). One card = one `domain × temporal-class` ledger cell.

## 0. Identity
- **Property:** <one line — the temporal phenomenon>
- **Ledger cell:** domain = `reasoning-trace | text-corpus` · temporal-class =
  `DC-slow-drift | AC-order-sensitive | periodic | bursty/self-exciting |
  interaction/equality | long-memory`
- **Why it's not already covered:** <the axis it probes that backtracking /
  existing benches don't>

## 1. Hypothesis (frozen)
- **Hypothesised temporal character + WHY:** <e.g. "AC / order-sensitive because
  the signal at t depends on the recent history of X, not the marginal">
- **What would make it ABORT:** <the specific null result — "if ordered ≈ shuffled
  the property is composition/marginal, not temporal">

## 2. Labeler (the crux — validate or the measurement is meaningless)
- **Signal:** per-`<token|span|sentence>` `<categorical | scalar>`.
- **⚠ No-leakage rule (preregistered — Cycle-1 lesson).** The label MUST be
  assignable from the span's **own content alone**. If the instruction references
  the span's *relation to its neighbours* — "answers a preceding question",
  "follows from prior", "unlike the previous sentence" — the measured order
  statistic is **circular** (the temporal dependence is baked into the label).
  Redesign, or expect a skeptic kill: this is exactly how `question-answer-
  adjacency` died, and why `assumption-then-consequence` is only provisional.
- **Labeler + version:** <Claude judge model + the exact prompt, OR the corpus
  field / classifier>. Bulk = Haiku; adjudication = Sonnet.
- **Validation plan:** held-out inter-judge agreement (re-label a sample with an
  independent call / a second model) + an independent cross-check (keyword /
  lexicon / classifier), reporting an estimated **noise floor**. Effect must
  survive this noise (test it, as backtracking did).

## 3. Data (version-pinned)
- **Source:** <corpus name+snapshot | target LM + decoding params + prompt set>.
- **Unit of time:** <token / sentence / turn index>. **N units, held-out split.**

## 4. Statistics + order-destroying null(s)
- **Ordered statistic(s):** <ACF(k) / dwell distribution / MI(L_t;L_{t+k}) /
  Fano / spectral DC-AC share — pick what the hypothesised class predicts>.
- **Null(s):** <N1 within-block permutation (kills order, keeps marginal);
  N2 marginal-preserving process that keeps any trend but kills the structure;
  N3 homogeneous baseline>. **Temporal-ness gate:** ordered must exceed the null
  beyond sampling **and** labeler noise.

## 5. Baselines
- **Chance / oracle** for the eventual latent-recovery probe (prefer a *provable*
  floor — e.g. a data-processing-inequality bound — over an empirical gap).

## 6. Predictions per architecture (blind — write BEFORE any arch runs)
- per-token SAE: <predicted recovery + reason (e.g. DPI floor because …)>
- window families (TXC-pre / -post / Stacked / Spectral): <predicted + reason,
  keyed to whether the latent is additive-over-position or an interaction>

## 7. Mirror (Appendix B), to be fit only if the gate PASSES
- **Process:** <2-state Markov / AR(1) / semi-Markov / Hawkes / periodic+noise /
  renewal — keyed to the matched statistic>. **Matched param(s):** <…>. **What is
  deliberately NOT matched:** <…>.
- **⚠ Non-fitted-moment gate (preregistered — Cycle-1 lesson).** Name **≥1
  statistic the mirror is NOT fit to**, and the abs-error **tolerance** it must
  meet on held-out draws. A mirror that only reproduces the moment it was fit to
  is **circular validation** — this is how `quotation-burst` died despite a clean
  null gate. E.g. fit the transition matrix → require it to also reproduce dwell-CV
  and Fano within <tol>, which the matrix does not set directly.

---
_Frozen-by: <agent/model>. Amendments (dated, transparent): none._
