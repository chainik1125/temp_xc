# DRAFT mini-card — vocabulary-novelty trailing rate (fineweb)

**Status: DRAFT (runpod, `briefings/candidate-factory-broad.md`,
ledger entry `../CANDIDATES.md` B2).** The running agent freezes its
own card from this draft. **This draft (with the triage bars below)
is committed BEFORE `../labels/build_novelty.py` runs** — the bars
are the ship/kill authority for the bundle itself.

Data side: builder `../labels/build_novelty.py` (pure logic
`../labels/novelty_lib.py`, tests `tests/test_novelty_labels.py`) →
`../labels/novelty_fineweb_{gpt2,gemma2,llama31}.npz` +
`../labels/novelty_stats.json`.

**Economics (the point of this corpus choice):** `token_ids` are
builder-ASSERTED byte-identical to the committed
`replag_fineweb_<tok>.npz` (same pinned fineweb sample, same
tokenization) — any activation cache built for the replag screen
reads these labels with **zero new caching**. Fineweb caches exist on
runpod-e's volume for gpt2 / gemma-2-2b / llama-8b.

## The candidate logic

Event stream: per-token novelty bit = first in-document occurrence of
the token type (exact, tokenizer-level, zero-API). **Primary label:
`nov_resid`** — the kernel-smoothed trailing novelty rate over
PREVIOUS tokens only (lags 1–64, half-life 16; the current token
never contributes to its own label), position-detrended by
subtracting each log2-position-bin's train-doc mean (Heaps' law makes
the raw rate partly a position label; the raw face `nov_rate` ships
as DISCLOSED secondary with its position AUC printed next to it).
This is a topic-drift intensity: the rate spikes when the document
enters new material and decays over familiar ground.

**Conversion risk, stated plainly (axis a):** the complement
per-token bit ("current token seen before") is the replag graveyard —
converted at 0.74–0.97 AUC at every screened scale. That kills any
current-token face and is exactly why the primary excludes the
current token: the screen question is whether the trailing RATE is
maintained as state anywhere, or must be aggregated from per-token
novelty bits at read time. Both outcomes are informative; only the
aggregation one is expected — regime-2 is the bet.

## Clock (axis c)

Token-level kernel, no sentence bridge: a trailing window of length T
sees kernel mass ≈ 0.16 / 0.29 / 0.50 / 0.75 / 1.00 at
T = 4 / 8 / 16 / 32 / 64 (exact values in
`novelty_stats.json.kernel.mass_within_T`) — the panel ladder spans
16 %→75 % of the label's support, with T = 64 optional closure.

## Label-side triage — FROZEN BARS (kill authority)

Computed by the builder on test-doc rows at pos ≥ 64, both faces
(raw, resid), both leak routes:

- **KILL** the bundle if the PRIMARY face (`nov_resid` terciles, top
  vs bottom) has current-token type-mean AUC **≥ 0.65** or
  position-only AUC **≥ 0.65** (the interleave precedent: source at
  0.66 is the recorded example of a killed face; tss at 0.55 of a
  passing one).
- 0.55–0.65 on the primary: ships with the elevation disclosed as a
  screen caveat (the screen's per-token-first triage remains the
  activation-side authority).
- The RAW face's position AUC is expected HIGH (that is the Heaps
  trend, the reason detrending exists) — it does not kill the bundle;
  it demotes `nov_rate` to anchor/disclosure only.

## Predicted T-pattern (freeze candidate)

Per-token probes read the current token's own novelty (converted,
replag precedent) but NOT the trailing rate; window − per-token gap
on `nov_bin` positive and growing over T ∈ {4…32} tracking kernel
mass; order-free (mean ≈ flatten; shuffle-immune) — regime-2, with
the within-doc-shuffle null (`nov_rate_null`) as the frequency-only
mechanism receipt: real-corpus rate structure (its variance and its
recovery) must exceed the null corpus's.

## Triage RESULT (builder-derived, appended after the frozen bars ran)

**PASS — the bundle ships.** Primary face (`nov_resid` terciles, test
rows, pos ≥ 64), stable across all three tokenizers: current-token
type-mean AUC **0.551 / 0.563 / 0.551** (gpt2/gemma2/llama31) and
position AUC **0.472 / 0.478 / 0.477** — direction-agnostic ≈ 0.52–0.53
(the committed `rank_auc` is directional; the kill bar reads
|AUC − 0.5|). Both far under the 0.65 bar, at the interleave-`tss`
passing level. The RAW face's position AUC is 0.121–0.128 — i.e.
**0.87–0.88 direction-agnostic** (high-novelty rows sit early: the
Heaps trend, as predicted) — `nov_rate` is anchor/disclosure only, per
the pre-stated demotion. Base novelty rate 0.34–0.36; balanced
manifests 20k rows/class; `token_ids` asserted identical to replag ⇒
zero new caching.

**Label-side mechanism receipt (drift structure the frequency null
lacks):** residual spread 0.112 vs 0.093 under the within-doc-shuffle
null (~21 % excess), and pooled per-doc autocorrelation of the
residual at lags BEYOND the kernel's 64-lag support — where windows
share no input bits and the null sits ≈ 0 by construction — is
real **0.129–0.134 vs null 0.023–0.026** at lag 64 and
**0.056–0.064 vs 0.006–0.016** at lag 128 (all three tokenizers).
Real novelty has persistent topical-drift structure; the null is pure
composition + Heaps. This is the label-side face of the shuffle
receipt the screen will run on activations.

## Kill rule (draft — freezing agent finalizes for the screen)

KILL if ANY of: (1) label-side triage bars above fire (pre-screen
kill, free); (2) activation per-token probe reads `nov_bin` within
noise of the best window at every T (converted / ambient);
(3) no window − per-token gap beyond 3 σ_null at any T; (4) the gap
does not grow with T anywhere in {4…32}. A positive needs: `nov_bin`
window-readable, per-token-poor, T-growing along the kernel-mass
curve; shuffle-immunity then classifies it regime-2 (accepted), and
recovery on the null corpus at parity would reclassify the signal as
local bookkeeping (kill).
