# DRAFT mini-card — interleaved-document `tss` (anti-conversion candidate)

**Status: SCREEN-READY DRAFT (quantity mode).** Data side built by
runpod-b (`briefings/hunt-support-stats.md` item 3); promoted from
parked to screen-ready by runpod under
`briefings/candidate-factory-broad.md` (the Han quantity-mode
directive lifts the park for screening; ledger entry
`../CANDIDATES.md` B1). The running agent freezes its own card from
this draft — the predicted T-pattern and falsifier below are the
freeze candidates.

Data side (committed, CPU-complete, label-side only):
`../labels/build_interleave.py` → `../labels/interleave_fineweb_{gpt2,
gemma2,llama31}.npz` + `../labels/interleave_stats.json`; pure logic in
`../labels/interleave_lib.py` under `tests/test_interleave_labels.py`.
Same alignment contract as replag: **feed the exact `token_ids`** — do
not re-tokenize.

**GPU economics:** this corpus is fineweb text re-composed — but the
interleaved sequences are NEW token streams, so the screen needs one
cheap forward-pass caching run per model over `token_ids` (~330k
tokens per tokenizer; minutes on an H100), not the existing fineweb
caches. Say so in the screen plan; no other cost.

## Primary and anchor (frozen framing — the demotion is the design)

- **`tss` — tokens since the last source switch: PRIMARY.** Unigram
  floor ≈ **0.55 AUC** (top vs bottom tercile, held-out estimator) —
  near-blind to token identity; the label-side triage this candidate
  already passed. Its only generative signal is the switch hazard,
  kept weak by the 1–4-sentence jitter (measured h(t) ≈ 0.012 → 0.03,
  NOT memoryless — disclosed in
  `interleave_stats.json.switch_hazard`; don't oversell).
- **`source` — which doc is active: DISCLOSED ANCHOR (regime-1 kill
  face), not a primary.** Generatively useful (predicts vocabulary) ⇒
  expected converted. Held-out unigram readout: **0.66 AUC matched vs
  0.70 random** — lexical matching works but removes only ~0.04.
  **Frozen prior: per-token HIGH on `source` is the expected outcome
  and does NOT count against the candidate; `tss` is the face the
  candidate lives or dies by.**

## Why this class might resist conversion

Round-1 closure: conversion kills any latent WITH generative training
signal — the model linearizes per-token whatever helps predict the
next token. This corpus holds that variable near its minimum for
`tss` by construction (jittered block lengths), while keeping the
state real: two lexically-matched fineweb docs (greedy max-Jaccard
pairing, overlap 0.080 → 0.120 vs random) interleaved in strictly
alternating 1–4-sentence blocks.

## Screen sketch (freezing agent owns the card; ~2 h GPU incl. caching)

- Cache: forward passes over each npz's `token_ids` (models with
  existing loaders: gpt2 / gemma-2-2b / llama-8b; mid-depth
  resid_post per the hunt convention).
- Per-token-first triage (hunt convention) on `tss` terciles
  (train-split edges ≈ 19 / 46–47 tokens, per-tokenizer values in the
  stats JSON; balanced manifests `man_tss_*` at 20k rows/class,
  pos ≥ 32, split by interleaved doc). HIGH per-token ⇒
  presumptively converted ⇒ depth sweep as the WHY-diagnostic, stop.
- Window probes (mean + flatten) at T ∈ {4, 8, 16, 32}: median block
  = 47 tokens (q10 13, q90 103), so T = 32 typically reaches back to
  the previous switch while T = 4 rarely does — the ladder spans the
  clock.
- The shuffled-block null (`null_perm` materializes the null corpus;
  `tss_null`/`source_null` are its recomputed labels): run the reader
  over the null corpus and compare recovery — if `tss` is read
  equally well when document flow is incoherent, the signal is local
  bookkeeping, not maintained state. DECISION POINT for the freezing
  agent: adopt as the order/state receipt or as a secondary control.
- `source` runs as the disclosed anchor face only.

## Predicted T-pattern (freeze candidate)

`tss` per-token LOW (near its 0.55 unigram floor ⇒ not converted —
the anti-conversion bet); window − per-token gap positive and GROWING
over T ∈ {4…32} as windows reach the previous switch (regime-2 at
minimum; any order-carried component shows as flatten > mean);
recovery DEGRADED on the shuffled-block null corpus. `source`:
per-token HIGH (converted/lexical, expected), window adds little.

## Kill rule (draft — freezing agent finalizes)

KILL if ANY of: (1) per-token-first triage on `tss` is high
(converted — the bet fails at the first gate); (2) no
window − per-token gap beyond 3 σ_null at any T on `tss`; (3) the
only window win is on `source` while its per-token probe already
clears the lexical route (ambient vocabulary detection); (4) the gap
does not grow with T anywhere in {4 … 32}. A positive needs: `tss`
window-readable, per-token-blind, T-growing, and degraded on the
shuffled-block null.
