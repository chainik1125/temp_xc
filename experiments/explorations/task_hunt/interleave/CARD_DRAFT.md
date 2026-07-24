# DRAFT mini-card — interleaved-document state (anti-conversion candidate)

**Status: DRAFT (runpod-b, item 3 of `briefings/hunt-support-stats.md`).
NOT operative — the anti-conversion SCREEN is parked program-wide
(round-2 decision) and runs only if a pod frees up AND mac-local
greenlights. The running agent freezes its own card; this draft + the
committed data side exist so a screen can start within the hour it is
greenlit.**

Data side (committed, CPU-complete, label-side only):
`../labels/build_interleave.py` → `../labels/interleave_fineweb_{gpt2,
gemma2,llama31}.npz` + `../labels/interleave_stats.json`; pure logic in
`../labels/interleave_lib.py` under `tests/test_interleave_labels.py`.
Same alignment contract as replag: **feed the exact `token_ids`** — do
not re-tokenize.

## The candidate logic (why this class might resist conversion)

Arm-B closure (round 1): conversion kills any latent WITH generative
training signal — the model linearizes per-token whatever helps predict
the next token. This corpus is built to hold that variable at its
minimum: two lexically-matched fineweb docs interleaved in strictly
alternating 1–4-sentence blocks (jittered, seeded). Two per-token
labels:

- **`tss` — tokens since the last source switch (PRIMARY):** the
  window-readable state. Its generative usefulness is only the switch
  hazard, and the jitter keeps that weak — measured h(t) rises
  gently ~0.012 → ~0.03 across the block-length range (NOT memoryless;
  disclosed in `interleave_stats.json.switch_hazard`, don't oversell).
  Unigram floor ≈ **0.55 AUC** (top vs bottom tercile) — the label is
  nearly invisible to token identity.
- **`source` — which doc is active (CONTROL / KILL-RISK face):**
  generatively useful (predicts vocabulary) ⇒ *expected converted*.

**Frozen prior (state it before any screen): per-token HIGH on source
identity is the expected kill; the lexical control is what the
candidate lives or dies by.**

## The lexical control, measured (labels only, no activations)

Greedy max-Jaccard pairing over content types lifts pair overlap
0.080 → 0.120 (mean; p90 0.163), but the held-out unigram log-odds
readout of source identity is **0.66 AUC matched vs 0.70 random**
(means, stable across all three tokenizers; per-pair p90 ≈ 0.72) —
matching works but removes only ~0.04. The residual lexical route is
substantial, and per-token probes on ACTIVATIONS sit above any token-
identity floor by construction. So the honest triage expectation:
source identity fails per-token-first triage; `tss` is the face that
must carry the candidate. (Estimator note: the unigram distributions
come from held-out halves of the source docs — the tests showed any
in-corpus estimator leaks the source through count asymmetry.)

## Screen sketch (~2 h GPU once greenlit; freezing agent owns it)

- Per-token-first triage (hunt convention) on `tss` terciles
  (train-split edges ≤ 19 / > 46 tokens; balanced manifests
  `man_tss_*`, 20k rows/class, pos ≥ 32, split by interleaved doc).
  HIGH per-token ⇒ presumptively converted ⇒ depth sweep as the
  WHY-diagnostic, stop.
- Window probes (mean + flatten) at T ∈ {4, 8, 16, 32}: median block =
  47 tokens (q10 13, q90 103), so T = 32 typically reaches back to the
  previous switch while T = 4 rarely does — the T-range spans the
  clock.
- The shuffled-block null (`null_perm` materializes the null corpus;
  `tss_null`/`source_null` are its recomputed labels): run the reader
  over the null corpus and compare recovery — if `tss` is read equally
  well when document flow is incoherent, the signal is local
  bookkeeping, not maintained state. DECISION POINT for the freezing
  agent: adopt this as the order/state receipt or as a secondary
  control.
- `source` runs as the disclosed control face only.

## Kill rule (draft)

KILL if ANY of: (1) per-token-first triage on `tss` is high
(converted); (2) no window − per-token gap beyond 3 σ_null at any T on
`tss`; (3) the only window win is on `source` while its per-token
probe already clears the lexical route (ambient vocabulary detection —
the lexical-control criterion); (4) the gap does not grow with T
anywhere in {4 … 32}. A positive needs: `tss` window-readable,
per-token-blind, T-growing, and degraded on the shuffled-block null.
