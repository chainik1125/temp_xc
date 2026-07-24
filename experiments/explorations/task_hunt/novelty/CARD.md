# FROZEN card — vocabulary-novelty trailing rate (fineweb), Stage-1 screen

**Status: FROZEN at commit (commit-then-run; no screen cell has been
executed when this card is committed — git order is the evidence).**
Agent: runpod-e. Briefing: `briefings/task-hunt-r2-e.md` § 3 (quantity
mode; queue position 4). Frozen from runpod's `CARD_DRAFT.md`
(ledger `../CANDIDATES.md` B2), whose **label-side triage bars and
their PASS result are already committed and mac-local-REVIEWED**
(LOG 2026-07-24, "REVIEW: candidate factories — BOTH APPROVED"). This
card adds the activation-side screen and is the operative one for the
run (protocol: the running agent's card governs).

## 1. What the draft already settled (not re-litigated here)

Label side, verified by review against `../labels/novelty_stats.json`:
primary face `nov_resid` (kernel-smoothed trailing novelty rate over
PREVIOUS tokens only, lags 1–64, half-life 16, position-detrended by
log2-position-bin train-doc mean) passes both frozen bars on all three
tokenizers — current-token type-mean AUC 0.551/0.563/0.551 and
position AUC 0.472/0.478/0.477, i.e. 0.52–0.53 direction-agnostic,
far under the 0.65 kill bar. The RAW face `nov_rate` is demoted to
anchor/disclosure only (position AUC 0.87–0.88 direction-agnostic —
the Heaps trend).

**Consequence for this screen, stated so it is not silently assumed:**
the raw face is NOT screened. Its position confound makes a
window-vs-per-token gap uninterpretable, and the review's
qualification 1 (punctint) is the precedent for refusing to quote an
unmatched position-confounded face. Only `nov_resid` is screened,
plus the null-corpus receipt below.

## 2. Zero-new-caching, verified on this volume before freezing

The draft ASSERTS `token_ids` byte-identical to the committed
`../labels/replag_fineweb_<tok>.npz`. Verified here, and one step
further than the assertion — the factory's flat stream is checked
against **my windowed activation caches**:

- `token_ids` and `doc_off` byte-identical (novelty ↔ replag_fineweb),
  all three tokenizers;
- every one of my cache rows reproduces its contiguous flat-stream
  slice **exactly**: gpt2 5989/5989, gemma2-2b 5985/5985, llama31-8b
  5924/5924 rows match (n_prefix 0/1/1, content 128/127/127).

So the screen reads `/workspace/replag_caches/<model>/hs<screen>.npy`
with **no new forward passes**. Screen layers are the replag screen
layers (`replag/cache_acts.py` SCREEN_HS): gpt2 hs7, gemma2-2b hs14,
llama31-8b hs14.

## 3. Row construction (frozen)

Flat `(doc, pos)` manifest rows (`man_nov_*`, 20k/class, already
balanced and doc-split by the builder) are mapped to
`(cache_row, cache_pos)` by `chunk = pos // content`,
`cache_pos = n_prefix + pos % content`; rows whose chunk was dropped
as a document tail are discarded.

**Eligibility (uniform, so every screened T is read on IDENTICAL
rows):** `pos ≥ 64` (the builder's triage convention) **and**
`pos % content ≥ 63`, so a trailing window of up to T = 64 lies inside
one cache row and never crosses a chunk boundary. Measured yield
(gpt2): 24,917 train / 5,987 test eligible, per-class ≥ 1,744.
Split by the builder's `doc_split` (320 train / 80 test docs); caps
**4000 train / 1500 test per class**, seeded subsample
(MATCH_SEED 1013 + crc32 of the task/split string, my round-1
convention); **MIN_ROWS 300** floor per class or the target is skipped
and recorded as skipped.

## 4. Targets

- **`nov_bin` (PRIMARY)** — the committed 3-class terciles of
  `nov_resid`. Chance 1/3.
- **`nov_null_bin` (RECEIPT, not a target)** — terciles of
  `nov_resid_null`, the same statistic computed on the builder's
  within-doc-shuffled novelty bits (`null_perm`). Edges are computed
  over the eligible TRAIN pool here (recorded in results meta) and an
  independent balanced manifest is drawn from the same eligible pool,
  so the two faces are read on comparable row counts. **This is the
  card's mechanism receipt**: real activations should read the REAL
  drift better than the fake one. Parity ⇒ the signal is local
  composition/bookkeeping, not topical drift.

## 5. Probe grid (frozen; frozen `problib` stack, no retuning)

Per model, on the screen layer:

- **Per-token-first triage (binding hunt convention):** per-token
  linear + per-token MLP(512) on `nov_bin`, run and reported FIRST.
- `T ∈ {4, 8, 16, 32}`: window linear (flatten, right-edge anchor),
  window-MEAN linear, context-shuffled linear (slots 0..T−2 permuted
  per row, anchor fixed, seeded SHUF_SEED 1234 + crc32).
- **window-MEAN additionally at T = 64** (full kernel support). The
  flatten/shuffle arms stop at T = 32 for a stated reason, not a
  silent one: at T = 64 a flatten probe on llama-8b is 262,144
  features, beyond what this screen's probe fit can hold; the MEAN arm
  is d-dimensional and is the regime-2 reader the card's prediction
  actually names.
- MLP(512) window + shuffled-window at `T ∈ {16, 32}`.
- Permutation nulls (NULL_SEED 99) on the linear pair at T = 16.
- **Position-only floor** on the shipped rows: probe on
  `[cache_pos, cache_pos², log2(1+doc_pos)]` → the same 3-class
  target. Reported next to every window number (review qualification
  1's generalization: never quote a window gap without its position
  floor).
- Receipt face `nov_null_bin`: per-token linear + window-MEAN linear
  at `T ∈ {16, 32, 64}` (linear only — it is a comparison anchor).

Metric: `acc_test` (3-class, chance 1/3) + `per_class`.
Kernel mass within T (builder-computed, the T-axis interpretation):
**0.17 / 0.31 / 0.53 / 0.80 / 1.00** at T = 4/8/16/32/64.

## 6. Frozen predictions (scored either way in the LOG)

- **N1.** Per-token probes read `nov_bin` poorly — near the position
  floor and ≥ 0.05 acc below the best window cell at T ≥ 16 (the
  trailing RATE is not maintained as per-position state).
- **N2.** window − per-token gap is positive at T ≥ 8 and **grows
  with T along the kernel-mass curve** (0.17 → 1.00), with the
  largest step between T = 8 and T = 32.
- **N3 (regime-2).** The MEAN probe captures essentially all of the
  window advantage (flatten ≈ mean) and the context shuffle is
  **immune** (shuffled ≈ ordered) — order-free aggregation.
- **N4 (the mechanism receipt).** Real `nov_bin` recovery exceeds
  `nov_null_bin` recovery at matched T by ≥ 0.03 acc; the null face
  stays near its own position floor.
- **N5 (cross-model).** The pattern replicates in direction on all
  three models; per-token stays low at every scale (contrast with
  replag, where per-token conversion was total and scale-ordered).

## 7. KEEP / KILL (frozen)

**KEEP** iff N1 AND N2 hold on at least two of three models (window −
per-token ≥ +0.05 acc at some T, growing over the ladder), AND the
window advantage clears the position floor by ≥ 0.05, AND N4 holds
(real beats null by ≥ 0.03). Shuffle immunity (N3) then classifies it
regime-2, which the program ACCEPTS — it is not a kill.

**KILL** if ANY of:
1. per-token reads `nov_bin` within noise (≤ 0.02 acc) of the best
   window at every T on the majority of models — converted/ambient;
2. no window − per-token gap ≥ 3 σ_null at any T;
3. the gap does not grow anywhere over T ∈ {4…32};
4. `nov_null_bin` recovery reaches parity with the real face
   (< 0.03 apart) — the signal is local bookkeeping, not drift;
5. the window advantage does not clear the position floor.

σ_null is estimated from the permutation cells (NULL_SEED 99); with
one null pair per model the 3σ band is quoted from the pooled
across-model null spread, and that pooling is disclosed rather than
presented as a per-cell null.

## 8. Deliverable

`results/screen_<model>.json` (incremental/resumable per cell), one
LOG verdict scoring N1–N5, and — only if KEEP — a T-scaling figure.
No leaderboard rows (Stage-1 screens are raw-activation probes, not
runner cells).
