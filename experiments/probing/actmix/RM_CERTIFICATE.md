# RM ↔ btk-only EQUIVALENCE CERTIFICATE — **PRELIMINARY**

**Status: PRELIMINARY** (2026-07-28 ~02:05 UTC, runpod-1). Final
certificate ships after the paper-faithful shards drain and the 11:00
protected renders; nothing here is an exhibit yet. Framing per the
ratified census-first ruling (3b0a4df3d): **the integer fired-census
leads; trace telemetry gives rate bounds, never counters.**

## 1. Claim structure (the lemma pair)

- **Contact ⇒ divergence** (this lane): if rectify-after-select ever
  touches the selected set — a selected pre-activation goes negative —
  the RM and btk-only trajectories separate and stay separated
  (weights, then metrics).
- **Bit-identity ⇒ zero contact** (runpod-2's retro-proof, LOG
  fd3e4ff16 / 829f05070): tensor-identical checkpoints at step N
  retro-prove the boundary never crossed zero in [0, N].

Together: per-venue regimes of ONE mechanism. Probing (this venue) =
rare between-sample contact. RLHF (runpod-2's venue) = never-contact
(bit-identical through T16, boundary ≥ 2.21). runpod-a's dq/λ̂ split
(dq diverges — W_enc 0.35 — while λ̂ stays identical) is the same
selection-level story seen from the estimator side.

## 2. Identity set (torch.equal on every shared tensor)

| pair | verdict |
|---|---|
| `batchtopk_sae` s1, s42 | **IDENTICAL** (7/7 tensors) |
| `batchtopk_sae` s2 | METRIC-IDENTICAL (weights remote) |
| `txc_batchtopk_pre` s42 **T=1** | **IDENTICAL** (7/7 tensors) |

T=1 identity is the controlled limit: with one position there is no
window pool to mix, so the two compositions are the same program.

## 3. Divergence map (all trained pairs T ≥ 2 diverge)

Every `txc_batchtopk_pre` pair at T ≥ 2 DIVERGES: 6/7 tensors differ
(only `global_step` equal); **`num_tokens_since_fired` — the integer
fired-census — differs**, which is the selection-level contact
witness (an optimizer-noise or numerics story cannot move an integer
census). Δauc = RM − btk, latest row per slot, duplicate slots
surfaced by the checker (never pooled), alias exclusion list applied
(see RM_EQUIVALENCE.md):

| T | seed | Δ k5 | Δ k20 |
|---|---|---|---|
| 2 | 42 | +4.56e-03 | +3.76e-03 |
| 4 | 42 | +1.54e-03 | -3.21e-03 |
| 6 | 42 | -1.63e-02 | +7.77e-04 |
| 6 | 1 | -1.02e-02 | +2.46e-03 |
| 6 | 2 | -1.38e-02 | -5.19e-03 |
| 8 | 42 | +8.75e-03 | +1.02e-02 |
| 8 | 1 | -9.82e-03 | -5.12e-03 |
| 8 | 2 | +5.63e-03 | -1.23e-03 |
| 10 | 42 | -6.12e-04 | -6.84e-03 |
| 10 | 1 | -2.29e-03 | -6.87e-03 |
| 10 | 2 | -1.41e-02 | -2.49e-03 |
| 16 | 42 | +2.46e-03 | -1.67e-03 |
| 16 | 1 | -7.53e-03 | -4.30e-04 |
| 16 | 2 | +7.94e-04 | -6.10e-03 |

(T2 is METRIC-DIVERGES with weights remote; mirrored ckpt allows a
tensor-level re-check on demand.)

Regimes read off the map, multiplicity caveats attached:

- **T6, k5**: 3/3 btk-ahead at ~1.3e-2 — the largest consistent
  block; k20 mixed at the same T (noise-scale).
- **T8**: coin-flip in both k — the crossover region.
- **High-T k20 block (T ≥ 10): 6/6 btk-ahead** — one-sided sign
  P ≈ 0.5^6 ≈ 1.6%, NOMINAL ONLY: the block was flagged at 5/5, so
  the post-hoc selection caveat stands. Magnitudes 0.4–6.9e-3.
- Secondary: the T10 column is 6/6-slot negative across both k
  (s2 k5 −1.41e-2 is the 2nd-largest delta in the map); caveat:
  k5/k20 share weights per seed ⇒ ~3 independent draws, not 6.

## 4. Trace bounds (census-first; traces are BOUNDS)

Sampled every 250 steps, 80 samples per completed 20k-step cell
(`/workspace/logs/telemetry_rm/`):

- **v2 arms: 0/1120 sampled steps with negative selection boundary**
  across 14 completed traces. Rule-of-three per cell: contact at
  < 3.75% of sampled steps (95%). Traces cover 0.4% of steps; the
  census divergence proves ≥ 1 contact event in the unsampled 99.6%.
- **Boundary floor declines monotonically with T**: T6 {6.46, 6.62,
  7.18} · T8 {5.21, 5.61, 5.65} · T10 {4.19, 4.26, 4.66} · T16
  {3.29, 3.40} — approaching zero from above as the window pool
  deepens, never touching it in-sample.
- **Per-seed floors identical across arms at T10** (4.187 / 4.258 /
  4.664 in both trace sets) while the census diverges — extreme-
  statistic agreement consistent with contact being rare and
  weight divergence tiny.
- Coverage disclosure: btk arm traced at T10 only (telemetry fix
  landed mid-night, 71a4de31f); RM T16 has 2/3 traces; untraced
  cells are covered by endpoint census from mirrored ckpt buffers.

## 5. Controls

- **Positive control** (`positive_control.py`): thin-pool config
  (d_sae=64, k_pos=48) where rectify-after-select MUST fire from
  step 0 — the instrument reports DIVERGENCE (run fails unless it
  does). The checker detects what it is supposed to detect.
- **Untrained twins / alias exclusion**: byte-equal untrained
  clusters that once aliased a phantom "T8-exact" pair are on a
  standing exclusion list (RM_EQUIVALENCE.md); joins filter
  n_steps > 0 and the list.
- **T=1 controlled limit**: identity where the compositions provably
  coincide (§2).
- **Shuffle control**: every row carries the protocol 1.2.0
  within-window shuffle; identity rows show shuffle_identity = 1.

## 6. Durability

All twin ckpts (32 local keys incl. the night T10/T16 additions) on
the ratified mirror `han1823123123/temp-bench-data` under
`ckpts/<train_key>/`, LFS sha256 receipts in
`/workspace/logs/ckpt_push.log`; spot-check MATCH (e91d887fac22fb33).

## 7. What PRELIMINARY means / path to final

- Paper-faithful shards (CARD_PAPER_FAITHFUL.md) still training; E1
  is confirming in-flight (T16: 40/43 sampled steps negative-
  boundary, min −11.9 — a different regime from both v2 arms), but
  E1–E3 scoring waits for drain.
- 11:00 protected btk renders + archived-anchor labeling still owed.
- relu-mix remains certificate evidence ONLY (arm mapping 692b) —
  never a matrix column.
- PTR: every number above re-derives from `results/leaderboard.jsonl`
  + `rm_equivalence.json` + the trace files; no hand-carried values.

_Author: runpod-1 (claude-fable-5). Sources: rm_equivalence.py
outputs at da853dd01, telemetry parse entry 02:03 UTC, LOG receipts
a264241ac / 71a4de31f / 74c4c6f00 / 983baf1a9._
