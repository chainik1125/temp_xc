# RM ↔ btk-only EQUIVALENCE CERTIFICATE — **v1.0 (probing venue)**

**Status: v1.0** (2026-07-28 06:44 UTC, runpod-1; PRELIMINARY was
~02:05 UTC same date). All gating inputs are now in-tree: the
paper-faithful grid completed and scored (card §9: E1 CONFIRMED, E2
NOT CONFIRMED/null, E3 PASS — LOG 5075d098e), relu-mix T2/T4 bands
filled to 3 seeds, telemetry parsed, cross-venue receipts posted.
Supplementary appends expected only from runpod-b's rmx_b lane
(checks 5–6/6) — extensions, not gates. Framing per the ratified
census-first ruling (3b0a4df3d): **the integer fired-census leads;
trace telemetry gives rate bounds, never counters.**

**Headline**: relu-mix (rectify-after-select) and btk-only differ by
a REAL selection-level mechanism (integer census witness, §3) whose
probing-metric consequence is NULL at matched budgets (§3a) — and
the paper-faithful composition tells the same story from a third
angle (§7): frequent per-step boundary contact, tiny per-window
effect, no consistent metric gap. One mechanism, three compositions,
regime set by venue; no exhibit claims a performance win from the
composition change.

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
(bit-identical through T16, boundary ≥ 2.21; extended by runpod-b's
rmx_b relay: T8 trio CLOSED 3/3 identical — aliases incl.
f857417704b13efa↔7d51409daff2fa72, 06e2fbce45e80006↔a2fe8d7e382dc1cb
— checks continuing; check 4 T10/s42 CLOSED tensor-grade 7/7 via
the unblocked mirror relay, alias f03ff666cb8e8cb1↔aa4e62a74ed1686e;
checks 5–6/6 remain as extensions). runpod-a's dq/λ̂ split (dq
diverges — W_enc 0.35 — while λ̂ stays identical) is the same
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

### 3a. Metric consequence: NULL at matched budgets

The honest top-line: across the full map no k- or T-consistent
direction survives its multiplicity caveat, per-column magnitudes
stay ≤1.6e-2, and the pre-registered directional test on the
paper-faithful arm (card §9 E2: pf ≤ btk at T≥8) came back NOT
CONFIRMED (10/18 slots below, sign P ≈ 0.41; T8 k5 actually 3/3
ABOVE). **The selection-level mechanism is real (the census proves
contact); its probing-metric consequence at 20k-step matched
budgets is null.** The certificate certifies the mechanism and the
equivalence — it does not license any performance claim for either
composition, and no exhibit may cite the flagged blocks as wins
without their caveats. The relu-mix arm's own curve is 3-seed at
every T∈{2,4,6,8,10,16} (T2/T4 filled 05:43 UTC) with the T1 s42
IDENTICAL anchor.

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

All certificate ckpts on the ratified mirror
`han1823123123/temp-bench-data` under `ckpts/<train_key>/` with LFS
sha256 receipts in `/workspace/logs/ckpt_push.log`: the 32-key twin
set (spot-check MATCH e91d887fac22fb33), the 8 paper-faithful
shard-A/B ckpts, and the 4 relu-mix fill ckpts — 44 receipts total
from this venue, idempotent tool, re-verifiable.

## 7. Third-composition corroboration (paper-faithful arm)

The vendored paper composition (`paper_txc_base_v1t`, upstream
94119bc08 verbatim; adapter parity bitwise vs the shipped-ckpt
evaluator) ran the full 7T×3s grid and shows the SAME
mechanism-vs-metric split from the opposite regime:

- **Contact frequent per step**: training telemetry at T16 has the
  selection boundary negative at ~93% of sampled steps (min −11.9)
  — vs 0/1120 for the v2 arms.
- **Effect narrow per window**: E1 zero-picks monotone {0 at T≤6 ·
  0.04 at T8 · 0.20 at T10 · 0.46 at T16} = ≤0.14% of budget.
- **Metric consequence null**: E2 NOT CONFIRMED (§3a).
- **Provenance closed**: E3 PASS — the archived, never-retrained T5
  anchors interpolate the retrained T4–T6 columns at both k
  (k5 0.8336→0.8368→0.8413 rising; k20 0.8963→0.8952→0.8908
  falling) — the composition the paper shipped is the composition
  this certificate measured.

## 8. Scope & pointers

- relu-mix remains certificate evidence ONLY (arm mapping 692b) —
  never a matrix column.
- 11:00 protected btk renders are exhibit-side and cite this
  certificate; they do not gate it.
- PTR: every number above re-derives from `results/leaderboard.jsonl`
  + `rm_equivalence.json` + the trace files; no hand-carried values.

_Author: runpod-1 (claude-fable-5). Sources: rm_equivalence.py
outputs at da853dd01; telemetry parse 02:03 UTC (da853dd01); formal
scoring 06:11 UTC (5075d098e); E3 entry 1300ed2a5; fills ba8a4ff3e;
cross-venue 829f05070 / fd3e4ff16 / ae277d725 / f57d3c820; night map
receipts a264241ac / 71a4de31f / 74c4c6f00 / 983baf1a9._
