# Working state — agent `runpod-1`

**2026-07-27 21:15 London (date-verified) — day report posted (LOG
21:12). NIGHT GRID RUNNING at PIN db098b8c1: dead-latent RM
T{6,8,10,16}×3s + btk T{6,10}×3s + telemetry, 2 GPUs, drain ~01:30
pod. Morning queue below. GPU 2 = runpod-2, never mine.**

## The RM arc (context for any resume)

Identity discovered (low-T bit-identical twins) → halt approved →
Han override (dead-latent hypothesis needs high T) → divergence
MEASURED: onset T≥2, deepening with T. CORRECTED mechanism
arithmetic: per-window selection depth = k_pos·T/d_sae grows
linearly (0.11%→1.74% T1→T16); sae stays shallow ⇒ sae identical
×3 seeds (incl. cross-pod exact), pre identical ONLY at T1.
T16 endpoint: ~57% dead latents BOTH arms, ~40% disjoint survivor
sets, ~0.002 AUC cost (width-contingent per quote guards
7093c21f8). Dead-frac vs T is U-shaped (0.44→0.37→0.57).
Instrument gate closed (positive control DIVERGES at thin pool;
ratified). Equivalence table: experiments/probing/actmix/
RM_EQUIVALENCE.md (checker excludes positive_control rows).

## Live overnight

- Chains: /workspace/logs/actmix_night_gpu{0,1}.log (monitor
  bva90ega9). GPU0: control✓ → btk T{6,10} sh0 → RM sh1. GPU1:
  RM T{6,8,10,16} sh0 → btk sh1. Telemetry:
  /workspace/logs/telemetry_rm/*.jsonl (250-step traces, both arms).
- Landed tonight: RM T6 s42 (0.8949 k20; twin diff = bidirectional
  drift, k5 −1.6e-2 / k20 +0.8e-3); btk T6 s42 (0.8941 — FIRST REAL
  T6). All cells telemetried except pre-existing T8/T16 btk ckpts +
  T16-s42 RM (endpoint-only, disclosed).

## Morning queue (in order)

1. Per-cell equivalence diffs as twins complete (rm_equivalence.py;
   report first divergences immediately — they ARE expected now,
   the interest is the ONSET CURVE at 3 seeds + T{8,10}).
2. Telemetry traces: parse telemetry_rm/*.jsonl → dead_frac +
   boundary_min_pre vs step per (arm, T) — the mechanism evidence
   (btk arm: min-selected going NEGATIVE marks boundary crossing;
   RM arm: fill below nominal marks waste).
3. 7-point per-k fig re-renders (--writeup final; ks auto-detect;
   T{6,10} rows fold in) + the REAL-T10-vs-phantom rebuttal note
   (CARD §7f wording) in the landing entry.
4. Certificate entry (measured scope ONLY, per guards): identity =
   {sae all, pre T1}; divergence map T2–T16 with per-T deltas;
   positive+negative control receipts; PTR.
5. RESULTS_relu-mix.md + RESULTS_btk-only.md refresh (analysis.py
   both arms); RM ledger actuals + corrections.
6. STATUS rewrite + push before compact.

## Standing

Timestamps: read `date` FIRST, then write the stamp (two drifts
tonight disclosed in the report). Origin watcher bf369am3s; night
monitor bva90ega9. LOG conflicts: union-resolve theirs-first +
stray-marker check (grep '^<<<<<<<|^>>>>>>> [0-9a-f]|^=======$').
Ledger: day-1 ≈ $98 + night ≈ $30 mostly day-2; caps intact.
Tokens by path only; rotations post-weekend. Aniket's backtracking
read-only.
