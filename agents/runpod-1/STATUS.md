# Working state — agent `runpod-1`

**2026-07-27 ~12:45 London — governing directive 059a66239 (POD
SATURATION 10h, 12:00→22:00). Shared 3×H100 pod, GPUs 0,1 mine (GPU 2
= runpod-2; borrow only by LOG coordination). P1: probing grid drain +
INTERIM fig DONE → FINAL fig + formal verdict owed; P2: layer-sweep
share CLAIMED (sweep a). Report state by ~21:30 London.**

## Live state (check first on resume)

- **Grid** (`/workspace/logs/actmix_p1_gpu{0,1}.log`, PIN-lineage from
  131ea677f): pre 2-seed curve COMPLETE (s42+s1 all T); s2 cells in
  flight (GPU0: s2/T1→T2→T8; GPU1: s2/T16 mid-train →T4). Pre drains
  ~14:00 → **FINAL writeup fig then**. Post-42 pass next (~15:20-15:40)
  → **formal verdict entry then** (scaffold:
  scratchpad/verdict_draft.md). Post-1/2 runs to ~18:00 (card allows
  cut if clock demands). tsae b32 direct (pid 45188,
  `actmix_tsae_b32_direct.log`): s42 evals ~now, then s1, s2 (~25
  min/cell).
- **Analysis**: `experiments/probing/actmix/analysis.py` — FREEZES is
  now a freeze-lineage allowlist (RATIFIED 36655341a); `--writeup
  {interim,final}` renders `figs_writeup/fig_probing_shuffle_tsweep.*`
  knob-for-knob with runpod-2's frozen RLHF template (421f6fa37).
  INTERIM (2 seeds) pushed 5f21474c3 — 17:00-draft ready. Hue knob
  (single pair-hue vs blue-vs-orange) decided AT the 17:00 meeting —
  apply at FINAL only.
- **Interim numbers** (k20 pre, seeds {42,1}): T1 0.8992±0.0023
  (anchor |Δ|=0.0001 vs SAE 0.8993±0.0032), T2 0.9015, T4 0.8997,
  T8 0.8898, T16 0.8768; order-gap 0→.0077→.0196→.0296→.0223.
  Inverted-U confirmed. G1: l0 over-admission +3-5% uniform, +19%
  @T16 (A1 batches) — verdict must carry "decline despite extra
  capacity".
- **P2 sweep (a) CLAIMED** (LOG 5f21474c3): ttrend+cnov labels,
  llama31-8B L{7,14,21,28} + gemma2 L{6,13,20}, screen class, hunt3
  discipline (one card per sweep, scorer-before-results, screens only).
  λ̂-Ward = runpod-2's 20:00 slot unless I post a LOG update before
  19:45. Explore scout mapping label/cache/layer plumbing — check its
  result, then draft card, freeze scorer, launch at GPU slack (~18:00
  after post-1/2, or ~15:40 if post-1/2 cut).
- **Monitors**: origin watcher bf369am3s (four-path topology, keep as
  THE listener); grid watcher biki784bc; re-arm on every wake per
  directive.

## Done today (all pushed + ratified)

1. Phase B complete + ratified overnight (A12 T5-artifact closed by
   reproduction; every printed §5.1 number reproduced incl. σs).
2. Phase A: SAE band 3-seed 0.8993±0.0032; pre 2-seed curve complete;
   G5 anchor PASS k20 |Δ|=0.0001; untrained twins ~0.70 band.
3. FREEZES lineage fix + §7e no-extension note (both RATIFIED
   36655341a); tsae pgrep-self-match unstuck (launched direct b32).
4. INTERIM fig_probing_shuffle_tsweep pushed (5f21474c3).

## Wrap-up chain (in order)

1. ~14:00 pre drains → `analysis.py --arm btk-only --writeup final` →
   eyeball → commit/push figs_writeup FINAL.
2. ~15:40 post-42 + tsae drain → full analysis both arms → verdict
   entry from scaffold (CARD §4 quoted verbatim, E1-E4/G1-G5, dual
   convention, coverage honesty, PENDING TEAM REVIEW) + ledger actuals
   (RUNPOD section; ~$39 tsae sunk + ~$9 early + grid) → push.
3. P2 sweep (a): card → freeze scorer → launch (nohup, ledger line).
4. 21:30 report (LOG) + STATUS rewrite + push.

Tokens: paths only (/workspace/.tokens/*), rotate post-weekend.
Aniket's backtracking + origin/neurips-aniket = read-only. Never touch
GPU 2 pids.
