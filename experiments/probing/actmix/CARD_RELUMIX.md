# ACTMIX P1-RM CARD — § 5.1 sparse probing, relu-mix arm T-sweep (both-arms comparison)

Pre-registration. Directed by 6166c0293 § 3c (meeting outcome: "the
comparison the meeting asked for is NEW WORK"). Frozen with
`launch_relumix.sh` in one commit before any cell; PIN recorded at
launch. Everything not stated here inherits the btk-only card
(`CARD.md`) verbatim — same harness, same eval protocol 1.2.0, same
shuffle convention, same datasource
(`gemma_2_2b_it_l13_fineweb_24k128`), same aggregation conventions
(36-task CT-excl headline / raw-38 twin, per 89fd5c292).

## 1. Compositions (the ONLY change vs CARD.md)

Archs = the UNSUFFIXED registry entries — per the canonical
single-source convention (configs/archs.yaml note + mac-a's LOG
convention note: "these entries = btk-only; the unsuffixed names =
the relu-mix arm"):

- `batchtopk_sae` (per-token band)
- `txc_batchtopk_pre` (headline T-sweep, k_pos·T budget)
- `txc_batchtopk_post` (companion, k_pos/window)

ReLU in the sparsity path = the PAPER composition (audit § pins).
tsae relu-mix NOT in tonight's grid (directive names the sae/pre
comparison; tsae carries 7d serving caveats in both arms —
post-deadline if wanted).

## 2. Grid + queue

T ∈ {1,2,4,8,16} × seeds {1,2,42} × k_feat {5,20}, eval_cfg arm =
`relu-mix` (hashes into eval_key; rows disjoint from btk-only).
Queue (fail-fast, as CARD.md § 3): untrained twins seed 42 →
batchtopk_sae 3 seeds → txc_batchtopk_pre 3 seeds (42 first,
endpoints first) → txc_batchtopk_post seed 42 → post seeds 1/2
TRAIL (same cut clause as btk-only; they ran there, so
symmetric-if-clock-allows here). AMENDMENT-1 exposure-matched
window batches (4096/T) inherited — REQUIRED for comparability
with the btk-only curves.

## 3. Pre-registered directional expectations (before any cell)

RM-E1. relu-mix levels sit BELOW btk-only at matched (arch, T, k)
  — the btk card's own § 4 premise ("sae improves most" under
  btk-only) implies the reverse direction here; paper-arm anchors:
  SAE 0.8831 (38-raw, v1 ckpts) vs btk-only 0.8993.
RM-E2. Realized l0 ≤ nominal per window (ReLU zeroes negative
  selections) — bands: report-class [0.5·nominal, nominal+0.5],
  DISCLOSE-not-gate below nominal (the btk lower-bound guarantee
  does not exist here); over-admission above nominal+0.5 still flags
  (threshold-eval artifact, as btk G1).
RM-E3. T=1 anchor: |txc_pre@T1 − batchtopk_sae| ≤ 3σ_SAE (G5
  analog) — the anchor property is composition-independent.
RM-E4. The T-shape: paper's shipped claim direction (T improves)
  vs btk-only's flat-then-decline — OPEN. Both directions on
  record pre-run: if relu-mix ALSO shows no T-win, the § 5.1
  T-claim loses its last composition; if relu-mix rises where
  btk-only declined, the ReLU sparsity path is the T-win's
  load-bearing component (Dmitry's d(perf)/dT gate answered
  per-composition). k-inversion (fcf62963b): check both k.

## 4. Gates

G1–G5 as CARD.md with RM-E2's band replacing the btk l0 band.
Untrained twins seed 42 (G3). 38-task suite integrity (G4). Anchor
G5 both k. Shuffle identity per-token exact (G2).

## 5. Deliverable

`figs_writeup/fig_probing_shuffle_tsweep_botharms.*` — btk-only vs
relu-mix TXC-pre, ordered + shuffled, 36-task headline, same
template knobs (pair-hue per meeting pick when ruled; until then
mono with arm distinguished by hue family btk=#D55E00 vs
relu-mix=#0072B2, linestyle = order condition, disclosed in
caption) + RESULTS_relu-mix.md via analysis.py --arm relu-mix.

## 6. Venue, economics, discipline

MY GPUs 0,1 at post-shard drain (~18:40 real); runpod-a's GPU 0
joins ONLY by LOG agreement (6166c0293 offers it) — shard-count
adjusts at launch, disclosed in the launch line. Est 13–14 GPU-h
(sae 3 + pre 15 + post-42 5 trained cells + untrained pass + evals
at measured btk rates) ≈ **$40–45**; day-2 cap $150 fresh
(post-midnight cells bill to 07-28). Ledger line at launch;
TEMP_BENCH_ALLOW_DIRTY=1 after PIN assert (pool-row convention);
rows checkpointed at milestones; PENDING TEAM REVIEW on every
verdict-class statement.
