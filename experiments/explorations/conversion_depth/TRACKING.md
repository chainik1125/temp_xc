# TXC-tracking (em-redo Phase A) — does a trained temporal code claim the measured EM headroom?

**Agent:** runpod-c (H100). **Briefing:** `briefings/em-redo.md`.
**Context:** `RECORD.md` § 4 (the g(ℓ) map + decomposition) + § 7 review;
`docs/ideas/em_onset_anticipation.md`. Prime directive: a sound verdict,
never a win. This file's §§ 1–2 are FROZEN at the freeze commit, before
any cache build or training run — commit order is the evidence.

## § 0 — The question

The depth ablation (RECORD § 4) measured raw window-over-token access
for the rollout-misalignment label on the § 5.3 stage-4 cohort:
g(ℓ) = +0.120 / +0.134 / +0.097 AUC at resid_post L9 / L13 / L15, with
the position-sensitive slice g_order = +0.034 / +0.108 / +0.054. The
paper's § 5.3 negative (TXC underperforms T-SAE/per-token on Wang-style
steering + sparse-probe PR-AUC at L15) stands as an empirical result;
what fell is its ambience explanation. Test: train the dictionary panel
at the three map-chosen layers and see whether any temporal
architecture's advantage TRACKS the map.

## § 1 — Substrate + protocol (frozen)

**Training stream** — the paper's § 5.3 recipe (verified against
`origin/final:purified/configs/datasources.yaml` +
`qwen_em.py`): `cfierro/personality-qs-bad-medical-advice`, seed-42 row
shuffle, chat-template render, truncate+pad to 128, add_special_tokens
False, 6000 rows. **One deliberate deviation, briefing-directed:** the
forward runs through the MERGED organism (Qwen2.5-7B-Instruct +
`andyrdt/Qwen2.5-7B-Instruct_bad-medical`, `merge_and_unload`) — the
origin/final builder cached BASE activations (its `lora_adapter` was
consumed only at Wang/detection time), i.e. the paper trained
dictionaries on base activations and applied them to organism
activations at detection. The g(ℓ) map lives on organism activations;
the tracking test must train on the same substrate. Anchor cells (§ 2)
quantify the bridge to the paper's base-trained numbers. Recorded
property of the paper's corpus convention, reproduced deliberately:
median row = 53 tokens → **≈59% of cached token positions are
eos-pad** (`build_em_train_cache.py`).

**Layers:** resid_post L9 / L13 / L15 (hs 10/14/16); datasources
`qwen_2_5_7b_organism_medical_l{9,13,15}` (configs/data.yaml).

**Panel** (`em_redo_cells.py`, the frozen cell table): batchtopk_sae,
txc_batchtopk_post (k_pos 80/window = 20/token reuse parity),
txc_batchtopk_pre (k_pos ∈ {20, 40}, realized-nearest-20 selected at
analysis), tsae — all d_sae 32768, matched nominal per-token budget
20 atoms/token, realized l0_per_token measured per cell (Part II
discipline; loose match ≥25% off target flagged). 3 seeds {42, 1, 2},
3 layers → 45 panel cells. Training: n_steps 25 000, lr 3e-4, warmup
1000, bf16, batch 1024 (tsae: 32 sequences = 4096 token-positions/step,
matching the T=4 window archs' per-step token count; the paper's own
c6 pairing was also unequal here — txc_base saw 5120 positions/step vs
sae_arditi's 1024). No Bricken on the panel (the fair-backbone suite
carries its own AuxK/unit-norm/grad-orth stack uniformly).
**Anchors** (seed 42 × 3 layers): txc_base (paper knobs incl.
brickenauxk_a8) + sae_arditi — the paper's own pairing on the new
substrate. spectral_txc: stretch only, budget permitting.

**Currency 1 — paper (detection 3.0.0):** `src/temp_bench/evals/em.py`,
a code-faithful port of `origin/final` `detect_case_study` +
`c6_em_detection/run.py`: stride-1 T-windows (T = arch T) over the
1728-rollout cohort's assistant tokens, encode → |z| → amax pool,
top-S by train-fold |mean-diff|, L1 LogReg (C=1, liblinear, seed 42),
PR-AUC (average precision), GroupKFold(5) by prompt,
S ∈ {1,2,4,8,16,32}, within-window shuffle ablation for T>1.
**Primary = pr_auc_S16** (the paper's Fig convention). Runs inside the
canonical runner → leaderboard rows (evaluator "em", protocol 3.0.0).
Mechanical deviation from the origin/final driver: reads the
sidecar-verified cohort activation cache (`cache_em_cohort3.py`,
identical cohort/order as phase 4) instead of re-forwarding rollouts
per cell.

**Currency 2 — probe (`probe_codes.py`):** the raw-map's own rows and
stack — stride-4 right-edge T=16 windows, GroupKFold(4) qid%4, frozen
problib linear probe, rank-AUC. Per-token read = the arch's finest code
at p; window read = amax|z| over the 16-window (details in the script
docstring). Trained window advantage A = AUC(win) − AUC(tok); directly
subtractable from RECORD § 4's raw ceilings (same rows/folds):
raw tok = 0.645/0.673/0.748, raw win = 0.765/0.807/0.845 at L9/L13/L15.

**Analysis rules (frozen):** temporal advantage over per-token at layer
L, per currency: δ(arch, L) = metric(arch, L) − metric(batchtopk_sae, L),
mean over seeds; pre uses the realized-nearest-20 k per layer (chosen by
mean realized l0_per_token over seeds; the other k reported as
robustness). Seed spread reported as max−min. Machinery falsifier: if
batchtopk_sae pr_auc_S16 at L15 < 0.40 (vs positive-rate baseline
0.323), the pipeline is broken — debug, do not interpret. Dirty-tree
stance: freeze commit pins the code; the sweep runs
TEMP_BENCH_ALLOW_DIRTY=1 because leaderboard appends dirty the tree
(established practice: 7031/7116 existing rows carry dirty=true).

## § 2 — Frozen predictions (committed before any training)

Mac-local's priors (briefing), sharpened:

- **P1 (per-token layer profile):** batchtopk_sae tracks the RAW
  per-token curve, not the g map — its metric at L15 ≥ its metric at
  L13 in both currencies (mean over seeds). The g-map peak at L13 is
  window headroom, invisible to a per-token reader.
- **P2 (temporal advantage tracks the map):** for at least one of
  {txc_post_k80, txc_pre_k*}, δ(L13) > δ(L15) AND δ(L13) > δ(L9) in the
  probe currency (mean over seeds). Direction bet, lower confidence:
  the same ordering appears in pr_auc_S16.
- **P3 (shuffle gap at the g_order peak):** any TXC cell whose window
  advantage is genuinely temporal shows shuffle_gap_S16(L13) >
  shuffle_gap_S16(L15); the paper's own decision threshold (gap ≥ 0.02
  across S) is the bar for claiming temporal detection.
- **P4 (pre-registered weak-realization branch):** if NO temporal arch
  beats batchtopk_sae anywhere (all layers, both currencies, mean over
  seeds) despite raw g ≥ +0.097 everywhere, that is the third
  weak-realization datum (after § 5.2's reader-predictability and the
  paper's § 5.3 loss) — reported as a designed outcome: trained
  temporal codes do not (yet) realize measured window headroom.
- **P5 (tsae is per-token in readout):** |δ(tsae, L)| ≤ 0.02 at every
  layer in both currencies — its temporal contrastive shapes training,
  not the readout geometry.
- **P6 (anchors, direction bets):** organism-trained sae_arditi at L15
  lands within ±0.05 of the paper's base-trained pr_auc_S16 (substrate
  shift is modest for a 128-atom sparse readout); txc_base's shuffle
  gap stays ≈ 0 at L15 (|gap| < 0.02, reproducing the internal
  diagnostic that seeded the ambience gloss) but exceeds +0.02 at L13
  if the g_order slice is real for trained codes (this clause is the
  same bet as P3 restated on the paper's own arch).

Falsifier accounting: every P above is scored in § 4 verdicts; misses
reported as plainly as hits.

## § 3 — Runs (filled during execution)

**Post-freeze, PRE-RESULTS annotation (2026-07-23, before any panel
cell ran):** extracting the paper's four medical detection rows
(`origin/final` leaderboard, protocol 3.0.0) for the anchor comparison
revealed a confound the frozen P6 did not account for: the paper's
cells each probed their OWN Wang-stage-4 cohort (n_sent 79k–107k,
positive_rate 0.410/0.466 for sae_arditi s42/s1, 0.315/0.345 for
txc_base) — per-cell cohorts with different base rates — whereas this
redo probes every arch on the ONE fixed 1728-rollout cohort
(positive_rate 0.323, the g-map's substrate; the cleaner cross-arch
design). PR-AUC is base-rate sensitive, so P6's ±0.05 window vs the
paper's sae_arditi mean 0.7175 (measured at base rates 0.41–0.47) may
miss for cohort-composition reasons alone. P6 stays as frozen and will
be scored as written; this note pre-registers the base-rate explanation
BEFORE results exist rather than reaching for it after. Paper anchor
numbers (pr_auc_S16): sae_arditi 0.690 (s42) / 0.745 (s1); txc_base
0.542 / 0.560 with shuffle_gap_S16 −0.059 / −0.002.

(cell log: `run_em_panel.py` wall log + leaderboard rows)

## § 4 — Verdicts vs § 2 (blind: written against the frozen text above)

(to be filled)

## § 5 — Phase B: onset pilot gate (stretch)

Feasibility pinned pre-freeze: the stage-4 cohort's α=0 native subset =
192 rollouts, 67 misaligned — above the briefing's ~40 STOP threshold.
Design per `docs/ideas/em_onset_anticipation.md` steps 1–4 only; its
own prereg lands in this file's § 5 BEFORE the labeler runs, if Phase A
completes in budget.
