# REBUTTAL HANDOFF — where Dmitry (or his agent) finds every deliverable

**Deadline: rebuttal 13:00 BST 2026-07-28; exhibits READY BY 11:00 BST**
(Han's list, LOG ~00:35 entry). **THIS DOCUMENT SUPERSEDES the
meeting PDF (`private/meeting_tsweep_plots_2026-07-27.pdf`) as the
deliverable surface — plots are embedded below and refresh
automatically as morning re-renders overwrite the same paths.** Every number is PTR (pending team
ratification) unless marked ratified. Licences and caveats live in
`experiments/explorations/task_hunt/LOG.md` — search the stamp given
per item. Master data: `results/leaderboard.jsonl` (append-only; every
row carries `code_version` + `train_key`/`eval_key`). Arm labels:
`eval_cfg.arm` = `btk-only` (BatchTopK) or `relu-mix` (the
ReLU-bearing v2 composition; the paper's exact `txc_base`
rectify-after-select composition is covered by disclosure, not
retrained — see COMPOSITION_AUDIT).

---

## 1+2. Sparse probing shuffle/T-sweep, k=5 and k=20

![probing k5](figs_writeup/fig_probing_shuffle_tsweep_k5.png)
![probing k20 headline](figs_writeup/fig_probing_shuffle_tsweep_k20.png)

- **Figs:** `figs_writeup/fig_probing_shuffle_tsweep_k5.{png,pdf}`,
  `..._k20.{png,pdf}` (k20 = headline `fig_probing_shuffle_tsweep.*`;
  SAEBench-36 CT-excluded convention; `_38task` twin = robustness).
  7-point re-renders (T{1,2,4,6,8,10,16}) land by morning — same
  paths, overwritten.
- **Tables:** `figs_writeup/tab_probing_shuffle_tsweep_k5.md` +
  `_k20.md` (+.csv) — land WITH the morning render (directive of
  00:35).
- **Data:** leaderboard rows: `experiment=probing`,
  `arch=txc_batchtopk_pre_btkonly` (btk arm) /
  `txc_batchtopk_pre` (relu-mix arm),
  `training_cfg.arch_hparams_override.T` ∈ {1..16}, `eval_cfg.k_feat`
  ∈ {5,20}, `eval_cfg.shuffle` ∈ {none, within_window}, seeds
  {1,2,42}. Datasource `gemma_2_2b_it_l13_fineweb_24k128` (paper
  probe cache; gemma-2-2b-it L13).
- **Renderer:** runpod-1's analyze/make_writeup_fig pathway
  (`experiments/probing/actmix/` — see their morning render commit).
- **Licences (LOG stamps):** k-inversion quote licence (07-27 ~12:2x +
  ratifications); framing guard: level story leads; "probe-budget-
  dependent, no monotone window win at any k". Both-arms: the
  divergence-onset certificate (see item 8).

## 3. RLHF shuffle/T-sweep

![rlhf](figs_writeup/fig_rlhf_shuffle_tsweep.png)

- **Fig:** `figs_writeup/fig_rlhf_shuffle_tsweep.{png,pdf}` (FINAL
  3-seed; 7-point re-render lands ~morning, same path).
- **Table:** `figs_writeup/tab_rlhf_shuffle_tsweep.md` (with render).
- **ARCH (explicit, anti-confusion): the RLHF TXC exhibit arch is
  `txc_batchtopk_post_btkonly` — the plain windowed BatchTopK
  crosscoder (POST composition; probing uses PRE). It is NOT
  txc_pro (refutation receipts: LOG ~01:55). BINDING DISCLOSURE
  (LOG ~02:0x): the PAPER's RLHF TXC arm was `agentic_txc_02`
  (matryoshka-contrastive, per-window TopK→ReLU, k_win=500) — the
  exhibit is the plain-TXC modernization at the paper's window
  budget (k_win=100·T = 500 at the paper's T=5; per-window
  selection granularity preserved via the POST composition).
  T-sweep/shuffle conclusions are statements about the plain arm.
  Every RLHF caption carries this.**
- **Data:** `experiment=rlhf` rows, btk arm complete at T{1,2,5,8,16}
  + T{4,6,10} landing overnight (x4 via runpod-a swap-drain; x6/x10
  drain ~06:30). **Relu-mix arm: DONE-BY-CERTIFICATE — RLHF twins
  are tensor-IDENTICAL through T16 (829f05070: Δauc exactly 0,
  boundary_min_pre ≥ 2.21, no negative-pre-activation contact at
  RLHF's k-regime). The btk fig + the certificate line IS the
  both-arms deliverable; rmx_b T{8,10} cells run as eq-extension
  measurement points (per-cell checks).**
- **Licences (LOG):** 21:10 verdict extension (T8 peak n=3; T16
  regime boundary, decline 2-of-3 seeds; shuffle quote form "gaps ≈ 0
  at every T ≤ 8, seed-mixed at T16"); R-E1 lead licence.

## 4. Backtracking intensity λ̂ (hunted task #1)

![lambda](figs_writeup/fig_lambda_shuffle_tsweep.png)

- **Fig:** `figs_writeup/fig_lambda_shuffle_tsweep.{png,pdf}`
  (retrained overlay, anchor gate 6/6; 7-point re-render adds
  T6/T10).
- **Table:** `figs_writeup/tab_lambda_shuffle_tsweep.md` (with
  render).
- **Data:** hunt-width cells (d_sae 2048) — overlay card
  `SHUFFLE_OVERLAY_CARD.md` + T_FILL card (c09485d1c) under
  `experiments/explorations/task_hunt/`; substrate
  `ward_real_lambda_base_l12` (R1-Distill-8B L12).
- **CAPTION-BINDING flags (LOG 00:01 + 00:18):** T6 dip below T4;
  T10 seed-fragile — VENUE-LOCALIZED training instability (same
  seed/T trains fine on dq). Both-arms: R30 identity at hunt widths
  (|Δ| ≤ 2.2e-8) + T16 spot-check pair (runpod-a drain).
- **Deep licences:** REBUTTAL_PACK.md §1 (R22 margin + mandatory
  disclosures; arm guard pre-vs-post).

## 5. Question-gap dq (hunted task #2)

![dq](experiments/explorations/task_hunt/figs_writeup/fig2_question_gap_tscaling.png)

- **Fig:** `experiments/explorations/task_hunt/figs_writeup/
  fig2_question_gap_tscaling.{png,pdf}` + fills (T6/T10 on-plateau,
  88cb4f867).
- **Table:** `figs_writeup/tab_dq_tsweep.md` (with morning render).
- **Caveats:** TOY-class per Dmitry's bar (meeting 07-27) —
  within-SAE use only; shuffle columns are SCREEN-class (overlay
  ruled out, LOG 00:05); passed-then-demoted framing.

## 6+7. Safety-relevant hunted tasks (THE GOLD — status, not yet exhibits)

- **State (00:50, corrected per mac-c's kill triage 2505cd937):**
  every found-corpus candidate resolved (kills with receipts —
  reask_hr 3/3, retryesc vocabulary bar fired LABEL-SIDE, warddebt
  geometry; see LOG). Salvage classes: retryesc = signal-UNTESTED
  (its probe never ran — the regenerated corpus tests a genuinely
  open question, not a shown phenomenon); dharm + warddebt =
  STRUCTURALLY UNSCREENABLE (rebuild-or-nothing). The ELICITATION
  HARNESS: scaffold frozen; Claude-API backend committed
  (a0646af0d) with the provenance weakening honestly disclosed
  in-card (bit-exact → reproducible-in-expectation); **evalage
  generation STARTING NOW; sycgen_age and retryesc_gen queued;
  mac-d pulls one card for parallelism.** First KEEP auto-triggers
  the full matrix retrain (pre-authorized).
- **Where verdicts will appear:** LOG (stamped entries) +
  `experiments/explorations/task_hunt/<candidate>/` cards/results.
- **For the 13:00 submission:** the honest sentence is a promised
  amendment — "dedicated safety-task experiments are running;
  results follow within the amendment window (Aug 3)." Exhibits
  land in that window, not by 11:00 (a screen KEEP + 7-T × 3-seed ×
  both-arms retrain is >12 h of pipeline from a standing start).

## 8. tsae width-match (additional item) — COMPLETE ✓

- **Verdict + quote-form:** LOG 00:18 entry ("no improvement —
  dictionary width was not what limited it… probing width only").
- **Probing:** `experiments/probing/actmix/WIDTH_MATCH_TSAE_CARD.md`
  + runner `width_match_tsae.py`; rows: `arch=tsae_btkonly` with
  `arch_hparams_override.d_sae=18432`, seeds {1,2,42} — NO LIFT
  (k20 0.8708 vs paper-width band 0.8718±0.0008).
- **RLHF:** was NEVER narrow — receipts in
  `experiments/rlhf/…/actmix_rlhf/results/papermatch.json` (shipped
  ckpts @18432) + CARD §7 A4 (runpod-a); 3-seed set complete
  (k500 0.621±0.004, k20 0.600±0.002).

## 9. Both-arms / composition certificate (underpins every item's arm claim)

- **Onset map:** identity = sae×3 + pre-T1 (machine precision);
  divergence from T=2, growing with window depth; T16 ~40% disjoint
  survivor sets, ≈0.002 AUC at d_sae 18432. Table + per-cell
  receipts: `experiments/probing/actmix/RM_EQUIVALENCE.md` (incl.
  ALIAS EXCLUSION LIST — any future arm-diff must exclude those
  train_keys). Morning: 3-seed onset map + boundary_min_pre traces +
  certificate (PRELIMINARY until then). Pack §3 carries the
  redrafted both-arms licence.
- **RLHF regime differs — now FULLY CERTIFIED:** twins
  tensor-identical through T16 (829f05070; pre-registered
  divergence refuted and disclosed). Unified mechanism frame:
  rare between-sample boundary contact (probing) vs no contact
  (RLHF) — one mechanism, two measured regimes. Quote per-task,
  never cross-task.
