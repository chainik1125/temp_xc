# CODE GUIDE — sparse-probing & RLHF shuffle ablations (for Dmitry's agents)

Companion to `REBUTTAL_HANDOFF.md` (the deliverable index). This file
answers: which code produced the two headline shuffle-ablation
exhibits, where the numbers live, where the checkpoints are, and what
each label means. Everything is on branch `arxiv`. Licences/caveats
are stamped entries in `experiments/explorations/task_hunt/LOG.md`
(cited by time-stamp below); all numbers PTR unless marked ratified.

---

## 0. The one canonical pathway

Every result row goes through `temp_bench.core.runner.run_experiment`
(hard rule 1 — there is no other leaderboard writer). Rows land
append-only in `results/leaderboard.jsonl`, stamped with
`code_version.{commit_sha, dirty, diff_sha256}`, `train_key`,
`eval_key`. Archs are plugins: one class file under
`src/temp_bench/archs/` + a YAML entry in `configs/archs.yaml`.

## 1. The architectures actually used (and what the arm labels mean)

| exhibit | arch id | class (file) |
|---|---|---|
| probing TXC | `txc_batchtopk_pre` / `txc_batchtopk_pre_btkonly` | `TXCBatchTopKPre` (`src/temp_bench/archs/txc_batchtopk.py:296`) / `TXCBatchTopKPreBTKOnly` (`src/temp_bench/archs/btk_only.py:194`) |
| RLHF TXC | `txc_batchtopk_post_btkonly` | `TXCBatchTopKPostBTKOnly` (`btk_only.py:208`) |
| **probing paper-faithful (the sprint)** | `paper_txc_base_v1t` | `PaperTXCBaseV1T` (`src/temp_bench/archs/paper_v1t.py:181`) — vendored 94119bc08 training stack VERBATIM + thin v2 wrapper; arm `paper-faithful`; card `experiments/probing/actmix/CARD_PAPER_FAITHFUL.md` (21 cells RUNNING across 4-5 GPUs, ETA ~06:30–07:30) |
| RLHF paper-faithful | agentic_txc_02 trainable port | runpod-2 building; card ETA ~04:30 (vendor pattern, `experiments/explorations/actmix_rlhf/vendor/`) — cells land after |
| baselines | `batchtopk_sae(_btkonly)`, `tsae_btkonly` | same two files; the T-SAE class legitimately carries matryoshka+contrastive (Ye et al.'s design) |

- **Arm labels** (`eval_cfg.arm`): `btk-only` = BatchTopK with NO ReLU
  in the sparsity path; `relu-mix` = the ReLU-bearing v2 composition.
  These are plain windowed BatchTopK crosscoders — **NOT txc_pro**
  (no matryoshka/contrastive/subseq anywhere in these classes; LOG
  ~01:55 refutation receipts).
- **Paper-arch relation** (binding disclosures): the paper's probing
  TXC (`txc_base`) is plain family but composition-distinct from BOTH
  v2 arms: paper = ReLU(TopK_{k_pos·T}(Σ_t preact)) per-window
  exact-k; v2 pre = per-position select-then-sum; v2 post =
  sum-then-BatchTopK (batch budget). The paper-EXACT composition is
  served by the eval-only adapter `paper_txc_base_v1`
  (`src/temp_bench/archs/paper_v1.py:255`, upstream 94119bc08) over
  the ARCHIVED ckpts — T=5 × 3 seeds ONLY. The paper composition is
  ALSO now TRAINABLE at every T via `paper_txc_base_v1t` (the sprint).
  **TABLE-LABELING RULE (binding, amended 02:58 07-28): matrix columns
  are "{ReLU+TopK} paper-faithful" (= `paper_txc_base_v1t` cells, arm
  `paper-faithful`) and "{BatchTopK}" (= `*_btkonly` cells, NO ReLU);
  the archived-T5 paper ckpts stay a separate "paper base (archived,
  T=5)" anchor row; relu-mix cells are certificate evidence ONLY and
  never appear as a matrix column; never conflate any of the four.**

  **The three compositions, exactly** (p_t = W_enc,t x_t + b_enc;
  k_win = k_pos·T; B = batch):
  - paper base: `z = ReLU( TopK_{k_win}( Σ_t p_t ) )` — exact
    per-window k, rectify AFTER selection.
  - v2 post (relu-mix): `z = BatchTopK_{k_win·B}( ReLU( Σ_t p_t ) )`
    — select-after-sum like the paper, but ReLU BEFORE selection +
    batch-level expected budget. btk-only twin: no ReLU (signed
    selection). Eval uses the EMA-threshold gate.
  - v2 pre (relu-mix): `z = Σ_t [ BatchTopK_{k_win·B}( ReLU(p) ) ]_t`
    — selection at (position, latent) granularity BEFORE pooling;
    the window code sums surviving per-position contributions.
    btk-only twin: no ReLU.
  Consequences are MEASURED, not assumed: relu-mix↔btk-only =
  bit-identical at T1 / ~1e-2 divergence from T2 (probing) / exact
  identity through T16 (RLHF certificate); realized-l0 receipts
  quantify the budget-convention difference.
  The paper's RLHF TXC was **`agentic_txc_02` =
  `MatryoshkaTXCDRContrastiveMultiscale`** (matryoshka+contrastive,
  multiscale shifts, per-window TopK→ReLU, k_win=500) — a DISTINCT
  class from txc_pro (no subseq curriculum/k-asymmetry). Our RLHF
  exhibit is the plain-TXC modernization at the paper's window budget
  (k_win=100·T = 500 at the paper's T=5), per-window selection
  granularity preserved via the POST composition. Full pins:
  `experiments/explorations/task_hunt/COMPOSITION_AUDIT.md` §0/§3/§6.
  (Provenance status: **INDEPENDENTLY VERIFIED** — hub 5-leg
  re-derivation, LOG 03:09 07-28: runner-source STAGE_1_ARCHS pin,
  dev↔release byte-identity of all four `top_features.json` blobs
  (agentic = `12a873891a…`), the blob's self-declared
  `src_class: MatryoshkaTXCDRContrastiveMultiscale`, the
  TopK→ReLU-per-window encode at the dev commit, and the seed-42
  ckpt live on HF `txcdr-base`.)
  The RLHF renderer prints this disclosure below the axis on every
  fig.

## 1b. THE CELL CENSUS — every cell we have results for, by arm

**`REBUTTAL_CELL_CENSUS.md` (repo root) lists every leaderboard cell
in rebuttal scope** — one line per (arch, datasource, T): trained
seeds, untrained-twin seeds, k budgets, shuffle coverage, row counts —
each labeled **{ReLU+TopK} PAPER-FAITHFUL** vs **{BatchTopK}
(btk-only, NO ReLU)** vs **relu-mix (the MISINTERPRETED
"{ReLU+TopK} paper-faithful" arm — certificate evidence only)**, plus
the hunted-task cells that live outside the leaderboard. It is
generated, not hand-written: refresh with
`.venv/bin/python scripts/cell_census.py --write` — cells are landing
all night (paper-faithful sprint, sycgen retrain), so regenerate
before quoting coverage. State at 07:20: **probing paper-faithful
COMPLETE 21/21** (last 3 cells hub-repatriated from pod B, LOG
~07:1x); probing btk-only COMPLETE 7 T × 3 seeds; sycgen retrain
rows on-board (15 trained + twins). Remaining gaps: RLHF
paper-faithful grid (pilot→G1 in flight), RLHF btk T{6,10}
(DEFERRED by Han's pf-priority order — T6 holds 2/3 seeds as
partial-bonus).

## 2. Sparse probing — code, data, results

- **Experiment dir:** `experiments/probing/actmix/` — start with
  `CARD.md` (the frozen sweep card), `CARD_PAPER_FAITHFUL.md` (the
  paper-faithful sprint card, PIN d9235755b — plugin contract tests,
  21-cell grid, 5-GPU shard split), `CARD_RELUMIX.md` (the relu-mix
  arm grid), `sweep.py` (cell launcher), `analysis.py` (aggregation +
  `make_writeup_fig` renderer), `prep_cache.py` (substrate).
- **Substrate:** gemma-2-2b-it L13 resid_post, the paper's probe
  cache — datasource `gemma_2_2b_it_l13_fineweb_24k128`; 38 cached
  SAEBench tasks, rebuttal-headline aggregation = SAEBench-36 (CT
  pair winogrande/wsc excluded). **PANEL MAPPING (T=5 sanity, LOG
  12:5x): the PAPER's fig:sparse_probing headline is the 38-task
  mean — archived T=5 ckpts read 0.8975±0.0039 there (matches the
  published 0.899–0.902) vs 0.9248±0.0033 under the rebuttal's
  36-task convention. Never cross-quote panels.**
- **The shuffle instrument:** evaluator `probing` protocol **1.2.0**
  (`src/temp_bench/evals/probing.py`); `eval_cfg.shuffle` ∈ {none,
  `within_window`}, `eval_cfg.shuffle_seed=0`. Within-window = token
  order permuted inside each T-window before encoding; T=1 shuffle ≡
  identity by construction.
- **Row filter for the exhibit** (see also handoff §1+2):
  `experiment=probing`, `evaluator_protocol_version=1.2.x`, arch ids
  above, `training_cfg.arch_hparams_override.T` ∈ {1,2,4,6,8,10,16},
  `eval_cfg.k_feat` ∈ {5,20}, seeds {1,2,42}. k_win = 20·T
  (constant per-token budget — the program-wide sparsity convention).
- **Figs/tables:** `figs_writeup/fig_probing_shuffle_tsweep{,_k5,_k20,_38task}.*`
  + `tab_*` twins (morning 7-point render). Licences: k-inversion
  (probe-budget-dependent, no monotone window win at any k); level
  story leads.
- **Equivalence/arm evidence:** `experiments/probing/actmix/
  RM_EQUIVALENCE.md` (+ `.json`, `rm_equivalence.py` checker) — the
  onset map (identity = sae + pre-T1; divergence from T=2, growing),
  the ALIAS EXCLUSION LIST (train_keys any arm-diff must exclude),
  `positive_control.py` (thin-pool forced-divergence instrument
  gate), `src/temp_bench/archs/telemetry.py` (opt-in
  `boundary_min_pre` traces via `TEMP_BENCH_TELEMETRY_DIR`).

## 3. RLHF — code, data, results

- **Experiment dir:** `experiments/explorations/actmix_rlhf/` —
  `CARD.md` (frozen card incl. the A5/A5b relu-mix rulings),
  `cells.py` (lanes), `build_cache.py`/`convert_train_cache.py`
  (HH-RLHF substrate), `analyze.py` + `render_writeup_fig.py`
  (fig+table renderer; carries the agentic_txc_02 disclosure
  footnote), `papermatch.py` + `results/papermatch.json` (shipped-
  ckpt provenance receipts), `rlhf_equivalence.py` +
  `RLHF_EQUIVALENCE.md`/`.json` (the identity certificate),
  `hf_durability_push.py` (ckpt mirroring).
- **Substrate:** Anthropic/hh-rlhf preference pairs, gemma-2-2b BASE
  L12 residuals (the paper's exact cache), preference ROC-AUC probe;
  k_win = 100·T.
- **Row filter:** `experiment=rlhf`, `datasource=
  gemma_2_2b_base_l12_phase7`, arch `txc_batchtopk_post_btkonly`
  (+ baselines), T ∈ {1,2,4,5,6,8,10,16} (T5 = the paper's operating
  point, kept as bonus), seeds {1,2,42}. NOTE the evaluator is
  `rlhf 2.0.0` and its rows carry an EMPTY `eval_cfg` — probe budgets
  are metric-name-encoded (`metrics.preference_auc_k20` / `_k50`) and
  the shuffle twins are in-row (`metrics.shuffled_*`,
  `metrics.shuffle_gap_auc_k20`); there is no `eval_cfg.k_feat` /
  `eval_cfg.shuffle` on RLHF rows (that is probing-1.2.x semantics).
  T=1 rows legitimately omit `shuffled_*` (shuffle ≡ identity).
  **Substrate identity (added 05:05 07-28, G2 catch):** newer RLHF
  rows carry `eval_cfg.hh_rlhf_cache` — a registry TAG
  (`l12base_phase7` / `l13it_paper`) hashing into the eval_key,
  with a `cache_expect` {subject, layer} hard-check before any
  metric. Empty-`eval_cfg` rows are historical (l12base_phase7 by
  default). Never compare rows across cache tags as
  same-substrate.
- **Fig/table:** `figs_writeup/fig_rlhf_shuffle_tsweep.*` +
  `tab_rlhf_shuffle_tsweep.md` (morning 7-point render). Licences:
  order-free inverted-U, T8 peak; shuffle gaps ≈ 0 at T ≤ 8, seed-
  mixed at T16 (quote form in LOG 21:10 + ratification 22:28).
- **Both-arms:** `RLHF_EQUIVALENCE.md` — twins tensor-IDENTICAL
  through T16, Δauc exactly 0, mechanism `boundary_min_pre ≥ 2.21`
  (no negative-pre-activation contact at RLHF's k-regime). The btk
  fig + this certificate IS the both-arms deliverable; rmx_b T{8,10}
  cells run as eq-extension measurement points.

## 4. Checkpoints

- **Durable mirror:** HF dataset `han1823123123/temp-bench-data`,
  path `ckpts/<train_key>/model.safetensors` (LFS sha256 = receipt).
  Lookup: leaderboard row → `train_key` → that path. Uploader:
  `scripts/push_ckpts_hf.py` / `hf_durability_push.py` (RLHF).
- **Mirrored now:** all 26 trained RLHF ckpts; 30 probing
  certificate-evidence ckpts (twin pairs across the 7-T grid, sae
  pair, positive control; spot-check MATCH). Remainder mirrors in
  priority order; pod-local at `checkpoints/<train_key>/`, indexed
  by `checkpoints/manifest.jsonl` (tracked).
- **Paper-era ckpts:** `han1823123123/txcdr-base`
  (`<arch_id>__seed42.pt`, incl. `agentic_txc_02`) +
  `temp-bench-models` (c3 cells) — see COMPOSITION_AUDIT §3.

## 5. FLEET MAP — what is running on every pod (snapshot 07:20 BST 07-28; probing pf grid COMPLETE)

This section dates fast. **Live sources: `agents/<id>/STATUS.md`
(each agent self-maintains its own) + the LOG tail** — trust those
over this snapshot if they disagree.

| pod | agents | GPU | running NOW | next |
|---|---|---|---|---|
| **old pod** (3×H100) | runpod-1 (GPU 0/1), runpod-2 (GPU 2) | 0+1 | drained (shards A+B + RM fills done) — CPU render pipeline: **E1–E3 fold-in + 7-point pf+btk figs/tables (GO issued ~07:1x)** | 10:15 RLHF checkpoint render |
| | | 2 | **RLHF paper-faithful pf_pilot RUNNING** (CPU-heavy phases read as 0% GPU — /proc is the liveness source) | G1 → grid (pf_lo/mid/hi lanes; relief if >14:00 projection) |
| **pod A** (2×H100) | runpod-a (GPU 0), runpod-b (GPU 1) | 0 | scheduled-idle: struqpos-verdict standby + warm zero-bootstrap fallback for the L40S screen | struqpos verdict scoring |
| | | 1 | rmx_b eq-extension cells 5–6 (T8 set CLOSED — relu-mix ≡ btk exact 3/3; T10 relay checks) | drain ~11:30 |
| **pod B** (2×H100) | runpod-c (session DOWN since ~05:1x; hub-operated) | 0+1 | **shards C+D COMPLETED before the session died** (last 3 cells hub-repatriated via HF, LOG ~07:1x); NOW: hub-run l13-IT substrate rebuild (stage B) + ckpt push for the repatriated cells | **pre-designated RLHF relief venue at G1** |
| **mac-d-struqscreen-0728** (L40S $0.99/h) | mac-d | — | struqpos screen chain v2 RUNNING (bootstrap death ~$1 disclosed, death-proofed monitor) | verdict → runpod-a scores; terminate at drain |
| ~~mac-c-screen-0728~~ | — | — | TERMINATED (evalage WEAK, lane closed) | — |
| ~~mac-d-retrain-0728 (pod D)~~ | — | — | **TERMINATED 07:01 API-verified** — sycgen lane closed (exhibit FINAL-at-15/18, tsae trio abandoned-disclosed); ledger ~$38 actuals | — |

**Priority order (Han, 02:38): paper-faithful sweeps outrank ALL
btk GPU work; hunted tasks need either arm only; relu-mix is
certificate evidence, never a matrix column.**

**CPU-side work in flight:** runpod-1 = E1–E3 formal fold-in +
the 7-point pf+btk probing figs/tables (render GO issued);
runpod-2 = G1 scoring at pilot landing + the 10:15 RLHF
checkpoint render (deliverable of record; supersede branch if x10
resumes post-grid); runpod-a = struqpos verdict owner; mac-c
(local mac) = retryesc_gen generation (Claude API, $300 cap) —
roll-call response pending; mac-d (local mac) = struqpos screen
executor on the L40S; mac-local = hub (review/ratify + takeover
executor where owners are down; RLHF eval-substrate semantics:
`eval_cfg.hh_rlhf_cache` registry tag hashes into eval_key with a
`cache_expect` hard-check — see §3 and the 05:05 G2 LOG entry).

**Where outputs land:** canonical rows → `results/leaderboard.jsonl`;
hill-climb scratch → `experiments/explorations/tscale/RESULTS.md`;
hunt corpora → HF `temp-bench-data/hunt_corpora/`; checkpoints →
HF `temp-bench-data/ckpts/<train_key>/`; figures/tables →
`figs_writeup/`; verdicts/licences → the LOG.

## 5a. SSH access to the pods

```
# old pod (3×H100 — runpod-1, runpod-2):
ssh j42plcul70a2es-64410eb7@ssh.runpod.io -i ~/.ssh/id_ed25519
# pod A (2×H100 — runpod-a, runpod-b):
ssh 0lmrs9lk8apyhm-644121b8@ssh.runpod.io -i ~/.ssh/id_ed25519
# pod B (2×H100 — runpod-c):
ssh l2bp61kg82epel-64411fb1@ssh.runpod.io -i ~/.ssh/id_ed25519
# pod D — TERMINATED 07:01 07-28 (sycgen lane closed); coordinates
# retired. Current mac-d pod: mac-d-struqscreen-0728 (L40S, screen
# lane only — see agents/mac-d/STATUS.md for its coordinates).
```

- **Repo checkouts are PER-AGENT (ssh-verified 02:5x 07-28) — there
  is no `/workspace/temp_xc` on any of Han's pods** (pod D, the one
  API-provisioned exception, DOES use `/workspace/temp_xc` — single
  tenant, mac-d only)**:** old pod →
  `/workspace/agents/runpod-1/temp_xc` + `/workspace/agents/runpod-2/
  temp_xc`; pod A → `/workspace/agents/runpod-a/temp_xc` +
  `/workspace/agents/runpod-b/temp_xc`; pod B →
  `/workspace/agents/runpod-c/temp_xc`. The leaderboard/results in
  each checkout are the same append-only stream (agents push/pull
  through origin); the branch-of-record is `arxiv` on origin — read
  results there first, ssh only when you need live logs.
- **Scripted (non-interactive) use:** the RunPod ssh proxy forces a
  PTY — pipe your command list over stdin
  (`printf 'cmd; exit\n' | ssh -tt <host> -i ~/.ssh/id_ed25519`) and
  strip bracketed-paste noise from the output (`| grep -av 2004`).
- **LOOK, DON'T TOUCH (house rule: never modify a pod you did not
  spin up):** every GPU is running deadline lanes — `nvidia-smi`,
  `tail -f /workspace/logs/*.log`, and reading result files are
  fine; do NOT kill/launch/modify anything. Coordinate through the
  LOG instead.
- Pod-local HF/GH tokens live at `/workspace/.tokens/` (never in
  git); training logs at `/workspace/logs/`.

## 5b. Standing caveats an agent must not trip over

- **A12:** the shipped c3 "T10/T20" cells are silent-T5 replicas —
  never quote the shipped c3 T-ordering; the real T-sweep is this
  exhibit (first REAL T10 cells landed tonight).
- The mechanism story is per-task: probing = rare-boundary-contact
  regime (divergence from T2); RLHF = never-contact (identity
  through T16). One mechanism, two measured regimes — never quote
  cross-task.
- Aggregate identity/divergence claims only via keyed twin diffs
  (train_key-level), never band summaries; exclude the alias list.
- `eval_cfg.positive_control=true` rows are instrument-gate cells,
  not sweep cells — filter them out of aggregations.

## 6. Hunted-task exhibits (deliverable items 4–7) — where to look

Per-item exhibit blocks live in `REBUTTAL_HANDOFF.md` §4–§7; code:

- **item 4, λ̂ backtracking-intensity (COMPLETE):**
  `experiments/explorations/task_hunt/sc_lambda/` (CARD.md,
  results/); fig `figs_writeup/fig_lambda_shuffle_tsweep.*`.
- **item 5, dq question-marks (COMPLETE, TOY-class disclosed):**
  `experiments/explorations/task_hunt/diafaces/` (DQ_T_FILL_CARD.md,
  results/); exhibit set per HANDOFF §5.
- **item 6, sycgen sycophancy-adjacent (IN FLIGHT — first hunt KEEP
  3/3, LOG 02:28):** `experiments/explorations/task_hunt/sycgen/`
  (GENERATION_CARD.md, SCREEN_CARD.md, RETRAIN_CARD.md,
  screen.py/screen_grids.py, run_retrain.py, shuffle_overlay.py,
  results/); datasource plugin `sycgen_real_age_llama31_8b_l14`;
  retrain rows land on the canonical leaderboard ~05:30–06:30
  (regenerate the census).
- **item 7 — OPEN. evalage resolved WEAK** (0 KEEP / 0 KILL, gains
  below the +0.05 bar, no order signal; LOG 03:14):
  `experiments/explorations/task_hunt/evalage/` (RESULT.md is the
  full verdict); corpus on HF `temp-bench-data/hunt_corpora/` with
  sha receipts. Pathway now: `task_hunt/retryesc/` (retryesc_gen
  regeneration) + StruQ premeasures (runpod-a).
- The hunted-task shuffle-overlay results are eval_extra-namespaced
  JSONs in each dir's `results/` (NOT leaderboard rows) — see the
  census §"Hunted-task cells outside the leaderboard".
