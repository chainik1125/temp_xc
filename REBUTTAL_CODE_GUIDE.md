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
before quoting coverage. Known in-flight gaps at 03:06: RLHF
btk-only T{6,10} (old-pod GPU 2), ALL `paper_txc_base_v1t` cells
(sprint, ETA ~06:30–07:30), sycgen retrain rows (~04:00). The
probing btk-only arm is COMPLETE at 7 T × 3 seeds (T10/s2 landed
03:00).

## 2. Sparse probing — code, data, results

- **Experiment dir:** `experiments/probing/actmix/` — start with
  `CARD.md` (the frozen sweep card), `CARD_PAPER_FAITHFUL.md` (the
  paper-faithful sprint card, PIN d9235755b — plugin contract tests,
  21-cell grid, 5-GPU shard split), `CARD_RELUMIX.md` (the relu-mix
  arm grid), `sweep.py` (cell launcher), `analysis.py` (aggregation +
  `make_writeup_fig` renderer), `prep_cache.py` (substrate).
- **Substrate:** gemma-2-2b-it L13 resid_post, the paper's probe
  cache — datasource `gemma_2_2b_it_l13_fineweb_24k128`; 38 cached
  SAEBench tasks, headline aggregation = SAEBench-36 (CT pair
  winogrande/wsc excluded per camera-ready convention).
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

## 5. FLEET MAP — what is running on every pod (snapshot 02:58 BST 07-28; sprint shards RUNNING)

This section dates fast. **Live sources: `agents/<id>/STATUS.md`
(each agent self-maintains its own) + the LOG tail** — trust those
over this snapshot if they disagree.

| pod | agents | GPU | running NOW | next |
|---|---|---|---|---|
| **old pod** (3×H100) | runpod-1 (GPU 0/1), runpod-2 (GPU 2) | 0 | **paper-faithful probing shard A RUNNING since 01:39 UTC** (T16×3 → T1/s42; card d9235755b, arch `paper_txc_base_v1t`) | drain → 11:00 renders |
| | | 1 | night-grid tail (btk s2/T10 — the last btk probing cell) | **shard B (T10×3 → T1/s1) armed at drain** |
| | | 2 | x6 ‖ x10 (btk RLHF T{6,10}; **YIELDS to the RLHF paper-faithful grid on contention — Han priority**) | RLHF paper-faithful grid (agentic port card ~04:30) |
| **pod A** (2×H100) | runpod-a (GPU 0), runpod-b (GPU 1) | 0 | **paper-faithful probing shard E RUNNING since 02:41** (T4×3 → T2×{1,2}; est done ~06:20) | — |
| | | 1 | rmx_b (eq-extension cells 2–6; cell 2 lands ~04:00) | overflow only post-drain (~11:30; boundary offer CLOSED) |
| **pod B** (2×H100) | runpod-c alone | 0+1 | **FROZEN hill-climb draining in-flight C4/C5-T16 (~02:50 / ~03:15)**; resume playbook in `tscale/RESULTS.md` §FREEZE; ckpt mirror to HF in background | **shards C (T8×3→T1/s2) + D (T6×3→T2/s42) at drain** — substrate on-pod, zero sync |
| **mac-c-screen-0728** (L40S) | mac-c | — | **TERMINATED 03:14 (API-verified)** — evalage screen ran and returned WEAK 3/3 (no retrain); lane closed on-budget | mac-c continues CPU-side: retryesc_gen design/generation |
| **mac-d-retrain-0728** (2×H100) | mac-d | 0+1 | **sycgen matrix retrain RUNNING — 36 cells, T{1,2,4,8,16} ≡ the λ̂ exhibit axis** (card 74d260321 + §5 amendment: T{6,10} can't tile eval L=32, receipts kept); shard0 DONE, shard1 ETA ~03:35–03:55 | shuffle overlay → repatriate rows → HF ckpts → T-sweep figure (~04:30) → TERMINATE pod |

**Priority order (Han, 02:38): paper-faithful sweeps outrank ALL
btk GPU work; hunted tasks need either arm only; relu-mix is
certificate evidence, never a matrix column.**

**CPU-side work in flight:** runpod-1 = 11:00 render
pipeline (7-point per-k probing figs+tables + the onset-map
certificate with traces); runpod-2 = `agentic_txc_02` port
(vendor pattern) + RLHF 7-point render + rmx cross-pod sha
checks; runpod-a = StruQ premeasures ($0, our bars); mac-c (local
mac) = evalage corpus COMPLETE (2.04M tokens, both card gates pass,
HF-pushed) → re-tokenization transplant + screens + retryesc_gen
design; mac-d (local mac) = sycgen retrain owner (watchers armed) +
overlay/figure at drain; mac-local = hub (review/ratify only, no
compute).

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
# pod D (2×H100 — mac-d's API-provisioned retrain pod, jge1fuj9hqu8et;
# direct sshd, no proxy — sycgen retrain lane, terminates at lane-done):
ssh root@64.247.201.51 -p 16977
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
