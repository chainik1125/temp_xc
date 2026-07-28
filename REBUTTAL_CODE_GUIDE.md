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
  pair winogrande/wsc excluded). **PANEL MAPPING + ERRATUM (LOG
  13:1x): the paper's caption/prose SAY 36-task but its PLOTTED
  summary is 38-task (its own appendix §c3 says so; trapezoid
  receipts: archived T=5 ckpts = 0.9007 on 38 ∈ published
  [0.899,0.902] vs 0.9334 on 36). Camera-ready 36↔38 caption
  inconsistency — amendment fix recommended. Rebuttal figs =
  36-task; same ckpts k20: 0.9248 (36) vs 0.8975 (38). Never
  cross-quote panels.**
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

## 5. FLEET MAP — snapshot 14:19 BST 07-28 (**FULL FLEET RESET at ~13:35**)

This section dates fast. **Live sources: `agents/<id>/STATUS.md` + the
LOG tail** — trust those over this snapshot if they disagree.

**⚑ Everything in the previous snapshot is gone.** Han terminated every
pod at ~13:35 ("I've killed all the pods. No more runpod chaos"). The
old pod, pod A and pod B are **EXITED**, and with them the agents
`runpod-1`, `runpod-2`, `runpod-a`, `runpod-b`, `runpod-c`. Their work
is durable on origin + HF; their containers are not.

### Agents now

| agent | where | owns |
|---|---|---|
| **mac-local** | Han's MacBook | hub: review, ratify, LOG, ledger oversight, this guide + `REBUTTAL_HANDOFF.md` |
| **mac-d** | Han's MacBook (drives pods via API) | **the RLHF pf grid and all six pods** |
| **mac-c** | Han's MacBook | **the task hunt, end to end** (`briefings/hunt-mac-c-takeover.md`) — item 7 is the open deliverable |

**⚑ All agent identities are sessions on ONE machine** — `Hans-MacBook-Pro`,
M5 Pro / 18 cores / **48 GB unified** (mac-d, `96e34816a`). Agent count
is **not** machine count; plan concurrency accordingly.

### Pods now (API-verified 14:19)

All **1×H100 80 GB SECURE, $2.99/h each = $17.94/h**, owner mac-d,
terminate-at-lane-end:

    mac-d-rlhfpf-0728    aqil2dkyikg3ze
    mac-d-rlhfpf-0728-2  p478c8uyllvkzz
    mac-d-rlhfpf-0728-3  5sbd2s9mh0njzo
    mac-d-rlhfpf-0728-4  mi7cnfpnuikybi
    mac-d-rlhfpf-0728-5  tnp7vvew4t80wi
    mac-d-rlhfpf-0728-6  c48kuf2z2dipmv

**Running:** the RLHF **paper-faithful** grid — 18 cells,
T{1,2,4,6,8,10} × 3 seeds, one cell per GPU, no co-tenancy.
T16 excluded (83 GiB > 80 with the resident buffer, and upstream has
no `t16` arch). Deferred btk T{6,10} resume after the pf grid.

**Priority order (Han, unchanged): paper-faithful outranks ALL btk GPU
work; hunted tasks need either arm only; relu-mix is certificate
evidence, never a matrix column.**

**Where outputs land:** canonical rows → `results/leaderboard.jsonl`;
hunt corpora → HF `temp-bench-data/hunt_corpora/`; checkpoints → HF
`temp-bench-data/ckpts/<train_key>/`; figures/tables → `figs_writeup/`;
verdicts/licences → the LOG.

## 5a. Pod access

**The old ssh coordinates in this section are RETIRED — those pods no
longer exist.** Current pods are API-provisioned and mac-d-owned:

```bash
agents/mac-d/podctl.sh mine              # list mac-d-* pods
agents/mac-d/podctl.sh ssh <podId>       # print ssh coordinates
agents/mac-d/podctl.sh status <podId>
agents/mac-d/podctl.sh terminate <podId> # refuses non-mac-d pods; verifies after
```

The key is env-injected from the macOS keychain inside that script
only — **never echoed, written to a file, or passed as an argument.**

- **Repo checkout on the new pods:** single-tenant, so `/workspace/temp_xc`
  (unlike the retired shared pods, which were per-agent
  `/workspace/agents/<id>/temp_xc`). The branch-of-record is `arxiv` on
  origin — **read results there first, ssh only for live logs.**
- **Scripted (non-interactive) use:** the RunPod ssh proxy forces a PTY
  — pipe your command list over stdin
  (`printf 'cmd; exit\n' | ssh -tt <host> -i ~/.ssh/id_ed25519`) and
  strip bracketed-paste noise (`| grep -av 2004`).
- **LOOK, DON'T TOUCH** (house rule: never modify a pod you did not spin
  up): `nvidia-smi`, log tails and result reads are fine; do not
  kill/launch/modify. Coordinate through the LOG.
- **Liveness = `/proc` receipts, never GPU point-samples + log size.**
  Three separate CPU-bound phases were misread as dead processes on
  07-28; do not make it four.
- Pod-local HF/GH tokens live at `/workspace/.tokens/` (never in git).

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
