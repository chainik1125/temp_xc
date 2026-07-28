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
  the ARCHIVED ckpts — T=5 × 3 seeds ONLY. **TABLE-LABELING RULE
  (binding): sweep columns = "TXC (v2, relu-mix/btk-only)"; paper
  composition = a separate "paper base (archived, T=5)" anchor row;
  never conflate.**
  The paper's RLHF TXC was **`agentic_txc_02` =
  `MatryoshkaTXCDRContrastiveMultiscale`** (matryoshka+contrastive,
  multiscale shifts, per-window TopK→ReLU, k_win=500) — a DISTINCT
  class from txc_pro (no subseq curriculum/k-asymmetry). Our RLHF
  exhibit is the plain-TXC modernization at the paper's window budget
  (k_win=100·T = 500 at the paper's T=5), per-window selection
  granularity preserved via the POST composition. Full pins:
  `experiments/explorations/task_hunt/COMPOSITION_AUDIT.md` §0/§3/§6.
  (Provenance status: the agentic_txc_02 identification is
  AUDIT-PINNED with byte-identity receipts + corroborated by
  Dmitry's own HF seed-audit table; hub independent re-derivation
  queued — task #11 / LOG ~02:1x. If it ever fails, the disclosure
  is pulled immediately.)
  The RLHF renderer prints this disclosure below the axis on every
  fig.

## 2. Sparse probing — code, data, results

- **Experiment dir:** `experiments/probing/actmix/` — start with
  `CARD.md` (the frozen sweep card), `CARD_RELUMIX.md` (the relu-mix
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
  `eval_cfg.k_feat` ∈ {100→k20-style, 500}; k_win = 100·T.
- **Row filter:** `experiment=rlhf`, arch `txc_batchtopk_post_btkonly`
  (+ baselines), T ∈ {1,2,4,5,6,8,10,16} (T5 = the paper's operating
  point, kept as bonus), seeds {1,2,42}, same shuffle semantics.
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
  `push_ckpts_hf.py` (repo root) / `hf_durability_push.py` (RLHF).
- **Mirrored now:** all 26 trained RLHF ckpts; 30 probing
  certificate-evidence ckpts (twin pairs across the 7-T grid, sae
  pair, positive control; spot-check MATCH). Remainder mirrors in
  priority order; pod-local at `checkpoints/<train_key>/`, indexed
  by `checkpoints/manifest.jsonl` (tracked).
- **Paper-era ckpts:** `han1823123123/txcdr-base`
  (`<arch_id>__seed42.pt`, incl. `agentic_txc_02`) +
  `temp-bench-models` (c3 cells) — see COMPOSITION_AUDIT §3.

## 5. Standing caveats an agent must not trip over

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
