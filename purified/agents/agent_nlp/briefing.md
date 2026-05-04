<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_nlp; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_nlp
last_state_update: 2026-05-03T22:00:00Z
component: c3, c4
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent NLP**. You own C3 + C4 only. Files you may edit:
- `agents/agent_nlp/briefing.md` (your own — agent-owned sections only)
- `docs/components/c3.md` and `docs/components/c4.md`
- `experiments/c3_probing/`, `experiments/c4_qualitative/`
- Code under `src/temp_bench/` that you author + commit (eval modules
  for probing / qualitative; data loaders under `temp_bench.data.nlp`)
- `configs/datasources.yaml` — adding new C3/C4 datasources is fine.
  YAML edits to other components' datasources require a Han ping.

**Files that are OUT OF SCOPE — do NOT edit even if it seems harmless:**
- `agents/agent_*/` — every other agent's directory.
- `docs/paper/*` — agent_paper's territory.
- `agents/agent_paper/decisions.md` — global decisions log.
- `agents/README.md` — roster/contract; agent_paper-owned.
- `configs/locked_archs.yaml` — only agent_paper edits this.
- `pyproject.toml` and `uv.lock` — dependency changes affect every
  agent's venv; pyproject + lockfile must be committed atomically,
  and only agent_paper coordinates that. If you need a new dep,
  surface in Open questions.

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. This
is non-negotiable even if Han verbally approves — the audit trail of
who edited what depends on each agent staying in their lane.

### Han decisions 2026-05-04 PM (CRITICAL — TrainingConfig re-issued per Phase 5)

A "batch=256 → 2048" cross-agent directive was issued earlier today
(commit a9200560) and reverted (commit 0beae2bf). It treated only the
contrastive archs (T-SAE, TXC-pro) as needing higher batch — Han caught
that as unfair to non-contrastive baselines. **The new directive is
Phase-5-faithful, applied uniformly across all archs and all pods.**
Read `decisions.md` § 12 in full before running anything. Gist:

- **`TrainingConfig` defaults are now**: `batch_size=1024`, `n_steps=25_000`,
  `plateau_early_stop=False` (disabled — see below). Just default-construct
  `TrainingConfig()` in your runner — no per-component overrides for these
  knobs.
- **Uniform across pods**: H100 + A40 both run at batch=1024. The point
  is every arch trains under identical conditions, matching the SAE-
  comparison-paper standard (T-SAE §4.1: "All SAEs are trained with...
  chosen to allow for comparability"; TFA App. B.1: "all SAEs trained
  from scratch... to enable a consistent and fair evaluation").
- **Plateau-stop is OFF; 25K cap is binding for every cell.** The schema's
  plateau detection (`max(loss[-5000:]) - min(loss[-5000:]) < 1e-4`,
  `training/sae_trainer.py:158-165`) is an absolute threshold over a fixed
  window — cross-arch unfair (archs at different loss scales would trigger
  at different points). SAE-comparison literature (T-SAE §4.1, TFA App. B.1,
  GemmaScope) uses fixed step counts. Every cell trains to the 25K cap;
  fairness mechanism is "exactly 25.6M tokens per arch."
- **If you observe loss still descending steeply at step 25K** (e.g.,
  final-1K-step drop > 5% of loss value), surface that as a comment on
  your run — the cap may need to be bumped uniformly across all archs.
- **Cache hygiene**: `batch_size`, `n_steps`, and the plateau-* fields
  are all in the `train_key` hash (`src/temp_bench/config.py:181-193`).
  New cells get fresh `train_key` / `eval_key` automatically. Old
  batch=256 cells stay in `results/leaderboard.jsonl` for diff
  comparison — **when rendering AUTO-RESULTS, filter for new rows only**
  (e.g., `training_cfg.batch_size == 1024`).

**Your specific re-run**: 24 C3 cells (4 archs × 3 seeds × 2 k_feats)
need new train_keys. C4 cells share the C3 checkpoints, so once C3 is
re-trained, C4 evaluation re-runs cheaply (cache-hit on training).
Bump `EVAL_PROTOCOL_VERSION` if you want to invalidate eval cache too,
but it's not strictly needed since `eval_key` derives from `train_key`.

You are agent NLP, lead on the language-model components of the paper:
**C3 (sparse probing)** and **C4 (qualitative latents)**. Both are on
the same subject: `google/gemma-2-2b-it` layer 13 residual stream. C4
piggybacks on C3's activation cache.

Hardware: pod `2× H100`, pinned to **GPU 0**. Pod mode `persistent` —
`/workspace` survives stop/start, HF backup is optional but
recommended at session end. agent_em shares the pod on GPU 1.

Your **long-pole task** is the activation cache (~14 GB,
~3 H100-hours): 24K FineWeb sequences × 128 tokens, fp16, layer 13. As
soon as it's on HF (`han1823123123/temp-bench-data`), agent_steer can
unblock — they are gated on this. Push as soon as ready, don't wait
for downstream training.

C3 hypothesis (from `docs/components/c3.md`): TXC-pro matches the best
per-token SAE at k=5 and small seed-significant win at k=20. TXC-base
matches at every k.

C4 hypothesis: TXC-pro matches T-SAE on Top-256 cumulative SEMANTIC
Pareto. **One metric only** — drop pdvar and any paper-style probe
variants (decision pinned in `docs/components/c4.md`).

**Task suite is locked**: `SAEBench+CT` (n=38) — upstream SAEBench's
canonical 36 binary one-vs-rest tasks (8 datasets, classes per
SAEBench's `dataset_info.chosen_classes_per_dataset`) plus the two
cross-token coreference tasks (WinoGrande + SuperGLUE WSC). See
`decisions.md` § 11 and `docs/components/c3.md` "Task suite" for
the full table + reproduction notes.

When you port the wasteland's `probe_datasets.py` + `crosstoken_datasets.py`,
apply three SAEBench-faithfulness fixes (do not blindly copy the
wasteland 36):
- **github-code**: use SAEBench's `codeparrot/github-code` with the 5
  SAEBench languages (C, Python, HTML, Java, PHP), NOT wasteland's
  `code_search_net` (python/java/javascript/go). NOT gated despite
  what the HF web viewer suggests — that page is disabled because the
  loader is a Python script, not because access is restricted. Just
  needs `trust_remote_code=True` (set via `HF_DATASETS_TRUST_REMOTE_CODE=1`
  in your shell or `set_agent_env.sh`) and `datasets<4` (already
  pinned in `pyproject.toml`). Smoke-test the loader once with a
  tiny `streaming=True` pull BEFORE the 3-H100-hour cache build.
- **amazon_sentiment**: emit BOTH 1.0-vs-rest AND 5.0-vs-rest binaries
  (wasteland only has 5.0).
- **amazon_categories**: hardcode `["1","2","3","5","6"]` and use a
  deterministic non-streaming pull (wasteland streaming-top-5 missed
  cat6 and is non-deterministic across runs).

Locked decisions in scope: #1 (two TXCs — no hill-climbing), #4
(cross-branch reads via `git show`), #6 (HF repos), #7 (Bricken
resample is C6-only by default; **C3/C4 keep it OFF** — revisit only
if time permits at the end of the paper sprint), #11 (SAEBench+CT
task suite).

References:
- `agents/README.md` (your roster row + pod specs)
- `docs/components/c3.md` and `docs/components/c4.md`
- `docs/paper/architecture.md` (locked TXC spec)
- `decisions.md` (10 locked policy items)
- `PROTOCOL.md` § 11 (framework discipline), § 12 (GPU pinning),
  § 9 *Session wrap-up*

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Persistent pod → script prints a manual `hf upload` recipe for every
checkpoint not yet on HF. Don't let Han stop the pod until that loop
completes (probe-cache + act-cache are on HF; your trained
.safetensors and per-cell run-dirs are NOT until you push).

---

## ⚠️ CRITICAL — TrainingConfig undertrained vs Phase 7 reference (2026-05-04)

**Affects every agent training SAE-family archs.** Han + agent_nlp
discovered that the autonomous TrainingConfig I shipped (batch=256,
n_steps=10_000, lr=3e-4) is **dramatically undertrained vs the Phase 7
reference** baked into
`origin/han-phase7-unification:experiments/phase7_unification/_train_utils.py::TrainCfg`:

| param      | Phase 7 ref | agent_nlp v1.0.0 + v1.1.0 | factor |
|------------|-------------|----------------------------|--------|
| batch_size | 4096        | 256                        | 16× too small |
| max_steps  | 25_000      | 10_000                     | 2.5× too few  |
| lr         | 3e-4        | 3e-4                       | ✓ |
| effective batches × steps | 102M | 2.56M | **40× less effective gradient information** |

**Symptom**: my v1.1.0 cells had `plateau_early_stop=True` with
patience=5000; none of the 24 cells stopped early → loss was still
descending at step 10K. Almost certainly undertrained. (`my_train_fn`
discards `result["log"]` so I have no per-step loss curve as direct
evidence — fixing in the next round.)

**Effect on C3 v1.1.0 numbers**: relative ordering (TopK-SAE >
TXC variants > T-SAE) is consistent with Phase 7 BASE-side reference,
so the *qualitative* C3 honest negative still holds. Absolute gaps
may shrink under proper training — TXC variants (subseq + matryoshka
+ multi-distance contrastive) likely benefit MORE from large-batch
training than TopK-SAE does. **Don't ship the v1.1.0 numbers as the
paper headline until rerun.**

**Cross-agent impact map (snapshot from `checkpoints/manifest.jsonl`,
2026-05-04)**: ALL FOUR agents are using `batch_size=256` (no exception)
which is 16× smaller than Phase 7's 4096. Affected components:

| component | agent       | archs trained                      | cfg used                | gap to Phase 7 ref |
|-----------|-------------|------------------------------------|--------------------------|---------------------|
| C3        | agent_nlp   | topk_sae, tsae_paper, txc_base, txc_pro | n_steps=10K, batch=256 | 16× batch, 2.5× steps |
| C4        | agent_nlp   | (reuses C3 checkpoints)            | inherits C3 cfg          | same as C3 |
| C5        | agent_steer | topk_sae, tsae_paper, txc_base, txc_pro | n_steps=30K, batch=256 | 16× batch (steps OK) |
| C6        | agent_em    | sae_arditi, txc_base               | n_steps=30K, batch=256, plateau_off | 16× batch (steps OK) |
| C7        | agent_back  | topk_sae, txc_base, txc_pro        | n_steps=30K, batch=256 | 16× batch (steps OK) |

C1 + C2 (synthetic toy by agent_paper) aren't affected — they use small
toy archs at d_sae=40 with their own training defaults, not the
SAE-family Phase 7 reference.

**The batch_size axis is the universally-shared mistake** — but the
*optimal* batch is COMPONENT-DEPENDENT, not a one-size-fits-all 4096.
Phase 7's 4096 was specifically tuned for H200 141 GB VRAM with
Gemma d_in=2304; on H100 80 GB or for components with bigger d_in
(C6 Qwen-14B d_in=5120, C7 Llama-8B d_in=4096) and bigger d_sae
(c6+c7 override to d_sae=32768) the memory ceiling drops. Plus:

- **Contrastive / matryoshka archs (txc_pro, T-SAE temporal)**:
  benefit MOST from large batch — InfoNCE quality scales with
  in-batch negatives count.
- **Vanilla TopK / per-token archs (topk_sae)**: less batch-sensitive;
  small batch can be fine if step count is high.
- **Memory-bound configs (C7 Llama-8B + d_sae=32768)**: physical
  ceiling on batch_size before OOM. May need batch=1024 or 2048,
  not 4096.
- **Some case studies may PREFER small batch** for stochasticity in
  the loss landscape (e.g., dead-feature recovery via Bricken, where
  stochastic batch noise helps explore the feature space).

**Decision is the overseer's, not agent_nlp's.** Each agent should
re-evaluate their own batch_size in light of:
  (a) their arch's loss structure (contrastive ↔ large batch)
  (b) their datasource's d_in × d_sae memory footprint
  (c) their pod's VRAM (H100 80 GB vs H200 141 GB vs A40 48 GB)
  (d) their step budget (more steps × small batch can equal fewer
      steps × large batch in token-equivalent terms)

Likely effect on each agent's headline:
- C3 (agent_nlp): TXC ranking ordering vs TopK probably stable, absolute
  AUC gaps may shrink at fair budget (TXC variants benefit more).
- C4 (agent_nlp): tsae_paper SEMANTIC count may move; txc_pro Pareto
  position vs T-SAE could flip if TXC contrastive trains properly.
- C5 (agent_steer): steering success rate depends on feature quality;
  could shift either way.
- C6 (agent_em): EM gap-close numbers depend on TXC-base feature
  identification quality; could shift.
- C7 (agent_back): backtracking PR-AUC depends on the SAE separating
  backtracking-related features; could shift.

**None of the C3/C4/C5/C6/C7 paper headline numbers should be considered
final until each agent re-trains at a Phase-7-faithful batch size.**

**RESOLUTION (agent_paper, commit 06681098, 2026-05-04 PM)**:
new schema defaults locked across all agents:

```
TrainingConfig:
  batch_size = 1024            # Phase 5 summary.md:250
  n_steps = 25_000             # SAE-literature standard fixed schedule
  learning_rate = 3e-4
  warmup_steps = 1_000
  plateau_early_stop = False   # disabled — absolute-delta criterion is
                               # cross-arch unfair (different loss scales)
  precision = "bf16"
```

Rationale per `decisions.md` § 12: identical TrainingConfig across all
archs and pods (T-SAE §4.1, TFA App. B.1, GemmaScope all use fixed
schedules); plateau-stop disabled because the schema's `max - min over
window < 1e-4` is unfair across archs whose losses land at different
scales (T-SAE/MSE ≈ 1, BatchTopK ≈ 1e-2). agent_em aborted ~12
H100-hours of in-flight batch=256 calibration (Han accepted the cost).

**Action for agent_nlp**: simplify `experiments/c3_probing/run.py::_real_training_cfg`
and `experiments/c4_qualitative/run.py::_real_training_cfg` to
default-construct `TrainingConfig()` (new defaults match Phase 5
exactly). Re-run all 24 C3 cells; C4 inherits checkpoints (re-evals
only). v1.0.0 (batch=256, n_steps=10K) and v1.1.0 (batch=256, n_steps=10K
with padding fix) rows stay in `results/leaderboard.jsonl` for diff
comparison. **analysis.py must filter on `training_cfg.batch_size==1024`
when rendering the headline.**

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-04 13:10 UTC (post-compact, batch=1024 re-run in flight)**

- `git HEAD`: at 149ceccb (agent_nlp analysis migration to
  `canonical_train_keys`). Pulled agent_paper's `9a39137a`
  (helper) and integrated. Recent local commits:
  - `b43ccf5b` — analyses use `temp_bench.report.canonical_train_keys`
    (drops the hand-rolled manifest-join helper).
  - `fca1a78a` (now `149ceccb` post-rebase) — runners default-construct
    `TrainingConfig()` (batch=1024, n_steps=25K, plateau=False).
- Leaderboard: 24 v1.0.0 + 24 v1.1.0 C3 cells + 9 v1.0.0 C4 cells (all
  batch=256 — UNDERTRAINED, kept for diff comparison only). New
  batch=1024 cells will write fresh `train_keys` automatically.
- Checkpoints: 12 unique batch=256 train_keys on disk + manifest. Will
  be SUPERSEDED by 12 new batch=1024 train_keys (old kept; runner
  cache-keys diverge by `train_key` hash).
- **GPU sharing situation**: agent_em borrowed GPU 0 (my pinned GPU)
  for C6 batch=1024 calibration runs (PIDs 61972 + 61902, started
  12:48 UTC). Their cells take ~3.5 hr each, ~17-18 hr end-to-end on
  both GPUs. Confirmed they took GPU 0 because my pre-compact briefing
  said "Re-train pending overseer go-ahead" — reasonable per § 13.
  - **Co-running mitigation (agent_nlp 13:06 UTC)**: launched only
    txc_base + txc_pro + tsae_paper (3 archs × 3 seeds = 9 unique
    trainings) on GPU 0 alongside agent_em. These archs use
    window-level / anchor-pair `z` (~38 MB), so peak VRAM is modest
    (~8-10 GB) and fits in the ~30-40 GB headroom alongside agent_em's
    Qwen-14B work.
  - **topk_sae deferred**: per-token `z` at batch=1024 needs 4.83 GB
    raw + 9.66 GB on `(z != 0).float()` conversion (`base.py:81`).
    Caused OOM on first attempt. Will re-launch topk_sae cells once
    agent_em frees GPU 0 (or solo on GPU 0).
- Recent decisions in scope: #1, #4, #6, #7, #11, **#12 (TrainingConfig
  batch=1024, n_steps=25K, plateau=False — Phase-5-faithful)**.
- In flight: PID 64239 (txc_base + txc_pro + tsae_paper × 3 seeds × 2 k_feats
  = 18 cells, 9 unique trainings). Started 13:06 UTC. ETA TBD —
  contention slowdown unknown until first 1000-step progress line lands.
  Logs at `logs/c3_v3_lowmem.log`.

## C3 final headline (decided 2026-05-04, **EVAL_PROTOCOL_VERSION=1.1.0**)

**Result**: TopK-SAE leads at both sparsities. TXC variants underperform
by ~0.005-0.015 AUC (gap narrowed vs v1.0.0 after the padding fix).
**Honest negative for C3 hypothesis**.

| arch         | k=5            | k=20           |
|--------------|----------------|----------------|
| `topk_sae`   | 0.8447 ± 0.002 | 0.9016 ± 0.002 |
| `txc_base`   | 0.8397 ± 0.006 | 0.8887 ± 0.003 |
| `txc_pro`    | 0.8381 ± 0.008 | 0.8860 ± 0.002 |
| `tsae_paper` | 0.8281 ± 0.007 | 0.8851 ± 0.003 |

The leaderboard ALSO contains 24 v1.0.0 rows (the original buggy
right-padded run) for old-vs-new comparison. analysis.py filters to
v1.1.0 only for the headline. v1.0.0 numbers were ~0.005 higher for
topk_sae and ~0.005 lower for TXC variants — the gap closed slightly
under the fix, but TopK-SAE still leads.

**The Phase 7 padding fix did NOT rescue winogrande/wsc** (their AUCs
are 0.40-0.50 across all archs both before AND after). Per-token
mean-pool aggregation can't capture cross-token coreference; tasks
that need multi-token reasoning are intrinsically hard for this
aggregation regardless of padding handling. The 36 SAEBench tasks
shifted enough on net to move the headline by ~0.005.

Caveats for the paper text:
- IT-side activations (Phase 7 reference was BASE-side; small shift
  expected). Phase 7 noted "the IT side is entirely missing".
- TrainingConfig: n_steps=10K (chosen to fit 24 cells in 10h window).
  Schema default is 30K; Phase 7 reference used ~50K. Bumping would
  invalidate train_keys and add ~15-18 hours of cells.
- σ_tasks suggests a per-task breakdown could expose where TXC
  variants lose vs win. Per-task floats `auc__<task>` are persisted
  on every leaderboard row for this analysis.

**Padding fix** (landed 2026-05-04): probe cache rebuilt as
schema-2.0.0 left-aligned (N, S=32, d_in) with per-example
`first_real` metadata; `_encode_pool` masks padding contributions.
Mirrors Phase 7's
`docs/han/research_logs/phase7_unification/2026-04-27-URGENT-probing-cache-fix.md`.
Cache pushed to HF schema 2.0.0 (266 files including `first_real_*.npy`
per task).

## C4 final headline (decided 2026-05-04)

**Result**: T-SAE produces the most SEMANTIC features. TXC-pro mid-pack.
TXC-base dominated. **Honest negative for the C4 hypothesis** —
TXC-pro does NOT Pareto-dominate T-SAE.

| arch         | mean SEMANTIC ± σ_seeds | judge_agreement |
|--------------|-------------------------|-----------------|
| `tsae_paper` | 74.7 ± 8.1              | 0.905           |
| `txc_pro`    | 60.0 ± 2.6              | 0.852           |
| `txc_base`   | 42.0 ± 2.0              | 0.768           |

Pareto frontier (probing AUC k=20 vs SEMANTIC count): {tsae_paper, txc_pro}.
- tsae_paper: probing 0.8844, SEMANTIC 74.7 (high-SEMANTIC end)
- txc_pro: probing 0.8859, SEMANTIC 60.0 (high-AUC end)
- txc_base: probing 0.8841, SEMANTIC 42.0 (dominated)

The honest framing for the paper: **T-SAE wins on qualitative
interpretability while TXC-pro matches T-SAE on probing utility**;
TXC-pro doesn't ALSO win on qualitative as we hypothesised.

C4 caveats:
- Wasteland reference (1 seed, concat_random only): tsae_paper=95/256,
  Track 2 T=20 (a different TXC variant, not txc_pro): 102/256. Our
  numbers are systematically lower — different seed-distribution,
  combined concat_A+B+random instead of just random, n_steps=10K vs
  Phase 7's longer training.
- 768 Haiku calls per cell × 9 cells = ~7K Haiku calls. Total cost
  ~$0.20. Judge agreement averaged 0.84 across all cells.
- Judge κ validation deferred to post-deadline per c4.md (decisions §
  11). All judge outputs persisted to
  `results/runs/<eval_key>/judge_outputs.jsonl` for `pandas + scipy.stats.cohen_kappa_score`
  computation when 20-feature blind hand-score lands.

## What I just did (agent owns — overwrite)

C3 + C4 plumbing fully shipped. Cells in flight on H100. Status:

**C3 — sparse probing**:

- ✅ `temp_bench.data.nlp.probe_tasks` — 38-task SAEBench+CT loader
  with all 3 SAEBench-faithfulness fixes (github-code via codeparrot
  with post-iter language filter; amazon_sentiment 1+5 binaries;
  amazon_categories non-streaming + shuffle for cat6). All 8 dataset
  loaders smoke-tested individually. Hardcoded SAEBench class lists.
- ✅ `temp_bench.data.nlp.probe_cache` — `build_probe_cache()` +
  `load_probe_cache()` + `list_probe_cache()`. Per-task structure
  is `(N, seq_len, d_in)` fp16 numpy arrays. Idempotent.
- ✅ Full probe cache built (38 tasks, 79 GB at
  `results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128/`).
- ✅ `experiments/c3_probing/run.py::my_eval_fn` — wired to real
  probe-cache. Returns flattened per-task floats (`auc__<task>` × 38)
  PLUS aggregates (`mean_auc`, `std_auc`, `mean_acc`, `std_acc`,
  `n_tasks`). Primary metric: `mean_auc`.
- ✅ Progress wrapper around `batch_iter` prints every 1000 steps
  (the canonical trainer is silent by design — wrapper added in
  `my_train_fn` for autonomous-overnight visibility).
- 🟡 IN FLIGHT: 18 cells (topk_sae × tsae_paper × txc_base × seeds
  {1,2,42} × k_feats {5,20}). Process PID 21612, started
  2026-05-03T22:49Z. n_steps=10K @ 5 steps/sec → ~30 min training
  per unique (arch,seed). 9 unique trainings + 18 probings. Total
  ETA ~6 hours.

**C4 — qualitative latents**:

- ✅ `temp_bench.eval.qualitative` — full implementation (was
  NotImplementedError stub):
  - `load_concat_corpus(name)` reads from `data/concat_corpora/`
  - `encode_concat_corpus(sae_model, subject_model, layer, token_ids)`
    forwards Gemma → hooks layer → SAE encode → (n_tokens, d_sae) z;
    dispatches per-token vs window on `model.T`
  - `pick_top_features_by_var(z, n)` variance ranking
  - `gather_top_contexts(token_ids, tok, z_col)` — top-N max-activating
    text windows
  - `_call_anthropic` — exponential backoff on rate limits
  - `call_judges` — 2 Haiku judges + agree/verdict
  - `persist_judge_record` — appends to
    `results/runs/<eval_key>/judge_outputs.jsonl` (κ-deferred lifeline)
  - `top_256_semantic` orchestrator returns float-only metrics
- ✅ `experiments/c4_qualitative/run.py` + `analysis.py`. Runner
  shares train_fn with C3 so checkpoints reuse via `runner.run_cell`'s
  auto-skip. Analysis joins to C3 leaderboard via (arch, seed,
  k_feat=20) for Pareto x-axis, draws upper-right frontier.
- ✅ `data/concat_corpora/{concat_A, concat_B, concat_random}.json`
  — pre-tokenized JSONs ported from wasteland.

Earlier session work (still relevant — no regressions):

- ✅ Activation cache `gemma_2_2b_it_l13_fineweb_24k128` — pushed to
  HF (`han1823123123/temp-bench-data/act_cache/e4916bcae1881963/`).
- ✅ `temp_bench.architectures.{tsae, txc_base}` — both ported and
  smoke-tested in --smoke mode.

## Decisions made + carried forward (overseer can override)

- **Per-task AUC reporting**: my_eval_fn returns BOTH per-task floats
  (`auc__<task>` × 38) AND aggregates (`mean_auc`, `std_auc`, ...) on
  every leaderboard row. analysis.py uses aggregates for headline +
  per-task floats for σ_tasks.
- **Smoke leaderboard rows kept** (`eval_cfg.smoke=true` filter).
  analysis.py filters them out.
- **Bricken A/B for C3**: SKIPPED per decision #7 default.
- **MLC port**: SKIPPED. Lower priority per agent_paper "Non-decisions";
  appendix-only OK. Test entry stays in `KNOWN_UNPORTED`.
- **EVAL_PROTOCOL_VERSION = "1.1.0"** — bumped for the Phase 7
  padding fix. Stays at 1.1.0 for the batch=1024 re-train (the eval
  pathway didn't change again; train_key change alone forces re-eval).
- **C4 unaffected by padding fix** (no probe cache; forwards Gemma over
  concat_corpora token_ids directly).
- **C4 cells share train_keys with C3**. After re-train, C4 hits
  CACHED on training, runs fresh evals only. Re-launch C4 ONLY after
  C3 re-train lands.

- **Trainer step rate (5/sec) is mmap-bound**: act cache lives on
  MooseFS (`mfs#us-ca-2.runpod.net`); random-index reads are slow
  vs sequential. After ~1000 steps the kernel page cache warms up
  and rate may improve, but in this session it stabilised at 5
  steps/sec. If a future run wants to be faster, options are:
  (a) copy the act cache to local SSD (none on this pod), or
  (b) use larger batch_size to amortise per-step I/O — risky for
  schema validation since batch_size is part of train_key.

## Next action — POST-COMPACT, BATCH-FIX RE-RUN (agent owns)

Everything plumbing-wise is shipped. What remains is to re-train under
the new (batch=1024, n_steps=25K, plateau=False) defaults locked in
`schemas.py` + `decisions.md` § 12 (commit 06681098).

### Step-by-step

1. **Standard session prep**:
   ```
   cd /workspace/temp_xc/purified
   git pull --rebase origin final     # use -c user.email / user.name (see "Don't repeat")
   bash scripts/agent_smoke_test.sh   # expect 122 pass; KNOWN_UNPORTED = {stacked_sae, tfa, tfa_pos, mlc}
   ls results/probe_cache/gemma_2_2b_it_l13_fineweb_24k128 | wc -l   # → 38 (schema 2.0.0)
   ```

2. **Update C3 + C4 runners to default-construct TrainingConfig**:
   - `experiments/c3_probing/run.py::_real_training_cfg()` — currently
     explicitly sets `n_steps=10_000, batch_size=256, lr=3e-4, warmup_steps=500, precision="bf16"`.
     **Change to**: `return TrainingConfig()` (one-liner).
     Optionally also drop the now-unused per-arch arch-time progress
     wrapper — schema's plateau is disabled but my `progress_iter` in
     `my_train_fn` still works fine; leave it for visibility.
   - `experiments/c4_qualitative/run.py::_real_training_cfg()` — same
     change. C4 inherits via SAME train_key → cache-hit on training,
     re-eval only.

3. **Update `experiments/c3_probing/analysis.py`**:
   ```python
   # Currently filters on r.eval_protocol_version == "1.1.0".
   # Add: r.training_cfg.batch_size == 1024  (the new headline batch).
   ```
   Per agent_paper's directive: when rendering AUTO-RESULTS, FILTER
   for `training_cfg.batch_size==1024` to exclude old undertrained
   v1.0.0 + v1.1.0 rows. Both stay in leaderboard for diff comparison.
   Eval rows for old train_keys can persist; just filter.

4. **Run C3** (24 cells, batch=1024 invalidates old train_keys):
   ```
   bash experiments/c3_probing/run.sh
   ```
   Or in parallel (faster — confirmed-working pattern from earlier):
   ```
   nohup .venv/bin/python -u -m experiments.c3_probing.run \
     --archs topk_sae tsae_paper --seeds 1 2 42 --k_feats 5 20 \
     > logs/c3_v3_groupA.log 2>&1 &
   nohup .venv/bin/python -u -m experiments.c3_probing.run \
     --archs txc_base txc_pro --seeds 1 2 42 --k_feats 5 20 \
     > logs/c3_v3_groupB.log 2>&1 &
   ```
   ETA per `decisions.md` § 12: ~32 H100-hr → 16 wall-hours on 2× H100.
   NOTE: agent_em is using both H100s for C6 calibration when I last
   checked — wait until they finish or coordinate via gpu_locks.
   Each unique training: batch=1024 × 25K steps. At ~1.5 sec/step on
   H100 (estimate based on previous batch=256 ≈ 0.2 sec/step × 4× larger
   batch but parallelism amortizes), ~10 hours per training is the upper
   bound; agent_paper's 16 wall-hours (parallelized) is realistic.

5. **Render C3** (after re-run):
   ```
   .venv/bin/python -m experiments.c3_probing.analysis
   ```
   Sanity-check: TopK-SAE k=20 should land in 0.85-0.91 range
   (Phase 7 BASE-side reference). If TXC variants close the gap, paper
   headline becomes "TXC-pro matches TopK-SAE at fair budget"; if gap
   remains, the honest-negative C3 result holds.

6. **Run C4** (training cache-hits on new C3 checkpoints; eval only):
   ```
   bash experiments/c4_qualitative/run.sh --archs tsae_paper txc_base txc_pro --seeds 1 2 42
   ```
   Pre-condition: `ANTHROPIC_API_KEY` env var or `/workspace/.tokens/anthropic_key`.
   ~30 min compute + ~10 min Haiku per cell × 9 cells = ~6 hours total.
   Cost ~$0.36.

7. **Render C4** + commit + push everything.

8. **HF push** of new checkpoints (persistent pod, optional):
   ```
   .venv/bin/python -c "
   from temp_bench.cache import iter_manifest_for_agent
   from huggingface_hub import HfApi
   token = open('/workspace/.tokens/hf_token').read().strip()
   api = HfApi(token=token)
   for row in iter_manifest_for_agent('agent_nlp'):
       if row.hf_url is None:
           api.upload_folder(folder_path=row.local_path.rsplit('/', 1)[0],
                              path_in_repo=row.train_key,
                              repo_id='han1823123123/temp-bench-models',
                              repo_type='model')
   "
   ```

### Reference: launching with parallelization (the pattern that worked)

- `python -u` essential (without it, nohup blocks `print()`s for ~5 min
  until first 4KB of stdout buffer flushes — see Don't repeat).
- 2 parallel processes on 2× H100 worked WITHOUT deadlock when the
  probe cache was already in OS page cache. First-eval mmap-warmup is
  the bottleneck; subsequent reads are RAM-fast.
- Earlier attempted parallelism BEFORE cache was warm: BOTH processes
  blocked in state D on MooseFS — had to kill one. Sequence: cache
  build first (single process), then parallel eval is safe.

### Reference: how to read the LATEST decision

`agents/agent_paper/decisions.md` § 12. The Identity+mandate section
of my own briefing also has the gist (Han's edit at line ~42-75).

### Reference: preserved leaderboard rows

After re-run, `results/leaderboard.jsonl` contains:
- 24 v1.0.0 cells (batch=256, n_steps=10K, OLD eval, OLD padding)
- 24 v1.1.0 cells (batch=256, n_steps=10K, OLD eval, NEW padding fix)
- 24 v1.1.0 cells (batch=1024, n_steps=25K, NEW eval, NEW padding) ← headline
- 9 v1.0.0 C4 cells (legacy, will be superseded)
- 9 NEW C4 cells (after re-run)
- a few smoke rows

Filter `(eval_protocol_version=="1.1.0") AND (training_cfg.batch_size==1024)`
for paper headlines.

## Don't repeat (agent owns — overwrite)

Locked-decision tripwires:

- **Two TXCs only** (decision #1) — don't introduce a galaxy steering
  variant or a non-locked TXC; raise it in `docs/components/c3.md`
  first if you genuinely need to.
- **Cross-territory edits** — see the OUT OF SCOPE list in mandate.
  Even if Han verbally approves in chat, surface the request in
  writing first. My last-but-one commit got partially rejected on
  exactly this (commit `2283aa15`).
- **Wasteland imports** — code is on `origin/han-phase7-unification`,
  not in `final`. Use `git show`. Never `from src.architectures...`.
- **Bypass `runner.run_cell`** — it's the only writer to the
  leaderboard. Schema validation is mandatory.
- **Hardcode hyperparameters** — anything paper-relevant goes in
  `configs/locked_archs.yaml` and `configs/datasources.yaml`. Edit the
  yaml, not the .py.

Hard-won technical gotchas from this session (verify before bypassing):

- **`datasets<4` pin is load-bearing** for `codeparrot/github-code`.
  v4+ removed `trust_remote_code` and the dataset uses a Python
  loading script. Pinned in `pyproject.toml`.
- **github-code `languages=[...]` does NOT filter the stream.** Must
  `if sample['language'] != target_lang: continue` after iter.
- **`tsae_paper.config.T == 1`, NOT 2.** Contrastive pair is a TRAINING
  construct sampled inside `train_step`. T=2 routes the probe to
  window-encoding (wrong for T-SAE).
- **`LeaderboardRow.metrics` is float-only** (Pydantic). Categorical
  diagnostics like `task_name` go outside `metrics`.
- **Background `nohup ... &`** — bash wrapper returns immediately;
  python keeps running. Verify via `ps -ef | grep python`.
- **Decoder grad-parallel removal** uses `register_post_accumulate_grad_hook`
  on `W_dec` (PyTorch 2.0+). See `tsae.py`/`txc_base.py`/`txc_pro.py::_project_dec_grad`.
- **`einops` is NOT a dep.** Use vanilla `torch.einsum`.
- **TQDM_DISABLE=1 must be exported per bash call.** `set_agent_env.sh`
  doesn't set it. Standard pattern at `export TQDM_DISABLE=1 && ...`.
- **`python -u` is essential for nohup'd long-running scripts** — without
  it, `print()` calls buffer in a 4KB block and don't appear in the
  log file for ~5 min, making the process look stuck. Cost me ~5 min
  of training when I killed/restarted thinking it was deadlocked.
- **`tokenizer.padding_side="right"` for cache build, NEVER left**.
  Left padding shifts position-IDs of real tokens; out-of-distribution
  for Gemma. Phase 7 fix uses right pad + per-example reslice
  (left-aligned in the destination 32-frame). See
  `temp_bench.data.nlp.probe_cache::_encode_texts`.
- **`.contiguous()` after `.T` on saved tensors** — safetensors rejects
  non-contiguous tensors. Bit me on `tsae.py::_normalize_decoder` after
  30 min of training; tests didn't catch because the failure is at
  save time, not init. Always wrap `W_dec.data = ....T.contiguous()`.
- **MooseFS mmap is slow on first random access** (~5 steps/sec on the
  14 GB activation cache), then RAM-cached after warmup. Two parallel
  processes BEFORE warmup deadlocked in state D. Sequence cache build
  → eval; THEN parallel runs are safe.
- **`topk_sae` per-token z explodes at batch=1024**. Shape is
  `(B, seq_len, d_sae) = (1024, 128, 18432)` bf16 = 4.83 GB raw.
  `(z != 0).float()` in `architectures/base.py:81::train_step` doubles
  to 9.66 GB allocation in fp32, on top of model + activations + grad.
  Total peak ~25-30 GB on H100. Co-running with agent_em (38 GB Qwen-14B
  process) → OOM. Mitigation: launch TXC archs (window-level z, ~38 MB)
  + tsae_paper (anchor-pair z, ~76 MB) first while sharing GPU 0;
  defer topk_sae to when GPU 0 is solo.
- **Git commit identity**: repo has no user.email/user.name set.
  Commits use inline `GIT_AUTHOR_*` env vars. Rebases use `git -c
  user.email=... -c user.name=... rebase ...` (env vars don't propagate
  cleanly through rebase's internal commits).
- **Leaderboard JSONL conflicts during rebase**: append-only, both
  sides add new rows. Resolution = strip conflict markers, keep both
  sets. Done it 5+ times this session via:
  ```python
  with open('results/leaderboard.jsonl') as fh: keep = [l for l in fh if not l.startswith(('<<<<<<<', '=======', '>>>>>>>'))]
  ```
- **`runner.run_cell` cache contract**: skips eval if `eval_in_leaderboard(eval_key) and metrics_exist(eval_key)`. Means changing the
  probe cache content WITHOUT bumping `EVAL_PROTOCOL_VERSION` or
  `force_eval=True` silently returns OLD metrics. Lesson learned twice.
  This is why the batch_size + n_steps live in `train_key` (auto-invalidate
  on change) — no manual version bump needed for re-train.
- **`first_real` mask in `_encode_pool`**: window archs (T>1) have edge
  case where n_real < T means NO valid window for that row. Code
  falls back to all-windows mean for those rows (probe noisy but no
  NaN). Affects winogrande/wsc on TXC archs; a few rows per task.

## Open questions for Han (agent owns — overwrite)

1. **Probe cache HF push at schema 2.0.0** — DONE 2026-05-04 morning.
   266 files (X_train + X_test + first_real_train + first_real_test +
   y_train + y_test + meta.json per task × 38 tasks). 22 GB on HF.
   agent_steer / ephemeral pods get the new cache via
   `hf download han1823123123/temp-bench-data --repo-type dataset
   --include 'probe_cache/gemma_2_2b_it_l13_fineweb_24k128/*'`.

2. **batch=1024 re-run wall-time confirmation** — agent_paper estimated
   ~16 wall-hours on 2× H100 (decisions.md § 12). My back-of-envelope
   based on prior batch=256 step rate (~5 steps/sec) and batch=1024
   estimate (~1.5 steps/sec) gives ~10 hours per training × 12 unique =
   120 sequential or 60 parallel. Han: if this takes longer than
   expected, n_steps=25K cap is binding (no plateau-stop). Worst case
   we ship with whatever cells finish before deadline.

3. **C5 / C6 / C7 are also re-running at batch=1024**. Coordinate
   GPU access — agent_em was using both H100s for C6 calibration last
   I checked. Probably their batch=1024 re-run starts after their
   in-flight runs complete. We may overlap on GPU 0; check before
   launching.

4. **Filter convention for analysis.py** — DONE. C3 + C4 analyses use
   `temp_bench.report.canonical_train_keys` (agent_paper helper from
   commit `9a39137a`) instead of hand-rolling the manifest join.
   Old rows (batch=256) stay for diff comparison; decisions § 12 says
   no EVAL_PROTOCOL_VERSION bump needed (train_key change auto-
   invalidates the eval cache via the runner contract).

5. **`base.py:81` memory hot-spot — opportunistic fix.** The shared
   `train_step` computes
   `l0 = (z_flat != 0).float().sum(dim=-1).mean()` which allocates a
   `(B*S, d_sae)` fp32 tensor (9.66 GB at batch=1024 / d_sae=18432 /
   per-token archs like topk_sae). Reordering to
   `(z_flat != 0).sum(dim=-1).float().mean()` defers the float
   conversion to the scalar reduction → drops 9.66 GB peak with no
   semantic change. Would unblock running topk_sae batch=1024
   alongside other 38-GB GPU-0 processes (currently triggers OOM —
   see Don't repeat). Out-of-scope for me (`src/temp_bench/architectures/`
   is shared code touched by every agent's training); flagging for
   agent_paper / Han to land if it seems worth it. Workaround in the
   meantime: defer topk_sae cells to when GPU 0 is solo.
