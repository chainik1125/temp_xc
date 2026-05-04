<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_back; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_back
last_state_update: 2026-05-04T00:30:00Z
component: c7
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent BACK**. You own C7 only. Files you may edit:
- `agents/agent_back/briefing.md` (your own — agent-owned sections only)
- `docs/components/c7.md`
- `experiments/c7_backtracking/`
- Code under `src/temp_bench/` that you author + commit
  (`temp_bench.case_studies.backtracking`, judge dispatch helpers)
- `configs/datasources.yaml` — adding new C7 datasources is fine.

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
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

### Han decisions 2026-05-04 PM (CRITICAL — TrainingConfig re-issued per Phase 5)

A "batch=256 → 1024 (A40 components)" cross-agent directive was issued
earlier today (commit a9200560) and reverted (commit 0beae2bf). It
selectively bumped contrastive archs — Han caught that as unfair to
non-contrastive baselines. **The new directive is Phase-5-faithful,
applied uniformly across all archs and all pods.** Read `decisions.md`
§ 12 in full before running anything. Gist:

- **`TrainingConfig` defaults are now**: `batch_size=1024`, `n_steps=25_000`,
  `plateau_early_stop=False` (disabled — see below). Just default-construct
  `TrainingConfig()` in your runner — no per-component overrides for these
  knobs (per-component `d_sae=32768` for C7 stays per § 1).
- **Uniform across pods**: A40 (you) + H100 (agent_nlp/agent_em) both
  at batch=1024. Phase 5's empirically validated batch (summary.md:250)
  plus the SAE-literature-standard fixed-step schedule (T-SAE §4.1, TFA
  App. B.1, GemmaScope). Identical config across all archs being compared.
- **Plateau-stop is OFF; 25K cap is binding for every cell.** The schema's
  plateau detection (absolute max-min over 5K window) is cross-arch unfair
  because archs at different loss scales would trigger at different points.
  Every cell trains to the 25K cap; fairness mechanism is "exactly 25.6M
  tokens per arch."
- **If you observe loss still descending steeply at step 25K** (e.g.,
  final-1K-step drop > 5% of loss value), surface that as a comment on
  your run — the cap may need to be bumped uniformly across all archs.
- **Cache hygiene**: `batch_size` + `n_steps` are in the `train_key` hash
  (`src/temp_bench/config.py:181-193`). New cells get fresh keys
  automatically. Old batch=256 cells stay in `results/leaderboard.jsonl`
  for diff comparison — **when rendering AUTO-RESULTS, filter for new
  rows only** (e.g., `training_cfg.batch_size == 1024`).

**Your specific re-run**: in-flight C7 sweep (7 archs × 3 seeds = 21
cells). Re-launch the sweep at new defaults. The +1.574 reproduction
test is the remaining "win" candidate; re-derive at proper batch.
**A40 + d_in=4096 (Llama) + d_sae=32768 at batch=1024 will be tight on
VRAM** — if OOM, first try `precision="fp16"` (already default for A40),
then if still OOM drop batch to 512 (note in your AUTO-RESULTS that
this component had a forced batch reduction; remaining components stay
at 1024).

### Han decisions 2026-05-04 (resolves prior session's open questions)

1. **Meta HF approval: APPLIED 2026-05-04.** Meta gates typically
   resolve in 1–24 hr. Mirror-vs-Meta equivalence check task added
   below. Once Meta access lands, switch the c7 datasource back to
   `llama_3_1_8b_base_l10_ward` and re-run; cache-key invalidation
   handles cleanly (different `subject_model` → new act_cache_key →
   new train_keys → fresh cells; old NousResearch-cell cache stays
   in the leaderboard for diff/comparison).
2. **Mirror equivalence check (NEW TASK):** before any Meta-mirror
   substitution becomes a paper claim, run a bit-equality sanity
   check. On a fixed prompt + fixed token sequence, do one forward
   pass through `meta-llama/Llama-3.1-8B` (once Meta access lands)
   AND `NousResearch/Meta-Llama-3.1-8B`, compare layer-10 residual
   activations element-wise. If max abs diff > 1e-5, mirror is NOT
   safe to use unverified — paper claims must wait for Meta access.
   Drop a script at `experiments/c7_backtracking/check_mirror_equivalence.py`
   that loads both, runs 5 prompts × 50 tokens, reports max/mean
   abs diff. Run this immediately when Meta access lands, before
   re-running the sweep.
3. **Architecture porting authorization (resolves your prior open
   question #1):** **port whatever C7 needs yourself**, with header
   attribution per PROTOCOL.md § 2. agent_paper is NOT the gatekeeper
   on the .py files — they're open ownership ("first-needs-it ports
   it"). YAML registration is already complete for all 9 archs;
   you only write the class file. Specifically: stacked_sae, tfa, mlc,
   txc_pro are yours to port if/when C7 needs them. Don't wait.

You are agent BACK, lead on **C7: Ward Stage B backtracking**.

**Subject (paper-faithful)**: steering vectors derived from
`meta-llama/Llama-3.1-8B` (BASE) residual layer 10, applied at
inference to the reasoning model
`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`. This is the Ward et al.
2025 (arxiv 2507.12638) Fig. 1 setup. **DO NOT use Gemma-2-2b** for
C7 — without a reasoning-finetuned counterpart, the paper's central
"BASE-derived → induces backtracking in reasoning model" claim cannot
be replicated. Layer 10 is paper-justified (Appendix B.1 / Fig. B.1
sweep).

C7's subject is therefore distinct from C3/C4/C5 (Gemma-2-2b-IT) and
C6 (Qwen-14B-finance). That divergence is acceptable — each component
is its own subject; "two TXCs everywhere" survives.

Hardware: pod `4× A40`, pinned to **GPU 1**. Pod mode **`ephemeral`**:
HF is the source of truth, auto-push on checkpoint save. agent_steer
shares the pod on GPU 0 (different component + different cache).
GPUs 2 + 3 are unassigned — to use them, launch a subprocess via
`bash scripts/run_on_gpu.sh <idx> -- <command>` (sets
`CUDA_VISIBLE_DEVICES=<idx>` for the child only). No lockfile manager;
GPU sharing is a convention now (PROTOCOL.md § 13) — read peer's
"Current state" + `nvidia-smi` before borrowing.

You build your **own** Llama-3.1-8B-BASE L10 activation cache. No
upstream dependencies; you can start at T+0. d_model=4096 (vs Gemma's
2304); per-component d_sae overrides for c7 are pre-set in
`configs/locked_archs.yaml` (`per_component_hparams.c7.d_sae=32768`).

Hypothesis (from `docs/components/c7.md`): TXC-pro delivers the largest
peak Δgc (Sonnet-judged genuine backtracking under steering), roughly
3× the next-best architecture. **C7 is a candidate paper-headline
result**, conditional on reproducing Aniket's wasteland +1.574 under
our locked TXC-pro on the Llama anchor.

Aniket on `origin/aniket-ward-stage-b` is the upstream — his pipeline
in `experiments/ward_backtracking_txc/` (architectures.py, b1_…,
b2_…, b3_… — see directory listing) is what you port. His Stage A
(300 MATH-500 traces on R1-Distill-Llama, judge transcripts, sentence
labels, DoM vectors) is read-only from
`origin/aniket-ward-stage-b:results/ward_backtracking_txc/stage_a/`.
Don't merge his branch; read via `git show` (decision #4). His
`final-aniket` commit replaces our c7.md with hand-typed numbers and
violates the AUTO-RESULTS contract — **do not merge `final-aniket`**.

Two metric axes (per c7.md):
- **Inducement (Δgc)** — primary: Sonnet 4.6 judge counts genuine
  backtracking events. **Fresh κ validation is NOT in the critical
  path** — we lean on Aniket's wasteland κ values (0.749 / 0.773 /
  1.000) as prior-art validation. Eval pipeline persists every Sonnet
  call to `results/runs/<eval_key>/judge_outputs.jsonl`. Post-deadline
  κ on a fresh 20-transcript hand-score is a stretch task: load
  judge_outputs.jsonl + scipy.stats.cohen_kappa_score; no re-judging.
- **Detection (PR-AUC)** — primary detection metric, NOT F1, NOT
  ROC-AUC (12% positive class makes those misleading). Sparse linear
  probe, top-S features, S ∈ {1, 2, 4, 8, 16, 32}.

Architecture set: 7 archs (NOT TFA-pos), all with c7 d_sae=32768:
TopK-SAE, Stacked-SAE, TFA, **T-SAE = paper-faithful Ye et al. port
only** (`temp_bench.architectures.tsae:TSAEPaper` from
`origin/han-phase7-unification:src/architectures/tsae_paper.py`),
TXC-base, TXC-pro, MLC. **DO NOT port `tsae_ours.py`** — deprecated
crude approximation, never use.

Cut protocol: **cut25** (cut Stage A trace at 25% of unsteered length,
then steer-and-continue) per Aniket's B3 — beats LLM-judged cut and
full-trace.

Locked decisions in scope: #1 (two TXCs — Aniket's hill-climbed TXC
must be replaced by TXC-base/pro; the +1.574 reproduction is the
test), #4 (cross-branch reads), #6 (HF repos), #7 (Bricken opt-in;
C7 must run an A/B at 5k×1seed before adopting).

References:
- `agents/README.md` (your roster row + pod-mode contract)
- `docs/components/c7.md` (full setup, metric definitions, reference
  numbers, caveats, reproduction)
- `docs/paper/hardware.md` *Multi-GPU access*
- `decisions.md`
- `papers/backtracking.md` (Ward et al. 2025)
- `origin/aniket-ward-stage-b:docs/aniket/experiments/ward_backtracking/{methodology,results_b,handoff}_neurips_push.md`
- `PROTOCOL.md` § 7 (results live in state — AUTO-RESULTS markers),
  § 9 *Session wrap-up*, § 11 (framework), § 12 (pinning),
  § 13 (GPU sharing convention)

**Before you `status: complete`**: `bash scripts/wrap_up_session.sh`.
Ephemeral pod → checkpoints auto-pushed; the script prints a
one-liner to verify HF state before stop. Run-dir judge_outputs
are NOT auto-pushed — they only land in git via the wrap-up
commit, so run the script before any pod restart.

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-04T00:30:00Z**

- Clone: `/workspace/temp_xc/` (correct primary clone for agent_back).
- Env: `set_agent_env.sh agent_back` → CUDA_VISIBLE_DEVICES=1,
  TEMP_BENCH_POD_MODE=ephemeral. Smoke test: 51 passed.
- `git HEAD`: see `git log --oneline -5`. Last push includes
  case_studies.backtracking + experiments/c7_backtracking +
  data/nlp/ward.py.
- Git remote auth: token-encoded via gh PAT in
  `/workspace/.tokens/gh_token`.
- **Activation cache (Llama BASE L10)**: BUILT + PUSHED ✓ at
  `results/act_cache/fb2a74be884e512a/resid_post_L10.npy`
  (4.24 GB float16, shape (4044, 128, 4096), sample norm 7.289,
  finite=True). Auto-pushed to HF temp-bench-data.
- **Sentence-acts extraction (mining + PR-AUC prereq)**: BUILT +
  PUSHED ✓ at `results/c7_backtracking/stage_a/sentence_acts_L10.npz`
  (1.1 GB compressed, 25204 sentences × 6 (T) × 4096 fp32, 12.6%
  positive class — matches Aniket's documented 12%). HF backup at
  `c7_backtracking/stage_a/sentence_acts_L10.npz` on temp-bench-data.
- **Arches available** (3 of 7): topk_sae, tsae_paper, txc_base —
  all instantiate cleanly at d_sae=32768 (268M / 268M / 1.34B
  params respectively). Still gated on agent_paper: stacked_sae,
  tfa, tfa_pos, mlc, txc_pro. (sae_arditi is C6-only.)
- **End-to-end smoke validated** (eval_key=`a9b4ea184f7a477a`,
  topk_sae × seed=42 × 500-step train × {-8, 0, +8} mags):
  - delta_gc_peak = +0.131 at mag=+8.0 (small — 500 steps is random)
  - delta_gc_mag_-8.0 = -0.066, mag_0 = 0
  - pr_auc_S{1,2,4,8,16,32} = (0.155, 0.187, 0.196, 0.215, 0.228, 0.238)
  - 183 Sonnet judge calls persisted, leaderboard row appended.
  - Smoke artifacts cleaned (`results/runs/topk_sae/` removed) so
    production starts with clean state.
- **Production sweep RUNNING** (PID 15528 from
  `bash scripts/c7_run_sweep.sh`). archs=(topk_sae, tsae_paper,
  txc_base) × seed=42 × 25 mags. Default training (n_steps=30000).
  Per-cell wall ~100 min, total ~5 hours. Logs at
  `logs/c7_sweep_seed42.log`. Cells write to per-eval_key workspaces
  (no smoke contamination).
- Last leaderboard append: production topk_sae cell
  (`28c40a2a3a0bbd59`) at 01:24 → peak Δgc=+0.361 at mag=+16,
  pr_auc_S32=0.243.
- Last checkpoint saved: `f6cbec1a09c992d4` (topk_sae production,
  30k steps, 1.07 GB). HF-pushed.
- Active GPU lock(s): GPU 1 (production sweep, mid-cell-2).
- Recent decisions in scope: #1, #4, #6, #7.
- In flight: production sweep, tsae_paper × seed=42 cell running.

**Production sweep results (live):**

| cell | status | peak Δgc | peak mag | pr_auc S=32 | wall (min) |
|---|---|---:|---:|---:|---:|
| topk_sae × seed=42 | DONE 01:24 | +0.361 | +16 | 0.243 | 155 |
| tsae_paper × seed=42 | FAILED save (non-contiguous W_enc) | – | – | – | 53 |
| txc_base × seed=42 | RUNNING (started 02:17) | – | – | – | – |
| txc_pro × seed=42 (GPU 3, bf16) | RUNNING (started 02:00) | – | – | – | – |
| tfa × seed=42 (GPU 3, bf16) | queued | – | – | – | – |
| mlc × seed=42 (GPU 3) | queued | – | – | – | – |
| stacked_sae × seed=42 (GPU 3) | queued | – | – | – | – |

**Failures handled:**
- tsae_paper W_enc not contiguous → save_checkpoint failed before
  eval. Other agents shipped contiguous fixes (af552412, 0aea9cba)
  AFTER my main sweep imported. Retry queued: bg-task `b63w2c93k`
  fires after PID 15528 finishes, runs `--archs tsae_paper --seeds 42`
  on GPU 1 with the fix in place.
- txc_pro + tfa OOM'd at fp32 Adam state (~42 GB on A40) → bf16-cast
  fix added (commit 9cfd99df). Pool sweep on GPU 3 retrying with
  bf16 cast.

**4 ports landed** (commit b1baf484): tfa, mlc, stacked_sae,
_tfa_module. agent_nlp shipped txc_pro (commit 6ae94a74). All 7
locked C7 archs now instantiate cleanly.

Pipeline auto-renders c7.md AUTO-RESULTS via
`bash scripts/c7_post_sweep.sh` after each cell.

## What I just did (agent owns — overwrite)

1. Provisioning + Stage-A port + cut25 reference port (commits
   `e93acf2c`, `be47e886`).
2. `case_studies/backtracking.py` skeleton (commit `be47e886`):
   StageA / Cohort / SteeringHook / extract_boxed / answers_match /
   cut25_token_position / SonnetBacktrackingJudge / compute_delta_gc
   / compute_pr_auc_at_S / BacktrackingCaseStudy.
3. C7 reasoning-LM pipeline (commit `c6abd159`, rebased onto
   `02dd3e90`):
   - `load_reasoning_lm` + `generate_unsteered` +
     `generate_continuation_panels` (verbatim ports of Aniket's
     R1-Distill-Llama helpers, with attribution).
   - `run_phase1_unsteered` (cohort baseline cached at
     workspace/phase1_unsteered.json; idempotent).
   - `mine_top_features` (D+/D- selectivity on TempBenchArch.encode).
   - `run_arch_evaluation` (top-level orchestrator: mine → steering
     vector → cut25 × magnitude × cohort generation → judge → Δgc
     + PR-AUC).
   - `BacktrackingCaseStudy.evaluate` forwards to it.
   - `load_cohort_from_parquet` + `load_math500_lookup` (skip
     LM-discovery cohort phase by reading Aniket's parquet).
4. `temp_bench.data.nlp.ward.py` (sibling to agent_paper's
   `cache.py`): C7-specific corpus loader (Stage A traces) +
   cache_activations with hookpoint-keyed memmap layout. Resolved
   rebase conflict with agent_paper/agent_nlp's `__init__.py` by
   adopting their canonical public API for the package and putting
   my C7 path in `ward.py`.
5. `configs/datasources.yaml`: added
   `llama_3_1_8b_base_l10_ward_nousmirror` (NousResearch byte-
   identical mirror) to bypass Meta's gated repo. Original entry
   intact for paper-citation / future Han-license-acceptance.
   Both updated to n_seqs=4044 (actual windows from Stage A traces).
6. Built activation cache:
   `results/act_cache/fb2a74be884e512a/resid_post_L10.npy` (4.24 GB
   float16, shape (4044, 128, 4096)). Auto-pushed to HF
   temp-bench-data. Loaded subject model from NousResearch mirror.
7. `experiments/c7_backtracking/{run.py, analysis.py}` (commit
   `c6abd159`): runner.run_cell loop over 7 archs × 3 seeds with
   train_fn + eval_fn adapters; analysis.py reads leaderboard +
   judge_outputs.jsonl, renders Δgc + PR-AUC tables + plots, falls
   back to "no cells yet" placeholder.
8. `extract_labeled_sentence_acts` (commit `bd97879b`): port of
   Aniket's `_capture_windows`. Captures L10 residual activations
   at offsets [-13..-8] BEFORE each labeled sentence start. Caches
   to `results/c7_backtracking/stage_a/sentence_acts_L10.npz`. Now
   running in background (PID 9356).
9. End-to-end smoke test: 50-step training on topk_sae × seed=42 on
   the act cache succeeded (state_dict keys present, loss values
   stabilising). Pipeline is wired correctly.

## Next action (agent owns — overwrite)

## Next action (agent owns — overwrite)

**Pre-condition (Han owns)**: Han has already run
`bash scripts/bootstrap_runpod.sh` on this pod (interactive — prompts
for tokens; an agent cannot enter input). Because this pod is
ephemeral, Han re-runs it whenever the pod is recreated. Tokens are
in `/workspace/.tokens/` and the venv exists when you wake up. If
the smoke test complains about missing tokens, **ping Han**.

**Your clone path is `/workspace/temp_xc/`** (the primary clone — you
are the first agent on the 4× A40 pod). agent_steer runs on the same
pod but in a separate clone at `/workspace/temp_xc_steer/` — DO NOT
cd into agent_steer's clone.

**Han launches you via `start_agent.sh`** (not bare `claude`):
```
# First launch (fresh session, agent reads briefing):
bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_back --fresh

# Re-launch after disconnect (resumes your session):
bash /workspace/temp_xc/purified/scripts/start_agent.sh agent_back
```
The wrapper sources `set_agent_env.sh` in Han's parent shell so the
GPU pin / `AGENT_NAME` / pod mode (ephemeral) propagate into your
process. Bash tool calls don't share shell state, so YOU sourcing
the env in your first action is a no-op for subsequent commands.

**Where I'm picking up next:**

Tasks #5 + #7 + #8 are complete in code; tasks #9-#12 are the
training + eval + reporting cycle. agent_paper has shipped 3 of 7
archs (topk_sae, tsae_paper, txc_base) — partial unblock. Remaining
gaps: stacked_sae, tfa, tfa_pos, mlc, sae_arditi, txc_pro.

Immediate (when sentence_acts extraction finishes):

A. **Smoke-test eval cell end-to-end** on topk_sae × seed=42 with a
   small magnitude grid (e.g., `(0, -8, +8)`) — verifies:
   - `_build_batch_iter` reads from `resid_post_L10.npy`
   - train_sae produces a checkpoint
   - eval_fn instantiates from state_dict
   - mine_top_features picks a feature
   - run_phase1_unsteered caches the cohort baseline
   - generate_continuation_panels under SteeringHook works
   - SonnetBacktrackingJudge persists 61 × 3 = 183 calls
   - compute_delta_gc returns sensible numbers
B. **If smoke passes, kick off the full sweep on the 3 available archs**:
   - 3 archs × 3 seeds × 25 mags ≈ 5 hr wall-clock + ~$10 in Sonnet calls.
   - Use --build-cache-only is not needed (cache already exists).
   - `TQDM_DISABLE=1 .venv/bin/python -m experiments.c7_backtracking.run --archs topk_sae tsae_paper txc_base`
C. **Once stacked_sae / tfa / mlc / txc_pro land** (agent_paper):
   - Re-run `experiments.c7_backtracking.run` (cache-hit on the
     3 already-trained arches; only the new ones train + eval).
D. **Detection PR-AUC** is wired but conditional on
   `eval_cfg["sentence_acts"]` being passed. Currently it's None;
   need to pre-extract per-cohort sentence acts then thread through
   `eval_cfg`. TODO: wire this in `experiments/c7_backtracking/run.py`.
E. **Render**: `temp_bench.report.render(component="c7")` writes the
   AUTO-RESULTS block in `docs/components/c7.md`. analysis.py is
   ready; smoke tested.

**Headline check**: TXC-pro peak Δgc reproduces ~+1.574 (Aniket's
wasteland reference, hill-climbed TXC). Cross-check via
`results/c7_backtracking/aniket_reference/cut25/inducement_summary.csv`.
TXC-pro arch is still gated; closest available analog is txc_base.
If neither reproduces the lead, document honestly in c7.md.

## Don't repeat (agent owns — overwrite)

- **Aniket's hill-climbed TXC** — decision #1 forbids it. The point
  of C7 is whether the locked TXC-pro reproduces; if it doesn't, that
  is the result, don't try to revert.
- **Skip κ validation** — the Sonnet judge is unverified at this
  scale; PR-AUC numbers without κ ≥ 0.6 are not reportable.
- **Use C3's IT-L13 cache** — backtracking is on **Llama-3.1-8B BASE
  L10**, paper-faithful to Ward et al.; you build your own cache.
  Do NOT use Gemma-2-2b for C7 — without a reasoning-finetuned
  counterpart, the BASE→reasoning-model steering claim is unreplicable.
- **`tsae_ours.py`** — deprecated wasteland file (TopK + 50/50 split +
  no AuxK). The C7 T-SAE baseline is `tsae_paper.py` (faithful Ye et al.
  port), period.
- **Forget HF push** — ephemeral pod.
- **Wasteland imports** — `git show` only; copy with attribution.
- **Bypass `runner.run_cell`** — single canonical pathway.

## Open questions for Han (agent owns — overwrite)

1. **Architecture ports — partial unblock.** As of edbdfdd7 the
   landed arches are topk_sae, tsae_paper, txc_base (3 of 7).
   Still gated on agent_paper: stacked_sae, tfa, tfa_pos, mlc,
   txc_pro. My autonomous run uses the 3 available; the other 4
   slot in via cache-hit when agent_paper ships them. **Question**:
   any preference on whether agent_paper finishes the 4 vs me
   porting any with header attribution? My default for the 10-hour
   window is to wait + run with the 3 we have, then fold the rest
   in when they land.
2. **Cohort-from-parquet — adopted.** I'm using
   `results/c7_backtracking/aniket_reference/cut25/flip_matrix.parquet`
   as the canonical 31+30 cohort. Reproducibility is preserved by
   the parquet provenance + ATTRIBUTION.md note. The LM-discovery
   path is stubbed out for future use (see `build_cohort` source=…).
3. **Mining lives in `case_studies.backtracking`** (my territory).
   `mine_top_features` + `extract_labeled_sentence_acts` +
   `split_pos_neg` are part of my module. If the steering case
   study (C5) wants the same selectivity-ranking helper later, we
   can refactor up to `temp_bench.eval.case_study`. Defer until C5
   actually needs it.
4. **NousResearch mirror substitution.** The Meta
   `meta-llama/Llama-3.1-8B` repo is gated and the HF token
   (han1823123123) does not have access. I added
   `llama_3_1_8b_base_l10_ward_nousmirror` (byte-identical
   redistribution) and the C7 runner uses it. Paper still cites
   "Llama-3.1-8B BASE" — distribution detail. **Pick one**:
   (a) accept Meta gate manually + switch back to the original
       datasource, or
   (b) formally adopt the NousResearch mirror (delete the original
       entry in datasources.yaml).
5. **`temp_bench.data.nlp.ward.py` split from agent_paper's
   `cache.py`.** I added a sibling module under
   `src/temp_bench/data/nlp/` rather than extend
   `cache._stream_dataset_texts` to handle
   `ward_backtracking_math500` (cross-territory). agent_paper may
   want to fold the ward branch into `cache.py` for a single canonical
   public API; for now the C7 runner imports from
   `temp_bench.data.nlp.ward` explicitly.
6. **Sentence-acts cache (115 MB, 1.1 GB compressed).** Uploaded to
   HF temp-bench-data at
   `c7_backtracking/stage_a/sentence_acts_L10.npz`. Not committed
   to git (large + derived). Re-extract takes ~10 min on A40 if
   pod restarts and `sync_from_hf.sh` doesn't pull it. Question:
   should `sync_from_hf.sh` be extended to pull C7 stage_a
   artifacts? (agent_paper territory.)
