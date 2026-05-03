<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_back; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_back
last_state_update: 2026-05-03T23:30:00Z
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

If you find yourself wanting to edit any out-of-scope file, **STOP**.
Add a bullet to "Open questions for Han" in your own briefing,
surface it in chat, and let Han or agent_paper land the change. Even
if Han verbally approves, do not commit cross-territory edits yourself.

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
GPUs 2 + 3 are spare pool slots — claim via
`temp_bench.utils.gpu_locks.claim_gpu(idx)` if you need parallelism
(PROTOCOL.md § 13).

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
  § 11 (framework), § 12 (pinning), § 13 (multi-GPU)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-03T23:30:00Z**

- Clone: `/workspace/temp_xc/` (correct primary clone for agent_back).
- Env: `set_agent_env.sh agent_back` → CUDA_VISIBLE_DEVICES=1,
  TEMP_BENCH_POD_MODE=ephemeral. Smoke test: 51 passed.
- `git HEAD`: post-Stage-A-port + case_studies.backtracking commit
  (see `git log --oneline -5`).
- Git remote auth: token-encoded via gh PAT in
  `/workspace/.tokens/gh_token`.
- Last leaderboard append: none yet.
- Last checkpoint saved: none yet.
- Active GPU lock(s): none.
- Recent decisions in scope: #1, #4, #6, #7.
- In flight: human-paused mid-port. Resume at task #5/#7.

## What I just did (agent owns — overwrite)

1. Provisioning: `set_agent_env.sh agent_back` + `agent_smoke_test.sh`
   (51 passed; preflight notes 8 expected arch-import gaps — agent_paper
   territory; my arch-deps blocker noted in *Open questions* below).
2. `sync_from_hf.sh` ran but produced no real downloads (HF repos
   appear empty for now). Expected — nothing to pull yet.
3. Read `docs/components/c7.md` end-to-end + Aniket's
   `handoff_neurips_push.md`, `methodology_neurips_push.md`,
   `results_b_neurips_push.md` (via `git show`).
4. Inventoried `origin/aniket-ward-stage-b:experiments/ward_backtracking_txc/`
   (architectures.py, b1/b2/b3, grade_backtracking, etc.).
5. Confirmed datasource `llama_3_1_8b_base_l10_ward` and
   `r1_distill_llama_8b_l10_traces` already in `configs/datasources.yaml`.
6. Ported Stage A artifacts (6 files, 12 MB) from
   `origin/aniket-ward-stage-b @ a62175ee:results/ward_backtracking/` →
   `results/c7_backtracking/stage_a/` with `ATTRIBUTION.md`. **Note**:
   these are the dom-vector / sentence-label traces (10 reasoning
   categories), NOT a MATH-500 cohort. The truly-wrong/correct cohort
   is built at run time from MATH-500 (Aniket's b3 phase-1 step).
7. Ported Aniket's wasteland Stage B reference outputs (cut25 protocol,
   5 files including `flip_matrix.parquet` + `summary.json`) to
   `results/c7_backtracking/aniket_reference/cut25/` with
   `ATTRIBUTION.md`. Use only for cohort qid lookup + cross-checking
   `compute_delta_gc` against Aniket's headline +1.574.
8. Wrote `src/temp_bench/case_studies/backtracking.py` with:
   - `StageA`, `load_stage_a`, `Cohort`, `build_cohort` (cohort
     builder needs proper MATH-500 GT lookup — current fallback drops
     all 300 because Stage A traces have empty `answer` fields).
   - Verbatim ports (with attribution): `SteeringHook`, `extract_boxed`,
     `_strip_latex_to_plain`, `answers_match`.
   - `cut25_token_position` helper.
   - `SonnetBacktrackingJudge` async judge with mandatory
     `judge_outputs.jsonl` persistence (judge_id, prompt_hash, raw kept
     for audit; resumable via `existing_keys()`).
   - `compute_delta_gc` (baseline-corrected per qid at mag=0).
   - `compute_pr_auc_at_S` (sparse-probe PR-AUC, GroupKFold by qid).
   - `BacktrackingCaseStudy(CaseStudy)` skeleton (setup loads Stage A
     + cohort + judge; evaluate is a stub awaiting reasoning-model
     loader / arch ports).
9. Smoke test of the new module: imports clean; Stage A loads
   (300 prompts/traces/labels + dom_vectors); `extract_boxed`,
   `answers_match`, `cut25`, `parse_judge_reply` all pass.

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

**Resume context (2026-05-03 evening, paused mid-port for human break):**

Provisioning + Stage-A port + reference port + module skeleton are
DONE and pushed. The blocker that gates training (tasks #8, #9, #10)
is the missing arch implementations (see *Open questions*). Tasks
that don't depend on the archs:

A. **Finish `BacktrackingCaseStudy`** — wire `evaluate(arch, seed)`
   to a shared helper `run_arch_evaluation(arch, seed, …)` that:
   1. Loads R1-Distill-Llama-8B onto GPU 1 (one-time per session).
   2. Runs unsteered MATH-500 phase-1 on cohort qids → cache to
      `results/c7_backtracking/phase1_unsteered.json` (per-arch
      independent → reusable across arches; only run once per pod).
      Use Aniket's cohort qids from
      `results/c7_backtracking/aniket_reference/cut25/flip_matrix.parquet`
      to skip cohort discovery.
   3. For each magnitude in `DEFAULT_MAGNITUDE_GRID` × cohort qid:
      cut at `cut25_token_position`, attach `SteeringHook` to layer
      10, generate continuation, persist judge call.
   4. Returns metrics dict `{delta_gc_peak, delta_gc_peak_mag,
      stability, pr_auc@S, ...}` keyed for the `CaseStudyResult`.
B. **Update `build_cohort`** — current implementation drops all 300
   because Stage A traces have empty `answer` fields. Switch to
   reading qids from `aniket_reference/cut25/flip_matrix.parquet` as
   the primary path; LM-evaluation fallback as a secondary path
   that only runs if reference is missing.
C. **Build `experiments/c7_backtracking/{run.py, analysis.py}`**
   from `experiments/_runner_template.py` + `_analysis_template.py`.
   `run.py`: thin loop over (arch ∈ 7, seed ∈ {1,2,42}) calling
   `runner.run_cell(component="c7", …)`. `analysis.py`:
   reads leaderboard.jsonl + judge_outputs.jsonl, computes Δgc table
   + PR-AUC table + plots, then `report.render` rewrites c7.md
   AUTO-RESULTS block.
D. **Decide steering-vector source** — the SteeringHook needs a
   `vec` (decoder direction). Aniket mined per-(arch, feature_id) by
   sentence-mean-difference on the labeled D+/D- sets. We need to
   port the mining helper or build one fresh. The candidate location
   is `temp_bench.case_studies.backtracking.mine_top_features(model,
   stage_a, top_k=32)` — owns mining feature ranking and vector
   selection. Spec it in `c7.md` first since the choice affects
   reproducibility.
E. **Once arch ports land** (agent_paper):
   1. `bash scripts/agent_smoke_test.sh` — preflight gaps drop to 0.
   2. Build Llama-3.1-8B BASE L10 activation cache via
      `temp_bench.data.nlp.cache_activations(...)` (port from
      Aniket's `cache_activations.py`). 8000 seq × 128 tok →
      ~2 GB fp16 on disk. `cache.save_activations` auto-pushes to
      HF temp-bench-data on ephemeral pods.
   3. Train 7 archs × 3 seeds via `runner.run_cell(...)` (cache hits
      after the first seed make this fast).
   4. Run inducement Δgc + detection PR-AUC sweep.
   5. Run `temp_bench.report.render(component="c7")`.

**Headline check**: TXC-pro peak Δgc reproduces ~+1.574 (Aniket's
wasteland reference, hill-climbed TXC). Cross-check using
`results/c7_backtracking/aniket_reference/cut25/inducement_summary.csv`.
If TXC-base/pro fall well short, document honestly in c7.md and adjust
framing.

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

1. **Architecture-port blocker.** C7 needs 7 archs (TopK-SAE,
   Stacked-SAE, TFA, T-SAE-paper, MLC, TXC-base, TXC-pro). Only
   `topk_sae` is implemented today; the other 6 are agent_paper
   territory (`configs/locked_archs.yaml` is locked + the registry
   files belong to agent_paper per their briefing's C1+C2 todo).
   Current preflight reports 8 expected gaps. **Question**: should
   I (agent_back) wait for agent_paper to ship the ports, or am I
   authorised to land the 6 missing arch files myself with header
   attribution from `origin/han-phase7-unification:src/architectures/`
   and a courtesy ping to agent_paper for review? Either path
   unblocks C7 training (#8/#9/#10/#11/#12); the wait-path is
   stricter on territory but slower.
2. **Cohort-discovery shortcut.** Aniket's
   `flip_matrix.parquet` has the cohort qids he used (31 wrong + 30
   correct out of MATH-500). My current cohort builder relies on
   running R1-Distill-Llama on all 500 prompts at session start to
   re-discover the wrong/correct split (expensive — ~30 min on A40
   for one full pass). **Confirm**: OK to use Aniket's cohort qids
   as the canonical C7 cohort + skip cohort-discovery? They're frozen
   inputs; reproducibility is preserved by the parquet provenance.
3. **Steering-vector mining.** Aniket's pipeline mines per-arch
   top-32 features by D+/D- mean-difference on labeled sentences,
   then selects one feature per arch as the "steering feature".
   **Question**: should the mining + feature-selection logic live in
   `temp_bench.case_studies.backtracking` (my territory) or
   `temp_bench.eval.case_study` (shared scaffolding)? If shared, I'd
   add `mine_top_features` and `select_steering_feature` there; if
   case-study-local, in backtracking.py.
