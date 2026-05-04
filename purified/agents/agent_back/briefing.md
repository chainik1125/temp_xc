<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_back; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_back
last_state_update: 2026-05-04T12:55:00Z
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

**Last verified: 2026-05-04T12:55:00Z (mid-v3-sweep)**

- Clone: `/workspace/temp_xc/`. Env: `set_agent_env.sh agent_back` →
  CUDA_VISIBLE_DEVICES=1, TEMP_BENCH_POD_MODE=ephemeral.
- Git remote auth: gh PAT in `/workspace/.tokens/gh_token`. Pull
  rebase + push works after stashing untracked `logs/` +
  `results/c7_backtracking/stage_a/sentence_acts_L10.npz` (HF-backed,
  not committed — large).

### Infrastructure (all DONE + pushed)

- **Activation cache** at `results/act_cache/fb2a74be884e512a/resid_post_L10.npy`
  (4.24 GB float16, shape (4044, 128, 4096) — Llama-3.1-8B BASE L10
  via NousResearch mirror; HF-pushed).
- **Sentence-acts cache** at
  `results/c7_backtracking/stage_a/sentence_acts_L10.npz`
  (25204 sent × 6 × 4096 fp32, 12.6% positive class, HF-pushed under
  `c7_backtracking/stage_a/`). Per-arch T-slicing in `mine_top_features`
  + `run_arch_evaluation` PR-AUC encode handles arch_T < 6 (txc_base
  T=5) and arch_T > 6 (txc_pro T=10) by trimming/padding the [-13..-8]
  window.
- **All 7 C7 archs instantiate** at d_sae=32768. Sizes:
  topk_sae 268M / tsae_paper 268M / txc_base 1.34B / txc_pro 2.68B /
  tfa 2.32B / mlc 1.34B / stacked_sae 1.34B. >1B archs bf16-casted in
  `my_train_fn` to fit A40 (commit 9cfd99df).
- **My ports** (commit b1baf484): tfa + mlc + stacked_sae + _tfa_module.
  agent_nlp shipped txc_pro (6ae94a74). agent_paper shipped topk_sae
  + tsae_paper + txc_base earlier.
- **Pipeline modules** (committed):
  - `src/temp_bench/case_studies/backtracking.py` — full case study
    (StageA/Cohort/SteeringHook/extract_boxed/answers_match/cut25/
    SonnetBacktrackingJudge/compute_delta_gc/compute_pr_auc_at_S/
    extract_labeled_sentence_acts/mine_top_features/run_arch_evaluation/
    BacktrackingCaseStudy)
  - `src/temp_bench/data/nlp/ward.py` — C7-specific corpus loader +
    cache_activations (sibling to agent_paper's `cache.py`).
  - `experiments/c7_backtracking/{run.py, analysis.py, smoke.py}`.
  - `scripts/c7_run_sweep.sh`, `c7_run_sweep_pool.sh`, `c7_post_sweep.sh`.
- **18 c7-backtracking tests** (`tests/test_backtracking.py`) — all pass.

### Han 2026-05-04 PM directive (decisions.md § 12) — APPLIED

`TrainingConfig` defaults are now **batch=1024, n_steps=25_000,
plateau_early_stop=False, precision=bf16**. Updated upstream by
agent_paper (commits 06681098 + 9718a442). Old batch=256 / 30k cells
stay in `leaderboard.jsonl` for diff but are **filtered out** by
`analysis._valid_train_keys()` (commit ab02aea2) — which deterministically
computes the canonical new train_keys from current defaults and
ignores everything else.

### v3 sweep (LIVE — launched 12:49 across 4 GPUs)

| GPU | PID | archs (in order) | status |
|---|---|---|---|
| 0 | 39420 | `txc_pro` → `mlc` | bf16 cast (21.5 GB), training |
| 1 | 39421 | `txc_base` → `stacked_sae` | training |
| 2 | 39422 | `tfa` → `topk_sae` | bf16 cast (18.5 GB), training |
| 3 | 39423 | `tsae_paper` | training |

Per-cell ETA at new defaults: ~30 min train + ~95 min eval = ~125 min.
Each GPU runs 1-2 cells sequentially. **Total sweep ETA ~16:30-17:00.**

Logs: `logs/c7_v3_gpu{0,1,2,3}.log`. Monitor `brx2lk13p` (persistent)
filters `(cell|panels|dispatching|judge done|computing PR-AUC|
phase1 ready|delta_gc|Traceback|FAILED|OOM|bf16 cast)` events.

### v1/v2 sweep RESULTS (old config; filtered out of AUTO-RESULTS)

These are the batch=256/30k cells. `analysis._valid_train_keys()`
excludes them. Leaderboard rows kept for cross-config diff:

| arch | seed | eval_key | peak Δgc | peak mag | pr_auc S=32 |
|---|---|---|---:|---:|---:|
| topk_sae | 42 | 28c40a2a3a0bbd59 | +0.361 | +16 | 0.243 |
| txc_base | 42 | 2549ebb929060421 | +0.393 | -16 | 0.262 |
| txc_pro | 42 | 9aabb8f9f47852b5 | +0.273 (partial 313/1525 judges) | -12 | n/a |

Direction is paper-faithful (TXC peaks at mag<0; SAE peaks at mag>0)
but magnitude is 17–25% of Aniket's hill-climbed +1.574. Hypothesis:
batch=256 under-trains the SAEs.

### Failures handled / lessons learned

- **MFS I/O error (Errno 5) is intermittent on this pod.** Three cells
  hit it during judge persistence or manifest append. Workaround: cells
  that fail mid-judge keep their partial `judge_outputs.jsonl`; a
  retry under same eval_key cache-hits the existing rows via
  `existing_keys()` and only does the missing calls. Minimal cost on
  retry.
- **`compute_delta_gc` got a string-label row from another agent's
  case study** → patched to coerce `int(label)` and skip non-parseable
  rows (commit 3cff8554).
- **txc_pro train_step needs seq_len ≥ T_max + max_shift = 12** →
  `_spec_window_size(spec)` in `run.py` handles per-arch (commit
  5e2bc1e6).
- **GPU memory leaks across cells** → added gc.collect() +
  empty_cache() in run.py finally block (same commit).
- **manifest.jsonl + leaderboard.jsonl conflict markers** (from a
  failed `git stash pop`) caused `JSONDecodeError`. Cleaned with
  `grep -v -E "^<<<<|^====|^>>>>"`. Git `stash push` for the working
  tree before pull-rebase needs `stash pop` AFTER, with conflict
  resolution if the file was append-mutated remotely.

Pipeline auto-renders c7.md via `bash scripts/c7_post_sweep.sh`.

## What I just did (agent owns — overwrite)

**Session day 0 (2026-05-03 evening)** — bootstrap:
- Provisioning + Stage-A port (commit `e93acf2c`) + cut25 reference
  port (`be47e886`) + case_studies/backtracking.py skeleton.

**Session day 1 (2026-05-04 morning)** — pipeline + first sweep:
- C7 reasoning-LM pipeline + load_cohort_from_parquet + run_phase1_unsteered
  + mine_top_features + run_arch_evaluation + BacktrackingCaseStudy
  (commit `c6abd159`).
- `temp_bench.data.nlp.ward.py` C7-specific corpus loader (sibling to
  agent_paper's `cache.py`).
- `configs/datasources.yaml` — added `llama_3_1_8b_base_l10_ward_nousmirror`
  (Meta gate workaround, mirror is byte-identical).
- Built + HF-pushed activation cache (`fb2a74be884e512a`) at 4.24 GB.
- `experiments/c7_backtracking/{run.py, analysis.py, smoke.py}`.
- `extract_labeled_sentence_acts` + sentence-acts cache (1.1 GB,
  HF-pushed). Atomic-write fix for partial-write race (`48ae027b`).
- 18 unit tests (`tests/test_backtracking.py`) all passing.
- Smoke validated end-to-end (eval_key `a9b4ea184f7a477a`).
- v1 sweep launched on GPU 1 + GPU 3, hit several issues:
  - tsae_paper save_checkpoint failed — non-contiguous W_enc; agent_nlp
    fixed in `af552412` (after my sweep imported old code).
  - txc_pro + tfa OOM'd at fp32 Adam (~42 GB). Added bf16-cast for
    >1B archs (commit `9cfd99df`).
  - txc_pro train_step needed seq_len ≥ 12. Added `_spec_window_size`
    helper (commit `5e2bc1e6`).
  - mining + PR-AUC encode crashed on T mismatch (sentence_acts T=6
    but txc_base wants T=5 / txc_pro wants T=10). Added per-arch
    T-slicing (commit `aaf17f16`).
  - Multiple MFS I/O errors during judge persist + manifest append
    (Errno 5). Cells partially completed; retries via `existing_keys`
    cache fast-forward.
- v1 yielded 3 leaderboard cells (topk_sae +0.361, txc_base +0.393,
  txc_pro +0.273 partial). All paper-direction-correct but at 17-25%
  of Aniket's hill-climbed +1.574 → suspected under-training at
  batch=256.

**Session day 2 (2026-05-04 afternoon)** — Han's batch=1024 directive:
- Han + agent_paper + agent_nlp identified batch=256 / n_steps=30k
  as severely under-trained vs Phase 5 reference. Reverted earlier
  contrastive-only batch bump (a9200560) for fairness.
- New `TrainingConfig` defaults: batch=1024, n_steps=25_000,
  plateau=False, bf16 (commits 06681098 + 9718a442 by agent_paper).
- I killed all v1/v2 sweeps mid-flight per Han's "STOP EVERYTHING".
- Filtered AUTO-RESULTS to current-config cells only via
  `analysis._valid_train_keys()` (commit `ab02aea2`).
- v3 sweep launched 12:49 across 4 GPUs (PIDs 39420-39423). Each cell
  trains 25k steps then evaluates 25-mag grid. ETA 16:30-17:00.
- 4 ports in commit `b1baf484`: tfa + mlc + stacked_sae + _tfa_module.
- Per Han: arch porting is open-ownership ("first-needs-it ports it"),
  not agent_paper-gatekeeper. agent_nlp shipped txc_pro (`6ae94a74`)
  in the same window.

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

**Where I'm picking up next (post-compact):**

The v3 sweep (4 parallel processes 39420-39423) is mid-training
on all 7 archs × seed=42 at the new defaults. Picking back up:

A. **First action**: pull rebase to stay current. Then check sweep
   PIDs are alive:
   ```
   ps -p 39420,39421,39422,39423 -o pid,etime,cmd 2>/dev/null
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
   ```
   If any are dead, check the corresponding `logs/c7_v3_gpu{0..3}.log`
   for the failure. Likely cause: MFS I/O error at judge persist or
   leaderboard append. Retry with same args — cache hits on train +
   existing_keys() fast-forwards judge calls already done.

B. **Re-arm the persistent monitor** (mine timed out across compact):
   ```python
   Monitor(
     command="tail -F logs/c7_v3_gpu0.log logs/c7_v3_gpu1.log "
             "logs/c7_v3_gpu2.log logs/c7_v3_gpu3.log 2>/dev/null | "
             "grep -E --line-buffered '(c7\.run\] cell (arch|failed)|"
             "delta_gc|judge done|Traceback|FAILED|OOM)'",
     persistent=True, timeout_ms=3600000,
   )
   ```

C. **As cells complete** (event matches `delta_gc`):
   - Run `bash scripts/c7_post_sweep.sh` to render c7.md AUTO-RESULTS
     incrementally.
   - The script auto-filters via `analysis._valid_train_keys()` so
     old batch=256 cells are excluded.

D. **When all 7 cells DONE on seed=42**: launch seeds 1 + 2 if time:
   ```
   for SEED in 1 2; do
     # split across GPUs as before
     CUDA_VISIBLE_DEVICES=0 ... --archs txc_pro mlc --seeds $SEED &
     # ... etc
   done
   ```

E. **Mirror-equivalence check** (Han 2026-05-04 OQ #2): once Meta HF
   access lands (`han1823123123/temp-bench-models` cache should auto-
   pick it up), write
   `experiments/c7_backtracking/check_mirror_equivalence.py` — load
   `meta-llama/Llama-3.1-8B` AND `NousResearch/Meta-Llama-3.1-8B`,
   compare layer-10 residual activations element-wise on 5 prompts ×
   50 tokens, report max abs diff. **Threshold: max abs diff < 1e-5**.
   If passes, both datasources are interchangeable for paper claims.
   If fails, must wait for Meta access.

F. **Headline test**: does any C7 arch (under proper batch=1024 / 25k
   training) reproduce Aniket's hill-climbed +1.574 peak Δgc on
   inducement? **txc_pro is the candidate** (matryoshka + multi-distance
   contrastive + subseq sampling). Cross-check at
   `results/c7_backtracking/aniket_reference/cut25/inducement_summary.csv`.

**Failure modes to expect during v3**:
- MFS I/O error → cell fails mid-judge or mid-save. Retry under same
  eval_key cache-hits the existing rows.
- OOM at batch=1024 → my run.py already bf16-casts >1B archs. If
  still OOM, drop to batch=512 (per Han's note).
- tsae_paper non-contiguous W_enc → agent_nlp's `af552412` fixed
  this; current code is fine.
- agent_nlp's txc_pro could have bugs → if cell fails on
  `txc_pro.encode` or `train_step`, surface in OQ + skip arch.

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

1. **Arch ports — RESOLVED 2026-05-04 PM.** Han confirmed
   "first-needs-it ports it" (open ownership on the .py files). 4
   ports landed by me (tfa, mlc, stacked_sae, _tfa_module). agent_nlp
   shipped txc_pro. All 7 archs instantiate.
2. **Cohort-from-parquet — adopted.** Stays.
3. **Mining-in-case-studies — adopted.** Stays.
4. **NousResearch mirror — RESOLVED 2026-05-04 13:01.** Meta access
   landed; ran `experiments/c7_backtracking/check_mirror_equivalence.py`
   (commit 0e09c867) on 5 prompts × {12-38} tokens, comparing layer-10
   residual element-wise. **PASS: max_abs_diff = 0.000e+00, rmse =
   0.000e+00** — the two repos are bit-exact identical. v3 sweep cells
   (which use the mirror datasource) stay paper-valid. No re-run
   needed; no DATASOURCE switch needed. Both datasource entries can
   coexist; analysis filter accepts either.
5. **ward.py vs cache.py split** — defer. agent_paper hasn't asked
   for unification. The two paths share the `act_cache_dir(act_cache_key)`
   convention and produce compatible memmaps; only the corpus +
   layer_specs sidecars differ. If a refactor is needed for C3/C4/C5
   alignment, agent_paper drives it.
6. **Sentence-acts HF sync** — defer. The cache is small enough that
   re-extraction (~10 min) is cheaper than expanding `sync_from_hf.sh`'s
   contract. Document in `experiments/c7_backtracking/README.md` instead.
7. **Multi-seed scope** — single-seed v3 is the minimum for paper
   results. Seeds 1 + 2 are nice-to-have for stderr reporting. Plan:
   only attempt seeds 1+2 if v3 single-seed completes by ~17:00 with
   remaining wall budget.

### Closed (no action needed)

- agent_paper PR for batch=1024 directive applied; my analysis filter
  ensures clean separation of v1 (batch=256) and v3 (batch=1024) cells
  in AUTO-RESULTS.
- Memory issues at d_sae=32768 fp32 Adam → bf16 cast for >1B archs
  (commit `9cfd99df`).
