<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_back; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_back
last_state_update: 2026-05-04T15:55:00Z
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

### Han decisions 2026-05-04 PM (GPU re-allocation + preloaded batch_iter)

Two directives, effective on your next launch:

**(1) GPU pinning re-allocation — you get GPUs 0 and 2.** New A40 pod
split (Han 2026-05-04 PM): you get GPUs **0 and 2**; agent_steer gets
GPUs 1 and 3. Two dedicated GPUs each, no more borrow patterns. **Do
not touch GPUs 1 or 3** — those are agent_steer's.

Your in-flight v3 sweep launched 12:49 used all 4 GPUs (PIDs
39420-39423 on GPU 0/1/2/3 from the prior session). PIDs on GPUs 1+3
must be killed when this directive lands; their cells finish on GPUs
0+2 only. Sequence:

```bash
# Identify which PIDs are on GPUs 1 and 3 (your prior sweep launched
# the second proc on GPU 1, fourth on GPU 3 — verify before killing):
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv
# Kill the GPU-1 and GPU-3 processes only; let GPU-0 and GPU-2
# processes finish their current cells.
```

After current GPU-0 and GPU-2 cells finish, redistribute remaining
work across just those two GPUs:

```bash
# 7 archs × 1 seed (per Han's 4c2a400d "20K + seed=42 only" override) =
# 7 cells, run sequentially across 2 GPUs (3-4 per GPU).
CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 AGENT_NAME=agent_back \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m experiments.c7_backtracking.run \
  --archs txc_pro mlc tsae_paper --seeds 42 \
  > logs/c7_v4_gpu0.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 TQDM_DISABLE=1 AGENT_NAME=agent_back \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m experiments.c7_backtracking.run \
  --archs txc_base stacked_sae tfa topk_sae --seeds 42 \
  > logs/c7_v4_gpu2.log 2>&1 &
```

(Adjust the arch-to-GPU split to balance wall-time — txc_pro / mlc are
slow so put the lighter archs on the same GPU together.) The 20K cap
× 1 seed budget keeps total cells light enough for 2-GPU completion
within wall budget.

**(2) Adopt the preloaded batch_iter pattern.** agent_nlp landed
`temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` (commit
`e12dc719`) — a `.clone()`-based pre-materialization that gives ~1.4×
end-to-end trainer speedup (~3.4× on the data path). **It's not a
drop-in for C7** because your `_build_batch_iter` in
`experiments/c7_backtracking/run.py:100-132` samples T-token sliding
windows from `(N, L, d)` while the helper returns whole sequences.
Apply the `.clone()` pattern locally to your existing function:

```python
# At module scope in experiments/c7_backtracking/run.py:
_PRELOADED_C7_ACTS: dict[str, torch.Tensor] = {}

def _build_batch_iter(act_cache_key: str, *, batch_size: int = 256,
                      T: int = 5, seed: int = 42):
    from temp_bench.config import act_cache_dir, load_datasource, compute_act_cache_key
    ds = load_datasource(DATASOURCE)
    expected_key = compute_act_cache_key(ds)
    if expected_key != act_cache_key:
        raise RuntimeError(...)
    cache_dir = act_cache_dir(act_cache_key)
    cache_path = str(cache_dir / "resid_post_L10.npy")

    # Preload once per (process, cache_path); subsequent cells share.
    if cache_path not in _PRELOADED_C7_ACTS:
        mmapped = np.load(cache_path, mmap_mode="r")
        # .clone() is load-bearing — without it, torch.from_numpy
        # zero-copy wraps the mmap and page-faults persist.
        _PRELOADED_C7_ACTS[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_C7_ACTS[cache_path]
    N, L, d = acts.shape
    if L < T:
        raise RuntimeError(f"seq_len {L} < arch T={T}")
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        seq_idx = rng.integers(0, N, size=n)
        pos_idx = rng.integers(0, L - T + 1, size=n)
        out = torch.empty((n, T, d), dtype=torch.float32)
        for i in range(n):
            out[i] = acts[seq_idx[i], pos_idx[i]:pos_idx[i] + T].to(torch.float32)
        return out

    return batch_iter
```

**Determinism**: same `np.random.default_rng(seed)`, same fp32 contract
— `train_keys` unchanged, checkpoints bit-identical to the mmap path.
**No fairness implication; adopt mid-sweep is safe** — cells from the
.clone() path and the mmap path are interchangeable in the leaderboard
for the same `(act_cache_key, seed)`.

**RAM cost on A40 pod**: ~4.24 GB per process for the Llama BASE L10
cache (you noted the shape `(4044, 128, 4096)` fp16 in your briefing).
2 GPUs × 1 process each = ~8.5 GB total; agent_steer's ~14 GB Gemma
cache on the same pod (via 2 of their own processes) brings the pod
total to ~22-37 GB. A40 pod has ~64 GB system RAM — comfortable
headroom.

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

Hardware: pod `4× A40`, pinned to **GPUs 0 and 2** (Han 2026-05-04 PM
re-allocation). Pod mode **`ephemeral`**: HF is the source of truth,
auto-push on checkpoint save.

agent_steer shares the pod on **GPUs 1 and 3** (separate component +
separate cache). The A40 pod is now fully partitioned: 2 dedicated GPUs
per agent, no unassigned slots, **no borrow pattern**. Launch parallel
processes via `bash scripts/run_on_gpu.sh <idx> -- <command>` for
GPU 0 or 2 only — never touch 1 or 3. See PROTOCOL.md § 13.

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

**Last verified: 2026-05-04T15:55:00Z (v4 sweep launched ~15:50).**

- Clone: `/workspace/temp_xc/`. Env: `set_agent_env.sh agent_back`
  (NB: pinning is now **GPUs 0 + 2** per Han's 35fe822e directive —
  GPUs 1 + 3 belong to agent_steer; do NOT touch them).
  TEMP_BENCH_POD_MODE=ephemeral.
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

### Han 2026-05-04 PM directives — APPLIED

1. **`TrainingConfig` defaults at batch=1024, plateau=False, bf16**
   (commits 06681098 + 9718a442 by agent_paper).
2. **n_steps=20_000 deadline override** (Han, 2026-05-04 PM via
   commits 4c2a400d). Implemented locally as
   `TrainingConfig(n_steps=20_000)` in `run.py` + `analysis.py`
   (commit `05545dff`). Matches agent_nlp's c3+c4 pattern (513a85ea).
3. **GPU re-allocation: GPUs 0 + 2 only** (briefing 35fe822e).
   GPUs 1 + 3 are agent_steer's. No more borrow patterns.
4. **Preloaded `.clone()` batch_iter pattern adopted** (commit
   `26d793f1`). ~1.4× trainer speedup. Determinism unchanged
   (same RNG, same fp32, same train_key).

Old batch=256/30K + new batch=1024/25K cells stay in
`leaderboard.jsonl` for diff but are **filtered out** by
`analysis._valid_train_keys()` which canonicalises on the
20K override; only batch=1024/n_steps=20K cells appear in
AUTO-RESULTS.

### v4 sweep (LIVE — launched 15:50, relaunched 16:05 with train_log)

Initial v4 launched 15:50 (PIDs 45886+45889) was killed at ~16:04 after
~14 min training to add train_log persistence per agent_paper's
convergence-check pattern (commit f047509a — adopted from agent_nlp's
033a3eb6). Relaunch:

| GPU | PID | archs (in order) | status |
|---|---|---|---|
| 0 | 53575 | `txc_pro` → `mlc` → `tsae_paper` | starting up |
| 2 | 53572 | `tfa` → `txc_base` → `stacked_sae` → `topk_sae` | starting up |

Train_log lands at `logs/c7_b1024_{arch}_seed42_trainlog.json` per
cell after train_sae returns. Post-sweep: check final-1K-step loss
drop > 5% threshold per arch (decisions.md § 12) — flags any cells
that are still descending at step 20K as under-converged; surface in
c7.md as a caveat or selectively extend that arch.

Per-cell ETA at 20K + preload (calibrated from v3 measurements
+ 0.56× scaling for 20K + preload combined):
- txc_pro (2.68B): ~5-6hr cell (4-5hr train + 1.6hr eval)
- tfa (2.32B): ~4-5hr cell
- txc_base / mlc / stacked_sae (1.34B each): ~3-3.5hr cell
- tsae_paper / topk_sae (268M each): ~2.7hr cell

Relaunched 22:05 UTC.
GPU 0 total: txc_pro + mlc + tsae_paper ≈ 11hr → finishes ~09:00 UTC May 5.
GPU 2 total: tfa + txc_base + stacked_sae + topk_sae ≈ 13.5hr → finishes ~11:30 UTC May 5.
Critical path: GPU 2 (4 cells). 48-hr deadline = 2026-05-06 ~22:00 UTC,
so ~34hr buffer after sweep completes (Han confirmed 20K + buffer is OK).

Logs: `logs/c7_v4_gpu{0,2}.log`. Monitor `b6tdo9r4w` (persistent)
filters cell events + delta_gc + error signatures.

### v3 sweep (KILLED — 25K, pre-override)

Launched 12:49 on 4 GPUs (PIDs 39420-39423). Ran 2.5hr of training
before Han's 20K override directive. Killed at 15:43. Two orphan
checkpoints persisted to disk + HF before kill (filtered out of
canonical via 20K-override train_keys):
- `e18cf041874c1dc9` — tsae_paper @ 25K
- `36e4ae1f38e037d0` — txc_base @ 25K

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
- v3 sweep launched 12:49 across 4 GPUs (PIDs 39420-39423) at 25K.
- 4 ports in commit `b1baf484`: tfa + mlc + stacked_sae + _tfa_module.
- Per Han: arch porting is open-ownership ("first-needs-it ports it"),
  not agent_paper-gatekeeper. agent_nlp shipped txc_pro (`6ae94a74`).

**Session day 2 (2026-05-04 PM, ~15:00-15:50)** — three Han directives
landed in quick succession:
- Mirror equivalence verified: `meta-llama/Llama-3.1-8B` vs
  `NousResearch/Meta-Llama-3.1-8B` are bit-identical (max_abs_diff=0)
  on 5 prompts × {12-38} tokens, layer-10 residual. v3/v4 cells
  using the mirror are paper-valid (commit 0e09c867).
- **n_steps=20_000 deadline override** (Han + agent_nlp). Killed
  the v3 25K sweep (~2.5hr training lost across 4 GPUs); 2 orphan
  checkpoints persisted (e18cf041874c1dc9 + 36e4ae1f38e037d0).
  Implemented as `TrainingConfig(n_steps=20_000)` in run.py +
  analysis.py canonical filter (commit 05545dff).
- **GPU re-allocation**: agent_back ⇢ GPUs 0+2; agent_steer ⇢
  GPUs 1+3 (Han via 35fe822e). v3 ran on all 4 GPUs (briefly);
  v4 restricted to GPUs 0 + 2.
- **Preloaded `.clone()` batch_iter pattern** adopted from
  agent_nlp's e12dc719 helper. Not a drop-in for C7 (we sample
  T-token windows, helper returns whole sequences) so applied
  the .clone() pattern locally to `_build_batch_iter` (commit 26d793f1).
  ~1.4× trainer speedup. Determinism unchanged.
- **v4 sweep launched ~15:50** on GPUs 0 + 2 only:
  - GPU 0 PID 45889: txc_pro → mlc → tsae_paper
  - GPU 2 PID 45886: tfa → txc_base → stacked_sae → topk_sae
  - bf16 casts confirmed (txc_pro 21.5 GB, tfa 18.5 GB).
  - ETA: full sweep ~04:00-08:00 UTC tomorrow.

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

The v4 sweep (PIDs 45886 GPU 2, 45889 GPU 0) is mid-training on 7
archs × seed=42 at 20K-override defaults + preloaded batch_iter.

A. **First action**: pull rebase, then verify sweep PIDs alive:
   ```
   ps -p $(cat /tmp/c7_v4_gpu0.pid) $(cat /tmp/c7_v4_gpu2.pid) -o pid,etime,cmd
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
   ```
   If a python PID is dead, check `logs/c7_v4_gpu{0,2}.log` tail for
   the failure. Common: MFS I/O error at judge persist (Errno 5);
   retry with same args — runner cache-hits the train_key, the
   existing_keys() check fast-forwards already-judged panels.

B. **Re-arm the persistent monitor** (mine timed out across compact):
   ```python
   Monitor(
     command="tail -F logs/c7_v4_gpu0.log logs/c7_v4_gpu2.log 2>/dev/null | "
             "grep -E --line-buffered '(c7\\.run\\] cell (arch|failed)|"
             "delta_gc|judge done|Traceback|FAILED|OOM|bf16 cast|"
             "phase1 ready|panels|computing PR-AUC|dispatching)'",
     persistent=True, timeout_ms=3600000,
   )
   ```

C. **As cells complete** (event matches `delta_gc`):
   - Run `bash scripts/c7_post_sweep.sh` to render c7.md AUTO-RESULTS
     incrementally + commit + push.
   - Script auto-filters via `analysis._valid_train_keys()` (20K
     canonical) so old batch=256 + 25K cells are excluded.

D. **Single-seed only** (Han 2026-05-04 PM): seeds 1 + 2 are NOT
   running — Han said "stay single seed". v4 = 7 archs × seed 42.
   Do not launch additional seeds without explicit go-ahead.

E. **GPU constraint**: do NOT launch on GPUs 1 or 3 — those belong
   to agent_steer (briefing 35fe822e). Restrict any retries or
   reruns to CUDA_VISIBLE_DEVICES=0 or =2.

F. **Headline test**: does TXC-pro reproduce Aniket's hill-climbed
   +1.574 peak Δgc under our locked arch + 20K + batch=1024 + preload?
   Cross-check at
   `results/c7_backtracking/aniket_reference/cut25/inducement_summary.csv`.

**Failure modes to expect during v4**:
- MFS I/O error → cell fails mid-judge or mid-save. Retry under same
  eval_key cache-hits the existing rows.
- OOM on GPU 2 if all 4 archs share peak mem → unlikely (sequential),
  but if seen, drop the heaviest from GPU 2 to its own launch.
- tfa or txc_pro silent-failure → check for `train_step done` or
  `bf16 cast` in log; if neither in 30 min, the arch may have hung
  on initialization. Re-launch.
- preload pattern OOM-on-RAM (4.24 GB cache × N processes) is
  comfortable headroom on 64 GB pod even with agent_steer's caches.

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
7. **Multi-seed scope — RESOLVED 2026-05-04 PM.** Han: "stay single
   seed". v4 = 7 archs × seed 42 only; no seeds 1+2 launches.

### Closed (no action needed)

- agent_paper PR for batch=1024 directive applied; analysis filter
  canonicalises on 20K + batch=1024.
- Memory issues at d_sae=32768 fp32 Adam → bf16 cast for >1B archs
  (commit `9cfd99df`).
- 20K override applied (commit 05545dff). Matches agent_nlp pattern.
- Preloaded `.clone()` batch_iter (commit 26d793f1). 1.4× speedup.
- GPU re-allocation: agent_back ⇢ GPUs 0+2; agent_steer ⇢ GPUs 1+3
  (briefing 35fe822e).
