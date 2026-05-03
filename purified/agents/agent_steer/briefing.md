<!--
DRAFT — written by agent_paper 2026-05-03 for Han's review.
Han: rewrite the "Identity + mandate" section as you wish before
spinning up agent_steer; agents will not touch it after.
Section ownership rules: PROTOCOL.md § 14.
-->

---
agent: agent_steer
last_state_update: 2026-05-03T23:30:00Z
component: c5
---

## Identity + mandate (Han owns — agents do not edit)

You are **agent STEER**. You own C5 only. Files you may edit:
- `agents/agent_steer/briefing.md` (your own — agent-owned sections only)
- `docs/components/c5.md`
- `experiments/c5_steering/`
- Code under `src/temp_bench/` that you author + commit
  (`temp_bench.case_studies.steering`, `temp_bench.eval.steering`)
- `configs/datasources.yaml` — adding new C5 datasources is fine.

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

You are agent STEER, lead on **C5: RLHF steering** on
`google/gemma-2-2b-it` layer 13 — same subject as C3/C4. The case
study is the T-SAE paper § 4.4 sentiment-steering task.

Hardware: pod `4× A40`, pinned to **GPU 0**. Pod mode **`ephemeral`**:
`/workspace` is wiped on pod stop, HF is the source of truth.
Bootstrap pulls from `han1823123123/temp-bench-{models,data}`;
`cache.save_checkpoint` auto-pushes on save (push failure is fatal —
we cannot risk losing a multi-hour training run).

agent_back shares the pod on GPU 1 (separate component, separate
cache). GPUs 2 + 3 are spare pool slots — if you need more parallelism
you may claim one via `temp_bench.utils.gpu_locks.claim_gpu(idx)` and
launch a second process with `CUDA_VISIBLE_DEVICES=<idx>`. See
PROTOCOL.md § 13 for the Primary + Pool contract.

**You are gated on agent_nlp** — they build the Gemma-2-2b-IT L13
activation cache (~3 H100-hr) and push to HF temp-bench-data. Your
first session pulls that cache via `sync_from_hf.sh`, so you don't
rebuild. Provisioning order: spawn agent_steer **after** the cache
appears on HF (~T+3 hr), not at T+0.

Hypothesis (modest, from `docs/components/c5.md`): TXC-base + TXC-pro
produce coh-vs-success curves comparable to T-SAE; both match T-SAE
at coh ≥ 1.75. **This is "matches" not "beats"** — accepted in
exchange for stronger C3/C4 claims.

Steering protocol: **V7 tiled-broadcast** (per-token decoder-row
addition, stride-T blocks, single uniform δ within each block).
Chosen for arch-uniformity, not peak performance per arch. **Pre-test
that V7 is OK for TXC-pro** (subseq encoder + multi-distance
contrastive may break under V7); switch to PP if so.

**Excluded by design**: Y/W hill-climbing winners (Galaxy 8/11/18,
SoftMaxPool, ContrastiveMergeH8). They beat T-SAE on steering but
lose 0.005–0.020 probing AUC, inconsistent with "two TXCs everywhere."

Locked decisions in scope: #1 (two TXCs — DO NOT use Galaxy/SoftMaxPool
hill-climbing wins), #4 (cross-branch reads), #6 (HF repos), #7
(Bricken resample is C6-only by default; **C5 keeps it OFF** —
revisit only if time permits at the end of the paper sprint).

References:
- `agents/README.md` (your roster row + pod-mode contract)
- `docs/components/c5.md` (full setup, V7 protocol, hypothesis)
- `docs/paper/hardware.md` *Multi-GPU access* (Pool example)
- `decisions.md` (esp. #1)
- `papers/temporal_sae.md` § 4.4 (the case study)
- `PROTOCOL.md` § 11, § 12 (pinning), § 13 (Primary + Pool)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-03T23:30:00Z (first session, code-port complete)**

- `git HEAD`: `d94dc17e` (origin/final, after `git pull --rebase`).
- Local edits not yet committed: 4 new files + 1 rewritten stub
  (see *What I just did* below; `git status` for the list).
- Last leaderboard append: (none yet — gated on agent_nlp porting
  the three archs `tsae_paper`, `txc_base`, `txc_pro`).
- Last checkpoint saved: (none yet).
- Active GPU lock(s): none.
- Recent decisions in scope: #1 (two TXCs), #4 (cross-branch reads),
  #6 (HF repos), #7 (Bricken off for C5).
- In flight: code-port done (steering module + run.py + analysis.py
  + 17 unit tests, all green); briefing update; pending Han review of
  the open questions below before training cells.
- **NOT blocked on agent_nlp's act-cache** — verified
  ``act_cache_key=e4916bcae1881963`` is on HF
  (``han1823123123/temp-bench-data``, shape ``[24000, 128, 2304]``
  fp16, datasource ``gemma_2_2b_it_l13_fineweb_24k128``). Pulled
  locally at ``results/act_cache/e4916bcae1881963/``.

## What I just did (agent owns — overwrite)

Newest first.

- Wrote `tests/test_steering.py` — 17 unit tests covering V7 tile
  layout + trailing-block overwrite, PP overlap-averaging, concept
  set, `coh_success_curves` + `flatten_metrics` aggregation, feature
  selection argmax. Full suite: 68 passed.
- Wrote `experiments/c5_steering/run.py` — thin orchestration over
  `runner.run_cell` for 3 archs × 3 seeds, with `--pre-test-only`
  flag for the V7 health-check on TXC-pro per c5.md.
- Wrote `experiments/c5_steering/analysis.py` — leaderboard query →
  per-arch coh-vs-success curves with errorbars + AUTO-RESULTS
  markdown (placeholder until cells run).
- Rewrote `src/temp_bench/eval/steering.py` (was raise-NotImplemented
  stub) → re-export wrapper for `case_studies.steering` public types.
- Wrote `src/temp_bench/case_studies/steering.py` (~900 lines, single
  file per the `case_studies/backtracking.py` convention). Ports V7
  tiled-broadcast hook + PP fallback hook, 30-concept set verbatim
  from phase7, paper-§B.2-verbatim grading prompts, Sonnet judge with
  `judge_outputs.jsonl` persistence per CaseStudy contract, and
  `SteeringCaseStudy(CaseStudy)`. Provenance lines in the docstring
  cite specific phase7 file paths.
- Read `papers/temporal_sae.md` § 4.4–4.5 + Appendix B.2 for the case
  study spec; located phase7 implementation in
  `origin/han-phase7-unification:experiments/phase7_unification/case_studies/steering/`.
- Diagnosed `scripts/sync_from_hf.sh` failure: it calls
  `huggingface-cli download …`, which is deprecated in
  huggingface_hub 1.13. The CLI emits help text + exits without
  downloading. Worked around with direct `hf download`.
- Pre-flight: `set_agent_env.sh agent_steer` (GPU 0 / ephemeral),
  smoke test green, `git pull --rebase` brought in c7 backtracking
  artifacts, discarded a stale local downgrade of `datasets`
  in `uv.lock`.

## Next action (agent owns — overwrite)

**Pre-conditions (Han owns)**:
- Han ran `bash scripts/bootstrap_runpod.sh` on this pod (interactive)
  AND `bash /workspace/temp_xc/purified/scripts/add_agent_clone.sh
  agent_steer` to provision the second clone. Tokens are in
  `/workspace/.tokens/` (`anthropic_key`, `hf_token`, `gh_token` —
  no `gemini_key`; see Open Questions).
- The smoke test confirms the venv + `e4916bcae1881963` act-cache.

**On every fresh / `--continue` session:**
1. `cd /workspace/temp_xc_steer/purified && source scripts/set_agent_env.sh agent_steer`
2. `bash scripts/agent_smoke_test.sh`
3. (Ephemeral pod restart only): re-pull act-cache via
   `.venv/bin/hf download han1823123123/temp-bench-data --repo-type dataset --include "act_cache/e4916bcae1881963/**" --local-dir results`
   (`scripts/sync_from_hf.sh` itself is broken — see Open Q #2).
4. `git pull --rebase origin final`
5. `.venv/bin/python -m pytest tests/test_steering.py -q` — sanity-check
   the steering module after any pull.

**Then continue from here:**

A. **Wait for / coordinate with agent_nlp** to finish porting
   `tsae_paper`, `txc_base`, `txc_pro` arch classes
   (`temp_bench.architectures.{tsae,txc_base,txc_pro}`). Until those
   land, `experiments/c5_steering/run.py` will fail at
   `instantiate_arch(...)` for any cell. The smoke test already
   reports this as 8 *expected* gaps. Check progress with
   `.venv/bin/python -c "from temp_bench.config import load_arch, instantiate_arch;
   import importlib; m=importlib.import_module('temp_bench.architectures.tsae'); print(m)"`.

B. Once one arch is up, **smoke-test a tiny cell**:
   `TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering.run --archs topk_sae --seeds 42 --n-concepts 3 --strengths 100 --pre-test-only`
   (uses `topk_sae` since it IS ported, just to validate the
   end-to-end path — won't be a paper cell because c5.md says only
   T-SAE/TXC-base/TXC-pro). This will load Gemma-2-2b-IT (~5 GB
   bf16) and call Sonnet (~$0.05). Verify `judge_outputs.jsonl`
   appears under `results/runs/<eval_key>/`.

C. **Pre-test V7 on TXC-pro** (per Hard Rule "fall back to PP if
   degenerate"): once `txc_pro` arch + checkpoint are available,
   `--pre-test-only --archs txc_pro`. Read `mean_coh` from
   `results/runs/<eval_key>/_pre_test_v7/curves.json`. If ≤ 1.0 → use
   `--protocol pp` for the full sweep and document in c5.md.

D. **Full sweep**: `python -m experiments.c5_steering.run` with all
   defaults. 3 archs × 3 seeds × 30 concepts × 9 strengths = 810
   greedy generations × 2 judge calls = 1620 Sonnet calls per arch.
   At 5 workers, ~5 min/arch judge time + ~5 min/arch generation =
   ~15 min/arch + Sonnet cost ~$1.50/arch → ~$5 + 45 min total.

E. After sweep: `.venv/bin/python -c "from temp_bench import report; report.render(component='c5')"` — rewrites
   `docs/components/c5.md` AUTO-RESULTS block from the leaderboard.

## Don't repeat (agent owns — overwrite)

- **Hill-climbing winners** — Galaxy 8/11/18, SoftMaxPool,
  ContrastiveMergeH8 are excluded by decision #1. If TXC-base and
  TXC-pro lose to T-SAE, the paper accepts that — don't chase.
- **Skip the V7 pre-test** on TXC-pro — its subseq + multi-distance
  contrastive may not be V7-compatible. `--pre-test-only` is
  cheap (~1 min) and the c5.md hypothesis is contingent on it.
- **Forget HF push on save** — ephemeral pod; checkpoint loss = run loss.
  The runner's `cache.save_checkpoint` auto-pushes when
  `TEMP_BENCH_POD_MODE=ephemeral`; verify after first cell.
- **Rebuild the act-cache** — already on HF (`e4916bcae1881963`).
- **Wasteland imports** — `git show origin/han-phase7-unification:…`
  only; never `import experiments.phase7_unification…`.
- **Hand-edit `docs/components/c5.md` AUTO-RESULTS** — that block is
  owned by `experiments/c5_steering/analysis.py` + `temp_bench.report.render`.
- **Use `huggingface-cli download`** — deprecated; use `hf download`.
  See Open Q #2.
- **Touch `scripts/sync_from_hf.sh`** even though it's broken — it's
  shared infra; surface to Han instead of patching unilaterally.
- **Touch `temp_bench/utils/tokens.py`** to add a `gemini` slot — same
  rationale (shared infra). See Open Q #1.

## Open questions for Han (agent owns — overwrite)

1. **Judge: Gemini or Sonnet?** This briefing + `eval/case_study.py`
   docstring say "Gemini for C5, Sonnet for C7". But:
   - `temp_bench/utils/tokens.py` only knows `hf`/`anthropic`/`gh`
     — there's no `gemini` slot. `get_token('gemini')` would raise
     `ValueError`.
   - `/workspace/.tokens/` has `anthropic_key`/`hf_token`/`gh_token`,
     no Gemini key.
   - Phase 7's `grade_with_sonnet.py` uses Sonnet 4.6.
   - `case_studies/backtracking.py` (already merged) uses Sonnet 4.6
     and the same paper-§B.2-verbatim 0–3 grading rubric.

   **Decision I made (please confirm or correct)**: I implemented the
   judge as `SonnetSteeringJudge` using `claude-sonnet-4-6` and
   `temp_bench.utils.tokens.get_token('anthropic')`. Swapping to
   Gemini later is mechanical — it's a single class swap, the prompts
   + persistence schema stay identical (the `judge_id` /
   `judge_model` fields in `judge_outputs.jsonl` make the choice
   audit-able). If you want Gemini, I need:
   - a `gemini_key` file under `/workspace/.tokens/`,
   - one extra entry in `tokens.py::_FILENAMES` + `_ENV_VARS`,
   - and a `GeminiSteeringJudge` class I can write in
     `case_studies/steering.py`.

2. **`scripts/sync_from_hf.sh` is broken** — it shells out to
   `huggingface-cli download`, which `huggingface_hub` 1.13.0
   deprecated and now exits with help text. Fix is one line:
   `huggingface-cli download` → `hf download` (args identical:
   `--repo-type` / `--type`, `--include`, `--local-dir` all work).
   I've worked around it with direct `hf download` calls. **Could
   you (or agent_paper) land the one-line fix?** I haven't edited it
   because `scripts/` is shared infra used by every agent.

3. **Subject model + layer**: `c5.md` and the briefing say
   Gemma-2-2b-IT L13 (matches the `gemma_2_2b_it_l13_fineweb_24k128`
   datasource). Phase 7 uses Gemma-2-2b BASE L12 (per `_paths.py`).
   T-SAE paper § B.2 uses base/L12 too. **I've gone with IT/L13** for
   consistency with C3/C4. Worth confirming this is the agreed
   decision and not a typo in the c5.md hypothesis. Reviewers may
   ask "why not the same as the paper" — c5.md doesn't yet answer
   that.

4. **`run_cell` doesn't thread `eval_key` to `eval_fn`** — the case
   study needs to write `generations.jsonl`, `grades.jsonl`,
   `judge_outputs.jsonl`, `feature_selection.json` to
   `run_dir(eval_key)`. I worked around by recomputing `eval_key` in
   `run.py` and passing the workspace via `eval_cfg["_workspace"]`.
   This works (deterministic hash) but feels like a missing
   primitive. Worth a small framework PR by agent_paper to pass
   `_eval_key`/`_workspace` in the runner's `enriched_cfg`. Not
   urgent for me.

5. **C5 gen + judge cost / time budget**: full sweep at defaults is
   ~$5 + ~45 min Sonnet calls. Confirm OK before I kick it off.
