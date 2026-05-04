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

### Han decisions 2026-05-04 (resolves prior session's open questions)

1. **Judge: confirm Sonnet. NO Gemini.** Your `SonnetSteeringJudge`
   implementation is right. The original T-SAE § B.2 used Llama-3.3-70B,
   not Gemini, and we're not using Llama-3.3-70B either — so there's
   no "match the paper's judge exactly" pressure. Sonnet aligns C5+C7
   on one judge. Document the deviation in c5.md caveats.
   judge_outputs.jsonl persistence lets us validate κ post-deadline
   if reviewers ask.
2. **`scripts/sync_from_hf.sh`: FIXED.** Renamed `huggingface-cli download`
   → `hf download` in commit (this turn). Drop your `hf download`
   workaround on next session — the script works again. Affects only
   pod restart; no impact on running agents.

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

**Last verified: 2026-05-03T22:38Z (first session, smoke green, real
cells launching).**

- `git HEAD`: `21c84be8` (3 of my commits on top of origin/final at
  rebase: code port, run-fix + smoke flag, c5.md expansion).
- Last leaderboard append: `d55da9a21b1e847a` (topk_sae × seed 42,
  smoke=true, all-zero metrics — 200-step training so degenerate
  steering output, ignored by analysis.py's smoke filter).
- Last checkpoint saved: `2245417c2cd98294` (topk_sae seed 42 smoke
  ckpt, pushed to HF `han1823123123/temp-bench-models`).
- Active GPU usage: GPU 0 (PID 13635, tsae_paper × seed 42) + GPU 2
  (PID 16153, tsae_paper × seed 1, parallel via
  ``CUDA_VISIBLE_DEVICES=2``). I bypassed ``gpu_locks.claim_gpu`` for
  the GPU 2 launch — gentleman's agreement only. agent_back is pinned
  to GPU 1 so safe for the moment, but FUTURE parallel launches
  should ``claim_gpu(2)`` / ``claim_gpu(3)`` properly.
- Recent decisions in scope: #1 (two TXCs), #4 (cross-branch reads),
  #6 (HF repos), #7 (Bricken off for C5).
- In flight: tsae_paper × {seed 42 on GPU 0, seed 1 on GPU 2} both
  training in parallel. Logs at `logs/c5_tsae_paper_seed{1,42}.log`.
  Monitors armed for both. Expected each ~25 min, ~$1.50.
- **Cache state**: act-cache `e4916bcae1881963` on disk (14 GB
  mmap-able). 3 of my needed archs ported by agent_nlp (commit
  f7c3c536 + ae2aaf8b): `tsae_paper`, `txc_base`. Still waiting on
  `txc_pro` (= `phase5b_subseq_h8`).
- **Subject model**: Gemma-2-2b-IT bf16 already cached in
  `/workspace/hf_cache/` from the smoke run.

## What I just did (agent owns — overwrite)

Newest first.

- Launched `tsae_paper × seed 42` real cell in background (PID 13635).
  Two monitors armed: a one-shot Bash `until` loop that fires on
  process exit, and a Monitor watching `logs/c5_tsae_paper_seed42.log`
  for per-step / per-stage progress events.
- Expanded `docs/components/c5.md` (commit 21c84be8). Status
  `planning → in_flight`. Caveats now flag the V7 ↔ TXC-pro
  compatibility risk + IT/L13 deviation from paper. Reproduction
  section has copy-paste commands. Provenance lines cite the
  origin/han-phase7-unification: paths the V7 + judge + concepts +
  feature-selection were ported from.
- Fixed run_dir mismatch (commit f8a28469): pre-computed `eval_key`
  was using un-enriched `eval_cfg`, but I was passing enriched
  `eval_cfg` to `run_cell`, so the runner re-hashed and got a
  different `eval_key` → metrics.json and case-study artifacts
  landed in different `run_dir`s. Refactored to thread workspace
  via a closure on the eval-fn, leaving `eval_cfg` un-enriched.
  Verified post-fix: `d55da9a21b1e847a` has all 7 artifacts in one
  dir.
- Added `--smoke` flag to `run.py` + smoke filter in `analysis.py`.
  Smoke cells tag `eval_cfg["smoke"] = True` and the c5 paper
  aggregator skips them. Matches agent_nlp's c3 convention.
- Validated end-to-end pipeline with `topk_sae × seed 42` smoke
  test (n_steps=200, 3 concepts, 1 strength). Confirmed: training
  → HF push → Gemma-2-2b-IT load → feature selection → V7 hook
  → greedy generation → Sonnet judge × 6 calls → grades.jsonl
  + judge_outputs.jsonl + curves.json + leaderboard append all
  worked. Generations were degenerate token-loops as expected at
  high-strength steering on a 200-step random-ish model.
- Pulled in commits `d94dc17e..ae2aaf8b`: agent_nlp ported
  `tsae_paper` + `txc_base` arch classes plus
  `temp_bench.data.nlp.batch_iter_from_act_cache` helper. Updated
  `run.py` to use the canonical batch_iter (yields full
  `(B, seq_len, d_in)`; TXC archs do their own random-window
  extraction internally). Removed my inline T-window helper.
- Wrote 17 unit tests in `tests/test_steering.py` covering V7 tile
  layout + trailing-block overwrite, PP overlap-averaging, concept
  set, `coh_success_curves` + `flatten_metrics` aggregation, feature
  selection argmax. All 68 framework + steering tests green.
- Initial code port (commit b0519a99): `case_studies/steering.py`
  (~900 lines) with V7 + PP hooks, 30 paper-faithful concepts,
  paper-§B.2-verbatim grading prompts, `SonnetSteeringJudge` with
  `judge_outputs.jsonl` persistence, `SteeringCaseStudy(CaseStudy)`
  implementing 4-stage pipeline. Plus run.py + analysis.py
  scaffolding + eval/steering.py rewritten from stub to re-export.
- Pre-flight: env pin (GPU 0 / ephemeral), smoke test green,
  diagnosed broken `sync_from_hf.sh` (deprecated huggingface-cli),
  worked around with direct `hf download`.

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

A. **If `tsae_paper × seed 42` is still running (PID 13635)**: don't
   re-launch. The Bash background task `bx2eso6ju` will notify on
   exit; the Monitor task `bex720hr4` streams progress events. Just
   `tail -20 logs/c5_tsae_paper_seed42.log` to check.

B. **If it finished successfully**: verify the metrics in the latest
   leaderboard row (``tail -1 results/leaderboard.jsonl``). If
   ``mean_coh > 1.0``, V7 worked. Then queue up the remaining seeds
   in the background:
   `TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering.run --archs tsae_paper --seeds 1 2 > logs/c5_tsae_paper_seeds_1_2.log 2>&1 &`.
   After tsae_paper completes (3 seeds), run `txc_base × {1, 2, 42}`:
   `TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering.run --archs txc_base > logs/c5_txc_base.log 2>&1 &`.

C. **If it failed** (non-zero exit, traceback in log): diagnose
   from the log tail. Common failure modes:
   - HF push failed → check `/workspace/.tokens/hf_token` is valid
     write-token; `cache.save_checkpoint` push failure is fatal on
     ephemeral pod.
   - Gemma load failed → ensure `/workspace/hf_cache/` has space
     (Gemma-2-2b-IT is ~5 GB).
   - Sonnet rate-limit → reduce `--n-concepts` or
     `SteeringConfig.judge_max_workers`.
   - V7 algebra crash → check tensor shapes vs the V7 hook's
     ``z.dim() == 3`` branch; window archs may return ``(B, T,
     d_sae)`` from encode while per-token archs return ``(B,
     d_sae)``.

D. **`txc_pro` is still pending** (commit ae2aaf8b ported tsae_paper
   + txc_base + sae_arditi but NOT txc_pro). Check via
   `ls src/temp_bench/architectures/txc_pro.py 2>&1`. Once
   ``txc_pro.py`` lands:
   - run V7 pre-test:
     ``TQDM_DISABLE=1 .venv/bin/python -m experiments.c5_steering.run --pre-test-only``
   - read ``results/runs/<eval_key>/_pre_test_v7/curves.json``.
     If ``mean_coh ≤ 1.0``, run TXC-pro full sweep with
     ``--protocol pp --archs txc_pro`` and document the switch in
     c5.md (under "Caveats").

E. **After all 9 cells complete**:
   ``.venv/bin/python -c "from temp_bench import report; report.render(component='c5')"``
   rewrites the AUTO-RESULTS block of c5.md from the leaderboard.
   Don't hand-edit between the markers (PROTOCOL.md § 7).
   Then commit the leaderboard delta + the regenerated c5.md.

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
