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

### Han decisions 2026-05-04 (NEW — C5 metric mismatch caught)

**The C5 headline metric was wrong** — Han caught it from the c5.md
plot ("all the points horizontally aligned, success rates tiny"
compared to wasteland's `unified-pareto.md` peak15 numbers ~1.5).

Two metrics, different semantics:

- **Old (your implementation, faithful to my buggy spec)**:
  ``success_at_coh_<τ>`` = mean(success_grade ≥ 2 | coh_grade ≥ τ),
  averaged over strengths. **Binary fraction in [0, 1].**
- **New (wasteland-comparable, now headline)**:
  ``peak_success_grade_at_coh_<τ>`` = for each strength, mean
  success_grade (0-3 continuous) over generations with coh ≥ τ;
  take MAX over strengths. **Continuous in [0, 3].**

The old metric collapses across coh thresholds because nearly all
success ≥ 2 events also have coh ≥ 2.0 — your numbers
``success_at_coh_1.75 == success_at_coh_2.0`` for every arch confirm
this. The new metric preserves dynamic range, comparable to
wasteland anchors (1.133 / 0.411 etc.) and the T-SAE paper § B.2 0-3
scale.

**What I (agent_paper) already did this session:**
1. Added ``peak_success_grade_at_coh_<τ>`` to ``coh_success_curves``
   in ``temp_bench.case_studies.steering`` — future cells emit it
   automatically.
2. Added ``mean_success_grade_at_coh_per_strength`` (per-strength
   continuous means) so the per-cell ``metrics.json`` retains the
   data needed for the peak metric.
3. Added a backfill helper:
   ``temp_bench.case_studies.steering.reaggregate_from_judge_outputs(
   judge_outputs_jsonl_path)`` — reads a cell's persisted judge
   calls and returns the new flat metrics dict. Use this to backfill
   existing cells without re-judging.
4. Updated ``experiments/c5_steering/analysis.py`` to render BOTH
   metrics in the AUTO-RESULTS block (peak-grade as headline when
   present; binary fraction always as supplementary). Re-rendered
   ``docs/components/c5.md``.
5. Updated c5.md "Metric" subsection to spec the new headline.

**What you (agent_steer) need to do this session:**

1. **Backfill the existing 9 cells.** Your judge_outputs.jsonl files
   are on the A40 pod (and pushed to HF via push_run_dir.py). For
   each c5 leaderboard row's eval_key, read the local
   ``results/runs/<eval_key>/judge_outputs.jsonl``, run
   ``reaggregate_from_judge_outputs(...)``, and emit a NEW
   leaderboard row with the new metrics + a bumped
   ``EVAL_PROTOCOL_VERSION`` (e.g. "1.0.1") so it doesn't collide
   with the old. The old rows stay in the leaderboard for
   reproducibility / diff.

   Sketch:
   ```python
   from pathlib import Path
   from temp_bench.cache import append_leaderboard, leaderboard_path
   from temp_bench.case_studies.steering import reaggregate_from_judge_outputs
   from temp_bench.report import query_leaderboard
   from temp_bench.schemas import LeaderboardRow
   import json, datetime, hashlib

   for r in query_leaderboard(component="c5"):
       jpath = Path("results/runs") / r.eval_key / "judge_outputs.jsonl"
       if not jpath.exists():
           continue
       new_metrics = reaggregate_from_judge_outputs(jpath)
       new_eval_key = hashlib.sha256(
           f"{r.eval_key}_v1_0_1".encode()).hexdigest()[:16]
       append_leaderboard(LeaderboardRow(
           eval_key=new_eval_key,
           train_key=r.train_key,
           act_cache_key=r.act_cache_key,
           component="c5",
           arch=r.arch,
           arch_version=r.arch_version,
           seed=r.seed,
           datasource=r.datasource,
           eval_protocol_version="1.0.1",
           eval_cfg={**r.eval_cfg, "rebuild_from": r.eval_key,
                     "metric_set": "v1_0_1_with_peak_grade"},
           metrics=new_metrics,
           primary_metric="peak_success_grade_at_coh_1.75",
           agent="agent_steer",
           ts=datetime.datetime.now(datetime.timezone.utc).strftime(
               "%Y-%m-%dT%H:%M:%SZ"),
       ))
   ```

2. **Re-render**: `python -c "from temp_bench import report;
   report.render(component='c5')"` — the AUTO-RESULTS block in
   c5.md will now show the wasteland-comparable peak grade as the
   headline.

3. **Update your future ``run.py``** to use
   ``primary_metric="peak_success_grade_at_coh_1.75"`` for new cells.
   The flatten_metrics function already emits both keys; runner.run_cell
   takes whichever you pass as `primary_metric`.

4. **Sanity check**: after backfill, your peak grades should be in
   the 0.5–2.5 range (T-SAE anchor was ~1.13 at coh ≥ 1.5; ~0.41 at
   coh ≥ 1.75). If your TXC archs come in around 1.0–1.5 at
   coh ≥ 1.75, that's the "matches T-SAE at high coh" hypothesis
   reproducing. If they come in much lower (say 0.1–0.3), it's the
   honest-negative the binary metric was already showing — but with
   credible numbers reviewers can compare to the paper.

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
cache). GPUs 2 + 3 are unassigned — to use them, launch a second
process with `bash scripts/run_on_gpu.sh <idx> -- <command>` (sets
`CUDA_VISIBLE_DEVICES=<idx>` for the subprocess only). No lockfile
manager — read peer's "Current state" + `nvidia-smi` before
borrowing, update your own state with the borrow + ETA. See
PROTOCOL.md § 13 *GPU sharing convention*.

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
- `PROTOCOL.md` § 11, § 12 (pinning), § 13 (GPU sharing convention)

---

## Current state (agent owns — overwrite at every compact)

**Last verified: 2026-05-04T13:00Z (post-metric-fix backfill).**

- `git HEAD`: pushed through `20ba7913` (sweep complete) plus this
  turn's metric backfill (rebased on top of agent_paper's
  `40a184bd Agent PAPER: C5 metric fix`).
- **C5 sweep complete — 9 of 9 cells + 9 backfilled rows**:
  Headline (peak success grade @ coh ≥ 1.75, 0–3 scale):
    - tsae_paper × {42, 1, 2}: 0.367, 0.400, 0.300 → mean **0.356 ± 0.029**
    - txc_base × {42, 1, 2}: 0.300, 0.400, 0.476 → mean **0.392 ± 0.051**
    - txc_pro × {42, 1, 2}: 0.375, 0.389, 0.300 → mean **0.355 ± 0.028** (n_steps=6000)
  All three archs land within 1 stderr — hypothesis "TXC matches T-SAE"
  is **supported** on the wasteland-comparable continuous metric.
  Old binary `success_at_coh_<τ>` numbers preserved as supplementary
  in c5.md (showed misleading 2× tsae lead due to threshold-event
  co-occurrence collapsing dynamic range).
- **Backfilled `eval_protocol_version="1.0.1"` rows**: 9 new rows
  with `metric_set="v1_0_1_with_peak_grade"` and `rebuild_from=<orig_eval_key>`,
  primary_metric `peak_success_grade_at_coh_1.75`. Original 1.0.0
  rows kept for reproducibility/diff per agent_paper's spec.
- **Helper fix landed**:
  `temp_bench.case_studies.steering.reaggregate_from_judge_outputs`
  now handles BOTH on-disk schemas: per-generation rows (with both
  grades) AND per-call rows (head + label format that
  SonnetSteeringJudge actually writes). Without this fix the helper
  silently returned 0s for every cell (because every per-call row
  has only one grade → both-grades check failed).
- Active GPU usage: none.
- Recent decisions in scope: #1, #4, #6, #7. NEW: c5 headline metric
  is peak_success_grade_at_coh_1.75, not the binary fraction.
- Active GPU usage: GPU 0 (txc_base seed=2), GPU 2 (txc_pro seed=2).
- Recent decisions in scope: #1, #4, #6, #7. Remember:
  paper-deviation n_steps=6000 for txc_pro × {42, 1}.

**Honest finding (refutes c5.md hypothesis)**: tsae_paper
outperforms TXC archs on success @ coh ≥ 1.75 by ~2x (0.067 vs
0.036). TXC archs have higher mean coherence (1.9–2.2 vs 1.6) but
don't reach the success threshold as often. Caveats: IT/L13 vs
paper's base/L12; txc_pro paper-deviation; 2-seed averages for
TXC archs.

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
