# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-25, interim A40 pod (~hour 3 of ~12 funded;
`briefings/a40-bootstrap.md` supersedes all box facts).
**`briefings/panel-support-audit.md` is COMPLETE at its acceptance
gate** — items 1, 2, 2.5 (mirror close-out), 3 and 4 all DONE and
PUSHED, LOG entry each, full suite green (332 passed). The briefing
stays in place until mac-local review (its own rail). **No task is
mid-flight. I am in standby as CPU support for the two live panels.**

## Who / where (this pod)
`/workspace/agents/runpod-b/temp_xc`, CPU-ONLY
(`CUDA_VISIBLE_DEVICES=` empty — torch seeing 0 GPUs is CORRECT).
Every shell: `source /workspace/agents/runpod-b/env.sh`. Push:
`git push https://x-access-token:$(cat /workspace/.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
— pull-rebase before EVERY push; LOG.md conflicts append-only
(upstream first, mine last). Anything not pushed does not exist.

## Shipped this session (all on origin/arxiv)
1. **Item 1 (0c49e544)** — `support_stats/stage2_variance.py`
   pre-flighted + fixed for BOTH Stage-2 panels: `--row-layout auto`
   (paired rows carry the v2 flag + both column sets — v1 selection
   was EMPTY on the new panels before), `--post-k-rule times-T`
   (post at k = 8·T), `--seeds` filter (also fixed a LIVE abort: the
   λ̂ top-up seeds broke the legacy default), honest two-T degradation,
   diagnostic aborts. 12 fixture tests + byte-identity guard
   (`tests/test_stage2_variance_panels.py`). **d/e: exact commands in
   `support_stats/PANEL_RECIPES.md`.**
2. **Item 2 (ad08678a)** — `PROBE_V2_SPEC.md` § 0: first-class
   lower-bound limitation (v2 low by up to 0.18 at low truth + dense +
   p/n ≥ 1; both misses artifact-receipted). §§ 1–4 numbering stable.
3. **Item 2.5 (259a755e)** — mirror probe-truth CLOSE-OUT from pushed
   data only: final `probe_truth.json` + fig regenerated from committed
   shards (cells 27 → 18 disclosed; empty fig panels annotated);
   amended scope ADOPT-consistent, frozen scope AMBIGUOUS-unresolved
   permanently; mix arms/transfer/line D lost, never read; card § 10;
   `briefings/mirror-probe-truth.md` DELETED (retired).
4. **Item 3 (48a7f2be)** — `task_hunt/RECEIPTS.md` +
   `receipts_check.py`: 50 recomputed values / 16 claims, ALL PASS,
   pytest-wired (`tests/test_receipts_index.py`). Caught + corrected
   one live quote (dialevel T32 triple was truncated, not rounded:
   correct is +0.057 gpt2 / +0.063 gemma / +0.035 llama).
5. **Item 4 (dbe782e4)** — pre-staged panel analysis appended to
   `PANEL_RECIPES.md`: expected row decomposition (84 full / 24
   replication), harness→scorecard→RECEIPTS order of operations,
   skeleton LOG scorecard, receipt reading guide.

## Standby duties (in priority order, if resuming mid-window)
1. **Serve the panels.** d (oprate, GPUs 0–2) and e (fineweb gemma,
   GPUs 3–5; already pushing batches) will run the variance harness
   per PANEL_RECIPES.md. If either hits a harness abort they cannot
   read, that is my bug queue — fix with tests, push fast.
2. **RECEIPTS.md upkeep:** any new quotable number from the panels
   gets a row via `receipts_check.py` (must print ALL PASS).
3. **Do NOT:** train anything, run mirror cells, resume em-redo or
   factory work (PAUSED under force majeure); never relitigate the
   v1-canonical methods decision.
4. If both panels are done/receipted and nothing remains: tell the
   operator so the pod can be stopped (bootstrap rule).

## Standing context
- METHODS DECISION TAKEN (LOG 2026-07-25): v1 canonical, paired v2
  reported, never quote v2 as canonical; v2 numbers are lower bounds
  in the low-truth dense regime (SPEC § 0).
- pytest trap: commit new files before the full-suite run
  (`test_diff_hash_consistent_with_dirty`).
- All numbers script-derived; upstream is hot — small commits, push
  per batch. Rewrite this file before any compact.
