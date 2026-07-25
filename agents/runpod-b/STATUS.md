# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-25, on the interim A40 pod (force majeure;
`briefings/a40-bootstrap.md` supersedes all box facts). Executing
`briefings/panel-support-audit.md`. Items 1, 2 and 2.5 are DONE and
PUSHED; item 3 (RECEIPTS index) is IN PROGRESS; item 4 (pre-staged
panel analysis) if the clock allows.

## Who / where (this pod)
`/workspace/agents/runpod-b/temp_xc`, CPU-ONLY
(`CUDA_VISIBLE_DEVICES=` empty — torch seeing 0 GPUs is CORRECT).
Every shell: `source /workspace/agents/runpod-b/env.sh`. Push:
`git push https://x-access-token:$(cat /workspace/.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
— pull-rebase before EVERY push; LOG.md conflicts resolved append-only
(upstream first, mine last; strip markers). ~12 funded hours from
session start (~hour 2 now); anything not pushed does not exist.

## Done and pushed this session (commits on origin/arxiv)
1. **Item 1 (0c49e544):** `support_stats/stage2_variance.py`
   pre-flighted + fixed for BOTH Stage-2 panels: `--row-layout auto`
   (paired layout: new-panel rows carry the v2 flag + both column
   sets), `--post-k-rule times-T` (post at k = 8·T), `--seeds` filter
   (also fixes the live legacy abort from top-up seeds 3–5),
   honest two-T degradation (trend skipped with reason), diagnostic
   aborts. 12 fixture tests + byte-identity guard
   (`tests/test_stage2_variance_panels.py`); full suite 331 passed.
   **d/e run their exact commands from
   `support_stats/PANEL_RECIPES.md`** (d substitutes its ds key).
2. **Item 2 (ad08678a):** `PROBE_V2_SPEC.md` § 0 — first-class
   lower-bound limitation (v2 low by up to 0.18 at low truth + dense +
   p/n ≥ 1; the two misses receipted from `probe_truth.json`
   `amendment.rows`: token/6%/truth 0.412 → v2 0.299 @p/n 1.0,
   0.232 @p/n 2.0). §§ 1–4 numbering untouched. Status block records
   the taken decision (v1 canonical; spec = post-deadline candidate).
3. **Item 2.5 (259a755e):** mirror probe-truth CLOSE-OUT from pushed
   data only. Final `probe_truth.json` + fig regenerated from
   committed shards (cells 27 → 18 disclosed — increment 2 embedded
   ~13 never-pushed mid-run cell-seeds; empty fig panels annotated
   "lost mid-run"). Labels: amended scope ADOPT-consistent (A_P1 7/8,
   A_P2 10/12, over-truth 0/12); frozen scope AMBIGUOUS-unresolved
   permanently (anchors never ran). Mix arms/transfer/line D: lost,
   never read. Card § 10 close-out; `briefings/mirror-probe-truth.md`
   DELETED (retired). Scorecard LOG entry follows the binding NOTE
   (lower bound in headline; mix arms stated first; p_eff quoted from
   pushed numbers only — increment 2's "70/2048 ⇒ 0.034, 3–30×" sat
   partly on unpushed cells, do not quote as receipted).

## In progress: item 3 — `experiments/explorations/task_hunt/RECEIPTS.md`
Claim→artifact index, one row per rebuttal-quotable number: claim as
stated / artifact path + JSON key / producing commit / recomputed-now
value / PASS-FAIL. Build a checker script (`receipts_check.py` beside
it) so every number is script-derived; flag mismatches LOUDLY. Seed
list (briefing item 3; extend it): λ̂ panel cells + trend p = 0.0093;
margin trend p = 0.0046; pre/T8 n = 6 CI [0.179, 0.235]; pre-vs-tsae
NOT-bounded (paired LB −0.041, Welch LB −0.016, p = 0.082 — never
quote as significant); shuffle/anticipation receipt; tsae fairness
(max |paired D| 0.011 vs 0.05); split-forensics zero leakage; five
Stage-1 KEEP screens WITH corpus size; amended order finding g_order
band + dialevel counterexample; NEW: mirror Stage-1 receipt numbers
(now closed) + PROBE_V2_SPEC § 0 numbers. Then item 4 if time.

## Standing context
- Both Stage-2 panels are live on this pod (d: oprate, GPUs 0–2 — was
  still claiming/cache-building; e: fineweb gemma, GPUs 3–5 — pushing
  batches + doc-identity floor receipts already). My harness serves
  them; if either pings about variance receipts, point at
  PANEL_RECIPES.md.
- METHODS DECISION TAKEN (LOG 2026-07-25): v1 canonical, paired v2
  reported, never quote v2 as canonical. Do not relitigate; no mirror
  training, no new mirror cells, em-redo/factory PAUSED.
- pytest trap: commit new files before the full-suite run
  (`test_diff_hash_consistent_with_dirty`).
- All numbers script-derived; upstream is hot — small commits, push
  per completed batch. Rewrite this file before any compact.
