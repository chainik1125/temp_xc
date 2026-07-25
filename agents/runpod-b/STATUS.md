# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-25 ~02:00 UTC — **mid-execution** of
`briefings/mirror-probe-truth.md` (overnight; deadline moved to
**Saturday midday PT** by the briefing AMENDMENT). NOT awaiting a new
task. **The AMENDMENT (mac-local, 2026-07-25, binding) arrived and is
APPLIED**; increment 2 is pushed ("probe-truth increment 2: briefing
AMENDMENT applied — the item-1 receipt fires ADOPT-consistent on the
amended scope"). Stage 1 + 2 COMPLETE and pushed; Stage-3 training grid
RUNNING (~36/132).

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU.
`/workspace/.agent_id` = runpod-b. Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
Scratch/logs: `/tmp/claude-1000/-workspace-temp-xc/07d1d2b4-d3f7-44aa-823a-5cd659bac28e/scratchpad/`
(`train_grid.log`, `train_lineD.log`, `calib_mix_s{1,2,42}.log`).

## The AMENDMENT and what it changed (read this before anything else)
`briefings/mirror-probe-truth.md` gained a binding amendment: (1) branches
fire ONLY on evidence swept through p/n ≈ 1.0 — p/n ≪ 0.1 fires NO
branch; (2) the direct known-truth probe (= Stage 1) is the PRIORITY
branch input, shipped the moment it exists; (3) 22/843 coverage accepted;
(4) deadline Saturday midday PT, branch 4 a good outcome if nothing
fires. Applied as card § 9 (post-freeze appendix, frozen §§ 2–6
untouched) + `analyze_probe_truth.py` emitting BOTH
`branch_evidence` (amended scope, PRIMARY) and
`branch_evidence_frozen_card_scope` (verbatim retention).

**The amended-scope receipt is COMPLETE and pushed: `ADOPT-consistent`.**
12 exact-truth cells at p/n ∈ {1.0, 2.0}: v1 sags 7/8 signal cells
(to −0.445; reports ≈ 0 where truth is 0.41 at 6% density), v2 tracks
10/12, v2 above truth on 0 cells, DECLINE support 1/8. The two v2
misses are the standing caveat: v2 is a LOWER BOUND at low truth +
dense (d2 to −0.180) — `PROBE_V2_SPEC.md` should carry this.
Frozen-card-scope label is AMBIGUOUS (G4) — a mid-run artifact until
the trained anchors run. All in LOG increment 2 + `probe_truth.json`.

## Running right now (check these first on resume)
| job | state |
|---|---|
| **Stage-3 main grid** (`run_probe_truth --stage train --workers 10`) | RUNNING ~36/132, ~3.4 h in. Critical path. |
| Watcher task `b9g1we1f8` | background `until grep DONE train_grid.log` → `kill -CONT` line-D PIDs automatically. |
| Line D (`--lines D --out-suffix _lineD --workers 4`) | SIGSTOPped at 7/12; auto-resumed by the watcher. |
| Mix-arm calibration (3 procs, niced 15, `probe_truth_calib_mix_s{1,2,42}.json`) | RUNNING, ~9% CPU each while the grid holds the box; no cells written yet. **Renice up when the grid ends.** Mix arms strengthen the amended input + unlock transfer Test B; they are enhancement, NOT blockers. |

Writers were SIGSTOPped twice for stash→rebase→pop→push cycles (upstream
is hot tonight); both times resumed cleanly. Pattern: pause grid PIDs,
stash the 3 appending files, rebase, pop (union-merge JSONL on
conflict), CONT, push.

## What to do next (in order)
1. Wait for the main grid; watcher auto-resumes line D to 12/12.
2. `probe_truth_anchor train 6` and `probe_truth_anchor train 6 _lineD`
   (~1–1.5 h). Anchors gate the FROZEN-scope P1/P2/P3/G4 only — the
   amended-scope label does not consume them.
3. Re-run `probe_truth_transfer` once mix shards exist (Test B currently
   under-resolved, 3 truth points; do not read it before).
4. Re-run `analyze_probe_truth` (final) → `probe_truth.json`;
   `render_probe_truth` → fig (title now carries the scope tag).
5. **Scorecard** LOG paragraph. **Two requirements mac-local pinned in
   the briefing NOTE (2026-07-25), binding on the final receipt:**
   (i) the low-truth + dense + p/n ≥ 1 downward bias (up to 0.18) goes
   in the HEADLINE reading, not the caveat list — "v2 tightens a lower
   bound" is a materially different claim from "v2 measures truth";
   (ii) if the mix arms BREAK the inversion, say that first and
   loudest — explicit backing to report against adoption. Then the
   structure: (a) the amended-scope ADOPT-consistent receipt WITH the
   lower-bound qualifier inline; (b) the frozen-scope label and why the
   trained ladder under-describes (truth ~0.95, bias negligible by
   mechanism; P1 FAIL there fires no branch per amendment item 1);
   (c) the p_eff finding (nominal p/n overstates the operative ratio by
   3–30×: post/T16/k8 = 70 active of 2048 ⇒ 0.034), bears on
   `PROBE_V2_SPEC.md`'s n_rows ≥ 8·p line; (d) coverage honesty.
   Decision remains mac-local's — increment 2 is READ, decision
   explicitly NOT yet taken (waits on mix arms + trained ladder).
6. Final commits: leaderboard + manifest + grid shards + lineD +
   transfer + receipt + fig; full pytest on a clean tree; pull-rebase;
   push. (Calib s1/s2/s42 shards are already committed COMPLETE —
   increment 2 fixed increment 1's mid-run snapshot.)
7. ~~doc_mean_only_auc KILL-threshold note~~ **SUPERSEDED, do not
   write**: the overnight review RATIFIED "disclosure statistic that
   TRIGGERS A CONTROL — do NOT promote to a kill bar" (LOG "REVIEW
   overnight wave" § 4; 11-face index + causal dialevel + punctint-q
   0.901 counterexample). Record the supersession in the scorecard
   increment, one line, with the pointer.
8. **QUEUED NEXT (auto-start, no idle): `briefings/panel-support-audit.md`**
   — begins ONLY once this campaign hits its acceptance gate and is
   pushed (the briefing's own rail + Han's instruction). Read its item 1
   first: pre-flight `support_stats/stage2_variance.py` against BOTH new
   panels' k_pos = 8·T row shape (the duplicate-cell abort I fixed once
   is guaranteed by construction on their datasources); then item 2:
   `PROBE_V2_SPEC.md` carries my lower-bound caveat as a first-class
   limitation; items 3 (RECEIPTS.md claim→artifact index) and 4
   (pre-staged panel analysis) as the night allows.

## Standing context
- Shared branch: pull-rebase before EVERY push; LOG.md conflicts keep
  upstream then re-append mine; commit SUBJECTS not SHAs; scripts and
  cards committed BEFORE outputs; no reviewer/meeting quotes in tracked
  files; **all numbers script-derived** (three eyeballed numbers were
  caught and re-derived in increment 1 — check every time).
- pytest trap: untracked files break
  `test_diff_hash_consistent_with_dirty` — commit all
  `results/probe_truth*` shards before the full-suite run.
- Disclosed self-corrections live in LOG increments 1–2 and card § 9;
  G3's primary reading applies NO exclusion (premise falsified,
  23/23 excluded cell-seeds pass the anchor licence); P4 scored on
  seed-means (per-draw sensitivity 1/288 disclosed).
- My p/n-trap catch is now program-binding ("METHODS RULE AMENDED to
  matched p/n" upstream); other agents' endgame allocation is locked
  (stage2-oprate/d, stage2-fineweb/e, factory-broad-3/runpod) — no
  action for me, but their panels carry paired v1+v2 columns, so the
  probe decision this campaign feeds never forces them a re-run.
- Rewrite this file before any compact.
