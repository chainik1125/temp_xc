# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-25 ~01:20 UTC (pre-compact #4) — **mid-execution**
of `briefings/mirror-probe-truth.md` (overnight; results due Saturday
morning PT). NOT awaiting a new task. Card frozen, all builds committed,
Stage 1 COMPLETE, Stage 2 COMPLETE, **Stage 3 training grid RUNNING
(34/132)**. Nothing pushed yet this campaign — the leaderboard is being
appended to by the running grid, so the push happens at the end.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU.
`/workspace/.agent_id` = runpod-b. Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
Scratch/logs: `/tmp/claude-1000/-workspace-temp-xc/07d1d2b4-d3f7-44aa-823a-5cd659bac28e/scratchpad/`
(`train_grid.log`, `train_lineD.log`, `calib_mix_s{1,2,42}.log`).

## Running right now (check these first on resume)
| job | cmd | state |
|---|---|---|
| **Stage-3 main grid** | `run_probe_truth --stage train --workers 10` | RUNNING 34/132, ~3.2 h in. Critical path. |
| Line D (added arm) | `run_probe_truth --stage train --lines D --out-suffix _lineD --workers 4` | **SIGSTOPped** (was an 18 GB / 6.5-core hog starving the grid). `kill -CONT` its PIDs when the main grid finishes. 7/12 done. |
| Mix-arm calibration | `probe_truth_calib --seeds <s> --arms mix00,mix10,mix20,mix35 --out probe_truth_calib_mix_s<s>.json --resume` | RUNNING, niced 15, 3 procs, no cells written yet. |

Box was oversubscribed (load 95); line D suspended to fix it. Load ~81.

## The task
Produce the receipt that fires mac-local's **pre-registered 4-branch rule**
(LOG ≈ line 2212): 1 ADOPT / 2 DECLINE / 3 REJECT / 4 AMBIGUOUS. **I produce
the receipt, NOT the verdict.** A result arguing AGAINST v2 is first-class
and is reported FIRST. No branch costs the window > token ordering, so no
outcome is incentivised.

## Findings so far (all script-derived, in `results/probe_truth*.json`)

**Stage 1 — constructed codes, truth EXACT, 108 cells × 3 seeds, COMPLETE.**
The mechanism, which is the campaign's core content: **the probe's downward
bias is governed by the UNEXPLAINED variance (1 − ρ²), not by p/n alone.**
At p/n = 1.0 v1's sag from exact truth is −0.07 when truth is 0.99 and
−0.33 when truth is 0.41; at the real panel's ~6% code density that becomes
**−0.42 — v1 reports −0.007 where truth is 0.41**, while v2 reports 0.292.
**v2 never exceeds truth anywhere** (max above truth: full −0.0001, token
−0.003) ⇒ no branch-3 support. But v2 is itself biased LOW (up to −0.18 at
low truth + dense + high p/n), so v2 numbers are lower bounds, not
estimates — a caveat `PROBE_V2_SPEC.md` does not currently carry.

**Stage 2 — 22 surviving checkpoints, COMPLETE.** 28 cells (22 new v2 rows +
6 eval cache hits), 0 dup keys, anchors licensed 28/28, v1 replication
≤ 7.6e-9, anchor gaps ≤ 0.0011. All at p/n ≤ 0.125 — a low-p/n control and
nothing more.

**Transfer test (`probe_truth_transfer.py`, committed evidence only).**
Test A HOLDS on the real panel: at matched nominal p/n = 1.00,
`txc_batchtopk_post` sparse (nnz 7.8) has gap +0.032 vs dense (nnz 127.9)
+0.184. **And the nominal capacity arithmetic is wrong**: the operative
ratio is p_eff/n over ACTIVE columns — post/T16/k8 is labelled p/n = 1.00
but has 70 active of 2048 ⇒ p_eff/n = 0.034; stacked/T16 is labelled 16.0
but sits at 0.789. This bears directly on `PROBE_V2_SPEC.md`'s
`n_rows ≥ 8·p` adequacy line and its Stacked p > n disclosure, both stated
in nominal p. Test B (inversion) is under-resolved with only 3 truth points
(5/12 cells inconsistent) — **do not read it until the mix arms land**, then
re-run `probe_truth_transfer`.

## What to do next (in order)
1. Wait for the main grid (`grep DONE .../train_grid.log`). Then
   `kill -CONT` the line-D PIDs (`pgrep -f "lines D"`) to finish 12/12.
2. `probe_truth_anchor train 6` and `probe_truth_anchor train 6 _lineD`
   (~1–1.5 h). Anchors are the gate on G4/P1/P2/P3.
3. Re-run `probe_truth_transfer` once the mix shards exist.
4. `analyze_probe_truth` → `results/probe_truth.json`; `render_probe_truth`
   → `figs/probe_truth.png`.
5. **Scorecard** LOG paragraph: which prediction held, which was falsified,
   what it licenses — and if it undercuts adopting v2, say so FIRST.
   **Expect a tension to report honestly:** the trained mirror ladder will
   likely show P1 FAILING (both probes within bar of truth at p/n ≥ 0.5)
   because every trained window arch on this mirror recovers λ at ~0.95,
   where the bias is tiny — so the mechanical label may read
   DECLINE-consistent while the exact-truth evidence at the panel's actual
   recovery level (0.13–0.26) points the other way. Report the mechanical
   label per the frozen card AND why it under-describes the evidence.
6. Commit results + `results/leaderboard.jsonl` + `checkpoints/manifest.jsonl`
   (deferred while the grid appends). **The committed calib shards are a
   MID-RUN snapshot (20 of 36 cells); the final commit must re-add the
   complete files.** Then full pytest on a clean tree, pull-rebase, push.
7. If time: the `doc_mean_only_auc` KILL-threshold note. **Checked: runpod's
   doc-level bootstrap CIs are NOT on the branch**, so the note the briefing
   describes cannot rest on them — either write the weaker version off the
   screened candidates' `doc_identity_check.json` spread and say exactly
   what it does not rest on, or leave it undone and say so.

## Design context that reshaped the campaign
- **The p/n trap** (card § 1.1): v1's hardcoded nw = 1024 ⇒ n = 1024·(32/T)
  = 2048 at T16, so the real panel sits at p/n = 1.00 while the mirror's
  committed budget sits at 0.001–0.08. Running the mirror at canonical
  budget would show "both probes agree" for reasons invisible in its own
  numbers and read as branch 2.
- **The checkpoint prune**: 843 mirror rows / 843 train_keys, 22 checkpoints
  on disk, manifest 9878 rows with 0 HF refs ⇒ the campaign is
  TRAINING-bound, not eval-bound.
- **The truth-level gap**: the frozen ladder is matched to the panel in p/n
  and density but NOT in true recovery, which turns out to dominate. Hence
  line D (per-token archs at the DPI floor) and the mix arms (tunable truth).

## Disclosed self-corrections (all in the LOG or commit messages, none silent)
- **G3 mis-scaled** (flat |chance| ≤ 0.05 vs a statistic whose null spread is
  ~√(p/n)); primary reading applies NO exclusion because the gate's premise
  is falsified — all 23 excluded cell-seeds with anchors pass the anchor
  licence. Branch reported under all three exclusion sets; identical.
- **P4 scored per-draw fired REJECT-consistent on 1 draw of 648** on a truth-0
  target; now scored on seed-means per the card's own § 4. Per-draw count
  kept as a sensitivity.
- **G1 regime-resolved**: the anchor recovers exact truth on 27/27 high-truth
  and 27/27 null cells but misses by up to 0.089 (mean −0.029) at truth 0.41.
  Anchor licence gained a regime condition (`anchor ≥ 0.8`); direction stated
  (an under-estimating anchor makes P1 conservative, P4 anti-conservative).
- **p = 8192 calibration corner dropped on cost**; no p > n coverage lost.
- **Unlicensable anchors** (p > 4096) computed at the floor budget, not the cap.

## Standing context
- Shared branch: pull-rebase before EVERY push; LOG.md conflicts keep the
  upstream entry then re-append mine; commit SUBJECTS not SHAs; scripts and
  cards committed BEFORE outputs; no reviewer/meeting quotes in tracked
  files; **all numbers script-derived** — three were eyeballed into the
  increment-1 LOG entry and corrected by re-deriving them before the commit
  was amended; do that check every time.
- Reproduction claims: "bit-identical **on the build platform**".
- pytest trap: untracked files break `test_diff_hash_consistent_with_dirty`
  — commit the `results/probe_truth*.json` shards before a full-suite run.
- Rewrite this file before any compact.
