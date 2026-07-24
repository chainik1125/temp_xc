# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 ~23:00 UTC (mid-campaign) — executing
`briefings/mirror-probe-truth.md` (overnight; **results by Saturday
morning PT**). Card FROZEN, all builds committed, Stage 1 + Stage 2
landed and committed (LOG entry "mirror probe-truth campaign … card
FROZEN, Stage 1 + Stage 2 landed"). **Stage 3 (the 132-cell training
ladder) is RUNNING.** Nothing pushed yet this campaign — the leaderboard
is being appended to by the running grid, so the push happens at the end.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU.
`/workspace/.agent_id` = runpod-b. Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.

## The task
Produce the receipt that fires mac-local's **pre-registered 4-branch
rule** (LOG ≈ line 2212): 1 v2 tracks truth where v1 sags ⇒ ADOPT; 2 both
track truth ⇒ DECLINE; 3 v2 reports ABOVE truth ⇒ REJECT for headline
use; 4 ambiguous/incomplete ⇒ v1 stays canonical with a caveat. **I
produce the receipt, NOT the verdict.** A result arguing AGAINST v2 is
first-class and is reported FIRST. Under every branch the window > token
ORDERING survives, so no outcome is incentivised.

## Where things stand

**Committed (in order):** `support_synthetic/CARD_PROBE_TRUTH.md`
(frozen before any cell) → `run_probe_truth.py` → `probe_truth_calib.py`
→ `probe_truth_anchor.py` → `analyze_probe_truth.py` →
`render_probe_truth.py` → the G3 amendment → per-seed calib CLI → the
p=8192 corner drop → increment-1 LOG entry + Stage-2 results.

**Stage 1 (calibration, exact truth, off-leaderboard) — RUNNING**, three
per-seed processes (`--seeds {1,2,42} --out probe_truth_calib_s<seed>.json
--resume`), logs `$SCRATCH/calib_s*.log`. Gate G1 passing: reproduces the
bench's own constants (DPI floor 0.41, ceilings 0.91/0.99) to 0.0043 and
the anchor recovers exact truth to 0.0013. Early signal at T16, `full`
arm, truth 0.986: v1 0.986→0.982→**0.912**→0.943 at p/n
0.004→0.25→1.0→2.0; v2 0.986→0.985→0.984→0.983. No cell above truth yet.

**Stage 2 (22 surviving checkpoints) — COMPLETE.** 28 cells (22 new v2
rows + 6 eval cache hits on cells Stage 3 had already trained), 0 dup
keys; anchors licensed 28/28, v1 replication ≤ 7.6e-9, anchor gaps
≤ 0.0011. All at p/n ≤ 0.125 — a genuine low-p/n control and nothing
more.

**Stage 3 (the body) — RUNNING**: `run_probe_truth.py --stage train
--workers 10`, log `$SCRATCH/train_grid.log`, 132 cells (66 trained + 66
untrained), lines C/P/M/S per card § 2.3. ~10/132 at 25 min; expect
~4–6 h under contention with the calib processes.

## What to do next (in order)
1. Wait for the training grid (`grep "DONE" $SCRATCH/train_grid.log`).
2. `.venv/bin/python -m …support_synthetic.probe_truth_anchor train 6`
   (the anchors; ~1–1.5 h).
3. `…analyze_probe_truth` → `results/probe_truth.json`, then
   `…render_probe_truth` → `figs/probe_truth.png`.
4. Read the receipt and write the **scorecard** LOG paragraph: which
   prediction held, which was falsified, what it licenses — and if it
   undercuts adopting v2, **say so first**.
5. Commit results + `results/leaderboard.jsonl` +
   `checkpoints/manifest.jsonl` (deferred until the grid stops appending),
   run the full pytest suite on a clean tree, pull-rebase, push.
6. If time: the companion note on a defensible `doc_mean_only_auc` KILL
   threshold. **Checked: runpod's doc-level bootstrap CIs are NOT on the
   branch yet**, so the note the briefing describes cannot rest on them;
   if they have not landed, either write the weaker version off the
   screened candidates' observed `doc_identity_check.json` spread and say
   exactly what it does not rest on, or leave it undone and say so.

## Campaign design — the two things that reshaped it
- **The p/n trap.** v1's `n_windows` is hardcoded 1024 ⇒ n = 1024·(32/T)
  = 2048 at T16, so the real panel (d_sae 2048) sits at **p/n = 1.00**
  while the mirror's committed budget sits at **0.001–0.08**. Running the
  mirror at canonical budget would show "both probes agree" for reasons
  invisible in its own numbers and read as branch 2. Card § 1.1 discloses
  the deviation: canonical line kept as the low-p/n control, ladder
  extended to span the real regime. **p/n, not T, is the x-axis.**
- **The checkpoint prune.** 843 mirror rows / 843 train_keys, **22
  checkpoints on disk**, manifest 9878 rows with **0 HF refs** (no
  restore path) ⇒ the briefing's "cheap eval-only pass may answer by
  breakfast" is a low-p/n control only; the campaign is TRAINING-bound.

## Disclosed self-corrections (both in the LOG, neither silent)
- **G3 mis-scaled.** The card froze |chance| ≤ 0.05; the chance floor is
  a *fitted* probe's held-out r on permuted targets, null spread ~√(p/n)
  (0.125 at the first cells). Analysis computes both readings and reports
  the branch under each exclusion set.
- **p = 8192 calibration corner dropped on cost** (>30 min/cell, 9 of
  them); no p>n coverage lost — the p=4096 cell's own nw sweep already
  gives exact truth at p/n 0.5/1.0/2.0.
- Anchors that cannot be licensed at any budget within the cap (p > 4096,
  line S at T16) are computed at the floor budget, not the cap.

## Standing context
- Shared branch: pull-rebase before EVERY push; LOG.md conflicts keep the
  upstream entry then re-append mine; commit SUBJECTS not SHAs; scripts
  and cards committed BEFORE outputs; no reviewer/meeting quotes in
  tracked files; all numbers script-derived (three were eyeballed into
  the increment-1 LOG entry and corrected by re-deriving them before the
  commit was amended — do this check every time).
- Reproduction claims: "bit-identical **on the build platform**".
- pytest trap: untracked files break `test_diff_hash_consistent_with_dirty`
  — the untracked `results/probe_truth*.json` shards must be committed (or
  moved) before a full-suite run.
- Rewrite this file before any compact.
