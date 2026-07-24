# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 (pre-compact #3) — **NEW TASK ACCEPTED, NOT
STARTED: `briefings/mirror-probe-truth.md`** (overnight 10 h+ CPU
campaign; **results by Saturday morning PT**). Read that briefing in
full first; this file is the resume state. Previous task
(`briefings/probe-adequacy.md`) is REVIEWED & APPROVED and RETIRED —
split-integrity checklist item CLOSED upstream, `PROBE_V2_SPEC.md`
accepted as THE freeze candidate, my forensics receipt independently
reproduced on mac-local's box. Nothing of the mirror campaign has been
run; no card frozen yet.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token`.

## The task: which probe reports TRUE recovery?

Produce the receipt that fires mac-local's **pre-registered 4-branch
decision rule** (LOG ≈ line 2212, "THE λ-READOUT METHODS DECISION:
DEFERRED ~12 h, with the rule PRE-REGISTERED NOW"): 1 v2 tracks truth
where v1 sags ⇒ ADOPT; 2 both track truth equally ⇒ DECLINE; 3 v2
reports ABOVE truth ⇒ REJECT v2 for headline use; 4 ambiguous /
incomplete by Saturday midday ⇒ v1 stays canonical with a stated
caveat. **I produce the receipt, NOT the verdict.** A result arguing
AGAINST v2 is first-class and must be reported as loudly (briefing § 1).
Under every branch the window > token ORDERING survives — no branch
costs the headline, so there is no incentive to any outcome.

## Verified inputs (this session, before compact)

**Substrate + conventions** (`support_synthetic/CARD.md`, reuse
exactly): DS `toy_backtracking_selfexcite_d64`, F = 20, N_STEPS = 30k,
seeds {1, 2, 42}, `eval_window_L` = 32, k_pos = 1 canonical headline
slice, canonical runner, untrained controls = commentary only.
Pattern to copy: `support_synthetic/run_dilution.py`
(`design.uniform_cells(...)` → `grid.run_pool`).

**Known truth on the mirror** (this is what the real panels cannot
supply): λ_i = σ(a + α·Σ_{l=1..K} κ_l b_{i−l}), **K = 2**, τ = 2,
α = 3.06 (bench_spec). So λ is a deterministic function of
(b_{i−1}, b_{i−2}) ⇒ a tile of length T ≥ 3 contains the whole driver.
Stated ceilings: per-token DPI floor **corr ≈ 0.41** (provable), window
ceiling **≈ 0.91 at T = 2, ≈ 0.99 at T ≥ 4**.

**THE DESIGN TRAP — read before writing the card.** v1's `n_windows`
is HARDCODED 1024 (`lambda_recovery_metrics` forwards no nw), so
n_rows = 1024·(32/T): 16384/8192/4096/**2048** at T = 2/4/8/16. At the
mirror's canonical d_sae (20–40) p/n ≈ 0.01 — **three orders of
magnitude away from the real panel's p/n = 1.0 at T16 (p = 2048,
n = 2048)**. A mirror run at canonical budget would show "both probes
agree" trivially and would be misread as branch 2 (DECLINE). The
mirror must be run ACROSS the p/n regime the real panel occupies:
d_sae ladder {20, 64, 256, 1024, 2048} × T {2,4,8,16} spans
p/n = 0.001 … 1.000 (at T16: 0.010/0.031/0.125/0.500/**1.000**).
p/n — not T alone — is the campaign's real x-axis.

**CHECKPOINT REALITY — the briefing's item 2 is mostly unavailable.**
843 mirror leaderboard rows / 843 distinct train_keys, but only **22
checkpoints survive on disk** (`checkpoints/<train_key>/model.safetensors`;
the rest pruned — manifest has 9878 rows, zero HF refs, so no restore
path). Survivors are all July 10–11, k_pos = 2 or 8, d_sae = 20, T ∈
{1,4,8}: batchtopk_sae/tsae T1, txc_pre T4(×3 seeds)+T8, txc_post
T4(×3)+T8, stacked T4(×3)+T8, spectral T4(×3)+T8. So the "cheap
eval-only pass, may answer by breakfast" shrinks to those 22 (still a
real paired sample — ship it with an honest coverage line); **the
campaign is TRAINING-bound, not eval-bound.** Measure per-cell wall
time on the first few cells and size the queue from it (tsae was
~115 s/cell in the support-synthetic campaign; TXC-pre window mode at
30k steps unmeasured on this box).

**Both probes on one row, free.** v2 emits only `*_v2` keys and the v1
block still runs, so every v2-flagged row carries its own paired v1
(nw1024/OLS) column. To separate the two knobs, run 3 v2 eval variants
per checkpoint (nw1024+ridge, nw8192+ols, nw8192+ridge) = 3 rows,
giving the full 2×2 with a shared v1 column — the same 2×2 d/e used.
Pass knobs via `c["eval_extra"] = {...}` (the `grid.run_cell`
pass-through I added; recipe in `PROBE_V2_SPEC.md` § 2). Frozen v2
defaults: RidgeCV `logspace(-2,4,13)` train-half-only, nw 8192,
split `trace` (mirror has no trace_ids ⇒ degenerates to v1's n//2 —
contract-tested).

## Plan of record (proposed; the card freezes it before any cell)

0. **Freeze `support_synthetic/CARD_PROBE_TRUTH.md` FIRST** (commit
   before any cell): substrate/budget, arms, ladder, seeds,
   pre-registered predictions, falsifier, and the explicit mapping of
   outcomes onto mac-local's 4 branches. State up front what pattern
   would argue AGAINST v2.
1. **Constructed-code calibration — do FIRST, costs no training, no
   leaderboard writes** (`probe_capacity.py` precedent: off-leaderboard
   diagnostic). Build codes of dimension p with *exactly known*
   λ-information — true tile event-history dims ⊕ noise dims, sparsified
   to a realistic L0 — plus a **null code (truth = 0)**. Sweep p/n
   through 1.0. Both probes. This answers the branch question in the
   strictest sense: does the reported number track a truth we set?
   Does v2 ever report ABOVE truth (branch 3 optimism check)? Ship as
   an early incremental commit + LOG line.
2. **Eval-only pass on the 22 surviving checkpoints** — paired v1/v2,
   coverage stated honestly (22/843).
3. **Overnight training body** — the d_sae × T ladder × 3 seeds +
   untrained controls, both probes; plus briefing § 3(a) the
   **matched-post arm at nominal k = 8·T** (runpod-d's code-rate
   convention — the confound that qualified the real panel, now
   testable against truth). Sequence so every few hours = one
   committable increment; a partial ladder with an honest coverage
   statement beats a rushed full one (briefing § 3).
4. **`probe_truth.json` + figure** (reported recovery vs T / vs p/n for
   BOTH probes with the TRUE level marked, per arm) + scorecard
   paragraph: which prediction held, which was falsified, what it
   licenses — and if it undercuts adopting v2, say so FIRST.
5. If early: the companion note on a defensible `doc_mean_only_auc`
   doc-identity KILL threshold (proposal only, no bar frozen —
   runpod's overnight scale-up is producing the doc-level bootstrap
   CIs it would rest on).

**Acceptance gate**: card frozen pre-run; incremental commits + LOG
lines; canonical runner + leaderboard hygiene (row decomposition
stated); figure + receipt + scorecard; STATUS rewritten. Stop for
review; briefing stays.

## Standing context
- Shared branch: pull-rebase before EVERY push; LOG.md conflicts keep
  the upstream entry then re-append mine; commit SUBJECTS not SHAs;
  scripts/cards committed BEFORE outputs; no reviewer/meeting quotes in
  tracked files; all numbers script-derived.
- Reproduction claims: say "bit-identical **on the build platform**" —
  mac-local saw 16th-digit x86↔ARM drift in `r_between_arms`.
- Environmental pytest trap: untracked files break
  `test_diff_hash_consistent_with_dirty` — commit/clean before full
  suite runs. Suite was 299–302 passed / 1 skipped depending on
  upstream state.
- Upstream: screen wave running (runpod-e: novelty NEGATIVE, punctint-q
  KEEP, punctint-list WEAK KEEP, interleave/tss KILL — all UNREVIEWED);
  runpod on the overnight corpus scale-up; runpod-d owes a record
  amendment (4/12 matched cells above the [5.0,8.0] band mislabelled
  in-band; verdict unaffected). GPU pods re-run nothing for the methods
  decision until my receipt fires the rule.
- Rewrite this file before any compact.
