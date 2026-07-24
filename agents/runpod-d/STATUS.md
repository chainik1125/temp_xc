# Working state — agent `runpod-d`

**Last rewrite:** 2026-07-24 (session 1, mid-flight — GPU chains running).

## Who / where
GPU RunPod pod (H100 80 GB, 224 cores, 2 TB RAM), Linux at
`/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-d`. Role:
**task-hunt arm A** (`briefings/task-hunt.md`). Branch `arxiv`, shared
with 4 other agents — **always `git pull --rebase` before pushing**
(credentials: `GIT_ASKPASS=/workspace/.tokens/git-askpass.sh`, already
set as `core.askPass`; identity configured).

## Environment (built this session — reuse, don't rebuild)
- `.venv` (torch 2.8+cu128) = the probe/training venv;
  **`/workspace/vllm_venv`** = separate vLLM 0.25.1 venv (candidate 3).
- `HF_HOME=/workspace/hf_cache`; tokens at `/workspace/.tokens/`.
- **Caches on the volume** (~1 h to rebuild if lost):
  `/workspace/conv_depth_caches/{ward_stream,base,distill}` (17 capture
  points each, fp16), `/workspace/task_hunt_labels/lambda_intensity/`
  (λ̂ grids incl. the dense Stage-2 variant),
  `/workspace/task_hunt_labels/forbidden_word/cotcontrol_data/`.
- `results/c7_backtracking/stage_a/traces.json` was missing from the
  branch and re-ported from `origin/aniket-ward-stage-b` per that
  directory's own ATTRIBUTION recipe. It stays untracked.

## Done (committed + pushed)
1. **Substrate rebuilt** and verified — reproduces
   `conversion_depth/RECORD.md` § 3 exactly (base L10 ant_kw
   0.843/0.886; distill L10 0.844/0.895).
2. **Candidate 1 (backtracking λ̂): screened, verdict KEEP (qualified)**
   — verdict paragraph in `task_hunt/LOG.md`, methods in
   `task_hunt/RECORD.md` § 1. T-scaling is real (g rises to +0.054 at
   T=32, no saturation, all 4 model×layer cells); **the order story is
   negative** (g_order ≤ 0 in 17/20 cells, shuffle costs ≤ 0.022).
   Three frozen predictions falsified and scored; one renderer
   rule-scoring correction disclosed in the LOG.
3. **Cards frozen pre-run** for candidates 2 and 3; Stage-2 plumbing
   (plugin datasource + two `configs/data.yaml` entries) committed.
4. **Leaderboard hygiene checked**: 7121 rows, 0 dup `eval_key`s.

## SESSION COMPLETE (2026-07-24) — all four candidates screened; acceptance gate MET; awaiting mac-local review

Every gate item done and pushed: hunt LOG with every screen verdict;
**≥ 1 survivor through Stage 2 with the T-scaling figure** (candidate 1,
`figs/stage2_tscaling.*`); STATUS rewritten (this file); leaderboard
hygiene (0 dup keys, 0 null-metric rows; 220 tests pass). No
reviewer/meeting quotes in tracked files. Briefing stays until
mac-local review. **em-redo PAUSED note:** that arm is runpod-c's, not
arm A's — not touched this session.

**Verdicts (all in `task_hunt/LOG.md`; methods in `task_hunt/RECORD.md`):**
- **Cand 1 (λ̂ intensity): KEEP → Stage 2 QUALIFIED POSITIVE.** At
  matched realized l0, TXC-pre beats the per-token SAE (0.113) and
  T-SAE (0.154), rising 0.13→0.19→0.21 across T=2/4/8 (peaks, then dips
  at T=16). TXC-post reaches 0.255 at T=16 but its l0 collapses to 0.49
  — budget-confounded, flagged not headlined. Stacked shows a
  training pathology at T=16. Order story negative (regime 2).
- **Cand 2 (proof-op runs): KEEP** — the MODEL AXIS is the finding.
  Contrast g_tir−g_op clears the null at every T on **distill L12**
  only; base ≈ distill (P5) decisively falsified. Mirror image of
  cand 1. Caveat: a shuffle gap that grows with T is generic to wider
  windows (the ambient anchor grows identically) — only the
  anchor-differenced contrast is an order claim.
- **Cand 3 (forbidden-word onset, SILOED): KILL** — the card's
  pre-registered ambience kill fires (per-token within 0.02 of window
  at every horizon; window never beats per-token beyond 3σ). 97.4 %
  violation rate (gate passed easily). **A post-hoc 17-layer depth
  sweep (added after the verdict; does NOT reopen it) shows the gap is
  shut at EVERY depth — 2/51 cells above 3σ, max +0.037 — so the kill
  is not an artifact of the frozen layer choice.** It also *corrected
  the mechanism*: my kill entry blamed a bag-of-words "circling" leak,
  but at the embedding layer per-token reads only 0.538 (near-blind)
  vs backtracking's +0.174 lexical gap there. Per-token instead CLIMBS
  0.538 → 0.668 with depth — the model **computes** the pressure from
  context and linearizes it per-position. **Conversion, not lexical
  ambience.** Blind predictions scored D3 ✓ / D1, D2 ✗. This is a
  FOURTH g(ℓ) shape for the atlas
  (built-and-immediately-linearized) — see the LOG entry's comparison
  table. **Design rule it yields:** screen for "will the model decline
  to maintain this as a per-position state?" (backtracking anticipation
  = a hazard over the trajectory, never converted) — NOT "is the
  concept semantically non-obvious". Predicts a logical-deduction
  variant ("don't mention XOR") fares no better and plausibly worse.
- **Shuffle receipt (existing case study): POSITIVE** — backtracking
  ANTICIPATION is order-sensitive (shuffle costs +0.028…+0.041 vs
  +0.003…+0.013 for the ambient is_bt), at fixed T=16 on identical
  rows (no T-confound).

**Net:** the hunt's strong target (window recovery rising vs T-SAE flat,
order matters) is only partially realized — a bounded matched-budget
window advantage on cand 1 (order-free), a model-specific window signal
on cand 2, a clean order receipt on the *existing* paper task, and a
principled ambient KILL on cand 3. No unqualified new positive; the
sound verdicts are the deliverable.

## If resuming: the follow-ups worth a next session (none started)
- Cand 2 Stage 2 on **distill L12** (its best cell) — separate from the
  cand-1 panel already run; the datasource plugin generalizes (swap
  labels).
- Cand 1 TXC-post efficiency: 0.255 recovery at l0=0.49 is striking on
  its own terms; a budget-matched re-run (raise post's k) would settle
  whether it's a real win or the starvation artifact.

## Two traps this session hit — read before touching the chain or the leaderboard

1. **Never emit NaN into a leaderboard metric.** The Stage-2 datasource
   first left `emission_features` empty (a real residual stream has no
   ground-truth directions), so `eauc`/`e_*` came back NaN. **The
   leaderboard IS the eval cache**; JSON stores NaN as `null`, and
   `LeaderboardRow` then rejects the cached read — six such rows made
   the canonical artifact **unloadable for every subsequent run**,
   surfacing as a `ValidationError` on an unrelated cell long after the
   rows were written. Fixed by giving `emission_features` a documented
   **reference basis** (DC + top PCs, seed 0 — a sanity check, NOT
   feature recovery). The six rows were removed (backup
   `/workspace/logs/leaderboard.backup.jsonl`); leaderboard is back to
   **7116 rows, 0 dup keys, 0 null metrics**.
2. **Never write a wait loop as `pgrep -f "<pattern>"`.** A monitoring
   shell whose own command line contains that pattern makes `pgrep`
   match itself, so the loop never exits — this silently deadlocked the
   first chain for ~30 min. The current `chain.sh` runs its stages
   straight through instead.

## RUNNING RIGHT NOW (ONE detached chain — do not fan out)
**GPU serialization is mandatory** — running two raw-activation screens
plus the Stage-2 pool concurrently caused CUDA OOM in both (a T=64
flatten probe needs ≈28 GB for standardization alone). Hence:

The `chain.sh` chain finished stages 1–2 (proof-op screen exit=0,
Stage 2 exit=0). Stages 3–4 (candidate 3) exited 1 in the chain because
generation needs the **vLLM venv**, not `.venv` — the chain ran it under
`.venv` (no vllm/pandas). Candidate 3 is now running standalone:

- **generation** — `/workspace/vllm_venv/bin/python -m
  experiments.explorations.task_hunt.forbidden_word.generate`
  (log `/workspace/logs/fw_generate.log`; needs `HF_HOME`, `HF_TOKEN`;
  I `uv pip install pandas` into the vllm venv). Writes
  `/workspace/task_hunt_labels/forbidden_word/rollouts.jsonl`.
- **cache + screen** — `.venv/bin/python -m
  experiments.explorations.task_hunt.forbidden_word.cache_and_screen`
  (log `fw_screen.log`) — runs in `.venv` (torch probes), forces
  `PreTrainedTokenizerFast` (the tokenizer trap, note below).

**Check the feasibility gate FIRST** (card § "Feasibility gate"): if
`forbidden_word_gen_stats.json` violation_rate < 30 % or < 200
violating rollouts, the screen is under-powered — record and either
add one more rollout/question (disclosed) or KILL as infeasible. Only
then screen and write the verdict.

## Next actions (in order)
1. When the proof-op screen finishes: score it against
   `proofops/card.md` and write the KEEP/KILL paragraph into
   `LOG.md` + the ladder table into `RECORD.md` § 2.
   **`tir` ladder at base/L12 is COMPLETE** (per-token 0.614):

   | T | flatten | mean | shuffled | g | g_order | shuffle gap |
   |---|---|---|---|---|---|---|
   | 8 | 0.642 | 0.642 | 0.634 | +0.028 | +0.000 | +0.008 |
   | 16 | 0.663 | 0.652 | 0.638 | **+0.049** | +0.011 | +0.025 |
   | 32 | 0.646 | 0.647 | 0.613 | +0.032 | −0.001 | +0.033 |
   | 64 | 0.651 | 0.629 | 0.590 | +0.037 | **+0.022** | **+0.061** |

   Two things matter here. (a) g peaks at **T=16** then declines —
   the card's P1 (nothing below T=32, then growth) is **falsified**;
   the shape is peak-then-decline (localized latent, STORY § 7).
   (b) The shuffle gap grows monotonically with T (+0.008 → +0.061)
   — **but this is NOT order evidence**: the ambient `op` anchor grows
   the same way (+0.010 → +0.065), so a growing shuffle gap is a
   generic property of wider windows under this probe. Corrected in
   the LOG; see the verdict entry.

   **Verdict written (primary layer): WEAK KEEP.** The card's real
   claim, the contrast g_tir − g_op, is −0.009 / +0.008 / −0.005 /
   **+0.019** at T = 8/16/32/64 — clears 3σ_null (0.0105) at exactly
   one T, non-monotone. No kill rule fires as written, but one-point
   survival is not the predicted ladder. P1 and P3 falsified.
   Confirmatory cells (base L10, distill L10/L12) were still running;
   **extend the LOG entry when they land — do not revise it.** Re-run
   `proofops/render.py` for the updated verdict JSON + figure.
2. ~~Shuffle receipt~~ **DONE** — 12/12 cells, verdict POSITIVE in
   `LOG.md`, full table in `RECORD.md` § 3, figure
   `task_hunt/figs/shuffle_receipt.*`.
3. When Stage 2 finishes: build **the T-scaling figure** (recovery vs
   T, one line per arch — the money plot) and write the record. Only
   `buffer_tokens=524288` rows are headline (see RECORD § 4.5).
4. Candidate 3 verdict after chain 2; check its feasibility gate
   (violation rate ≥ 30 %) BEFORE reading the screen.
5. Update `experiments/explorations/synthetic/STATUS.md` with the arm-A
   outcome; leave `briefings/task-hunt.md` in place until mac-local
   reviews.

## Deadline
Results wanted by **2026-07-26 morning PT**. Ample wall-clock remains;
the constraint is GPU serialization, not the calendar.
