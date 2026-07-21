# Working state — agent `mac-local`

**Last rewrite:** 2026-07-21 (Opus 4.8, local CC — pre-compact). Supersedes the
stale 2026-07-10 version.

## Who / where
Local CC on the Mac (Apple M5 Pro, MPS, no CUDA) at `~/research/projects/temp_xc`.
Role: **prototyping, review, orchestration**. Heavy grids go to `runpod`. I'm the
`mac-local` agent (inferred from darwin + this path).

## Git
Branch `arxiv`, **level with `origin/arxiv` @ `e64d7e39`**, clean tree.
`origin` = SSH (`git@github.com:chainik1125/temp_xc.git`).

## The program has two workstreams (full science in the research STATUS §0)

**A. Architecture B×A comparison — DONE + mature.** The full clean-room rerun
completed (protocol **1.3.0**, ~2239 cells, 0 failures, run on CPU). `REPORT.md`
is filled: per-token-matched matrix (6 latent-axis rows × 6 fair-backbone archs,
`{F,F/2}`) + NMSE/eauc companion panels + 3 figures (heatmap, capacity frontiers,
capability gate). The organizing principle — **where the nonlinearity sits**
(additive-over-position vs position-mixing) determines which latent an arch
exposes — is established. Covers backtracking/changepoint/frequency/signed_motion.

**B. Grounded-benchmark expansion — 3 autonomous cycles done + reviewed.** An
autonomous, gated measure→mirror loop (runpod) that discovers grounded benchmarks
from real reasoning traces + text. Standing infra: `expansion/README.md` (pipeline
+ 8 guardrails), the `src/explorations/synthetic/expansion/` harness, gates 7
(no-leakage labeler) + 8 (non-fitted-moment mirror), and the `expansion/LEDGER.md`
coverage grid. **Grounded SPECs produced:** `assumption_consequence` (AC/directed,
solid), `hedging_drift` (DC/slow, solid — `hier_ar1` mirror), `list_item_parallelism`
(bursty, redundant + weak mirror — low value), `self_reference_echo` (SPEC*, low).
backtracking = the hand-run anchor. **interaction/equality prize is UNCLAIMED** —
the categorical recipe *measures* real signals but no mirror in the menu holds the
categorical plateau (needs a hierarchical-categorical mirror).

## ⏳ In flight — STAGE 6 queued (the immediate next thing)

`briefings/stage6-grounded-eval.md` (runpod, active): build + **blind-eval** the
two solid grounded SPECs (`assumption_consequence` + `hedging_drift`) into the
framework as `✓`-registered benchmarks, run the fair-backbone grid, extend
`REPORT.md`, flip `BENCHMARKS.md` rows `✗→✓`. **Not mine to run** (heavy grid).
This closes the measure→mirror→**bench** loop for the first time. Predictions are
frozen in each `bench_spec.md` — the run must be blind (no tuning to match).

**When runpod finishes → review it** (my standing protocol, below). Specifically:
verify no prediction was tuned for, the §8 gate passed, the grid ran clean (0
failures, code-version stamped), and `REPORT.md`/`BENCHMARKS.md` regenerated
honestly.

## How I review runpod cycles (the protocol — I've caught real issues with it)
Pull → **verify integrity** (no arch pollution to `leaderboard.jsonl`/`manifest`;
spend real + itemized; prereg frozen *before* data via commit order; harness tests
pass; scoped commits / no stray logs) → **verify the science** (are ABORTs genuine
sophisticated skeptic kills, not giving up? are PROCEEDs sound — check for
reward-hacks, misclassification, over-sell?) → **apply corrections + queue next**.
Caught so far: self-reference-echo misclassification (self-exciting, not int/eq),
list-item over-sell (redundant + weak mirror), and confirmed the C3 list-item
relative-tolerance re-freeze was legit (preregistered, not a gate-loosen hack).

## 📌 Pending / recorded for later
- **Cycle 4 (expansion) needs its own briefing** when expansion resumes — targets
  recorded in `expansion/LEDGER.md`: the **interaction/equality prize** (build the
  hierarchical-categorical mirror), + **harden gate 8** (check ≥2 non-fitted
  moments; hybrid mirrors must preregister a non-fitted moment — the C3 circularity
  lesson). Decision was: **stage 6 first, then Cycle 4.**
- The old frequency "DCT-band decisive" calibration-debt note is likely moot (the
  rerun regenerated the record). Low priority; verify if revisiting frequency.

## Trackers (where to look — don't re-derive)
- **`experiments/explorations/synthetic/BENCHMARKS.md`** — the one-stop registry
  (every benchmark: spec-status / framework-registered / arch-verdict + the
  tried-&-set-aside record). Hand-maintained; I created it 2026-07-21.
- **`expansion/LEDGER.md`** — grounded coverage grid (proposed/abort/SPEC per cell).
- **`REPORT.md`** — the head-to-head (auto-gen, evaluated benches only).
- **research `STATUS.md` §0** — the living program state (read first, over this file).

## Recent commit chain (this stretch)
rerun+purge (`117afaf1`) → panels/figures → pollution-fix (`a2ebd6b6`) → C2 review
gates + cycle-3 briefing → C3 review corrections (`b67e860d`) → `BENCHMARKS.md`
registry (`138d345b`) → stage-6 briefing + retire C3 briefing (`e64d7e39`).
