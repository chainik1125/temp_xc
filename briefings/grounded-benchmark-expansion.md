---
status: active
created: 2026-07-13
for: runpod
venue: runpod
---

# Autonomous grounded-benchmark expansion — gated batch, two-domain balanced

**Goal.** Autonomously *expand the list of grounded benchmarks* by running the
proven measure→mirror loop (the one `backtracking/` already ran, by hand) as an
autonomous, gated pipeline: **hypothesize → select → calibrate on real LM data →
freeze**. One cycle produces frozen, architecture-blind benchmark specs (or clean
aborts) across BOTH data domains, then **STOPS for human review**. It does **not**
evaluate architectures — that is a separate, later, deliberately-blind step.

You are generalizing a *human-validated* loop, not inventing one. Read
`experiments/explorations/synthetic/backtracking/` end-to-end first —
`prereg.md` → `measure.py` → `kernel_order.py` → `mirror.py` → `measurement.md`.
Its labeler was a **Claude judge** (Sonnet, `is_backtracking` over 25,528
sentences, noise floor measured); its gate was the **N1/N2/N3 null battery**
(within-trace permute / trend-only Poisson / homogeneous). Reuse that machinery.

## ⚠️ Prime-directive guardrails (an autonomous generator WILL drift to positives)

These are load-bearing. A cycle that violates them is invalid regardless of output.

1. **An ABORT is a success.** topic_switching aborting was a *good* outcome. Never
   tune the labeler, statistic, window, or shuffle to force a temporal verdict.
2. **Blind to architectures.** No arch is trained or evaluated during a cycle.
   Selection and calibration never see an architecture score.
3. **The null/shuffle gate is make-or-break.** A property is "temporal" only if
   the ordered statistic beats the order-destroying null beyond BOTH sampling
   noise AND the labeler noise floor. If not → ABORT.
4. **Validate every labeler.** Claude-as-judge is only usable with a measured
   noise floor (inter-judge agreement on a held-out sample + an independent
   check, as backtracking did with keyword-vs-judge F1). Unvalidated labeler ⇒ no
   measurement.
5. **Adversarial skeptic pass on every PROCEED.** Before a spec is frozen, a
   *separate* Claude call (Opus) tries to kill the verdict — confound? labeler
   leakage? shuffle too weak? composition not order (the topic_switching trap)?
   A PROCEED survives only if the skeptic can't refute it. Record the attempt.
6. **Cost cap.** Bulk labeling on Haiku; Sonnet only for validation/adjudication;
   Opus only for hypothesis generation + the skeptic. Hard token/$ ceiling per
   cycle (set it, log spend, stop at the ceiling).

## The anti-drift mechanism (the reason "both domains" is safe)

Coverage is a **structural invariant**, not a hope. Maintain
`experiments/explorations/synthetic/expansion/LEDGER.md` — a grid of
**domain × temporal-class**:

- domains: `reasoning-trace`, `text-corpus`;
- temporal-classes: `DC-slow-drift`, `AC-order-sensitive`, `periodic`,
  `bursty/self-exciting`, `interaction/equality`, `long-memory`.

Each cell records: candidates proposed, calibrated, verdict (PROCEED/ABORT),
frozen spec. Two rules make drift impossible:
- **Per-domain floor:** each cycle calibrates **≥⌊N/2⌋ candidates from each
  domain** (never all from one).
- **Under-coverage bias:** selection prioritizes empty / abort-only cells over
  cells that already have a PROCEED.

Report the filled/empty grid at the end of every cycle so balance is visible.

## Stage 0 — bootstrap (Cycle 1 only)

- **Claude API:** anthropic SDK; key from `/workspace/.tokens/` (add if absent —
  ask the user, do not hardcode). Wrap in a small client with per-model routing
  (Haiku/Sonnet/Opus) + a spend meter that enforces the cap.
- **Reasoning traces:** a local instruct/reasoning model on the A40 (R1-Distill-
  Llama-8B or similar, bf16 ≈16GB — fits) to generate CoT traces on math/logic
  prompts; **reuse the stored traces** at `results/c7_backtracking/stage_a/` +
  generate more for diversity. Activations via `src/temp_bench/data/real_lm.py`.
- **Text corpora:** a fineweb (or similar) sample on the 200GB volume.
- **The factory harness:** generalize `backtracking/measure.py` + `mirror.py`
  into a reusable `expansion/` library — a labeler-runner (Claude judge → per-
  span signal + validation), the null battery (N1/N2/N3, generalized), the
  signature toolkit (ACF, dwell/run-length, MI vs lag, Fano/burstiness, spectral
  DC/AC share), and the mirror menu (Appendix B) with fit+validate. Everything
  routes real-LM reads through `real_lm.py`; never edit `temp_bench/core/`.

## Stages 1–4 — the cycle (every cycle)

1. **Hypothesize** (Opus) → frozen *prereg cards* (template:
   `expansion/prereg_template.md`), balanced across the ledger's under-covered
   cells, both domains. Each card: property, hypothesized temporal-class, the
   per-span labeler (must be Claude-judgeable or corpus-derivable), the statistic
   + its order-destroying null, chance/oracle, per-arch predictions-with-reasons.
   **Freeze (commit) before any data is touched.**
2. **Select** (blind) → score by *labelability × novelty × predicted-temporalness*,
   apply the per-domain floor + under-coverage bias, take the top-N. Cycle 1:
   **N = 4 (2 per domain)** — prove the loop end-to-end, don't scale yet.
3. **Calibrate** (per candidate, the `measure.py` template): build + **validate**
   the Claude-judge labeler (noise floor) → measure the signature on real
   traces/corpus (held-out) → **run the null battery** → temporal-ness verdict.
   If PROCEED: fit a mirror (Appendix B, keyed to the matched statistic) +
   validate it reproduces the statistic on held-out draws. Run the **skeptic
   pass**. Write a `measurement.md`-style calibration record either way.
4. **Freeze + ledger + STOP.** For each PROCEED: a frozen `bench_spec.md` +
   fitted-mirror params, staged for the (later, blind) B×A eval — graduate it to
   its own `experiments/explorations/synthetic/<name>/` subdir. Update
   `LEDGER.md`. Commit + push. **Do not start the next cycle; do not run any
   architecture.** Report: the coverage grid, the per-candidate verdicts, spend,
   and what the next cycle should target.

## Acceptance gate (per cycle)

- Balanced: ≥⌊N/2⌋ calibrated per domain; the ledger updated + the grid reported.
- Every candidate has a prereg **frozen before measurement**, a calibration
  record with the null-gate verdict + the labeler noise floor, and (PROCEED only)
  a validated mirror + a survived skeptic pass + a frozen `bench_spec`.
- No architecture was trained/evaluated. Spend ≤ the cap (logged).
- Committed + pushed to `origin/arxiv`. Cycle STOPS for review.

## Constraints (hard rules)

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**;
real-LM reads through `real_lm.py`; version-pin corpora + the judge model + the
target LM in every record (as backtracking pins Sonnet + R1-Distill). Prime
directive: a sound verdict, never a win. When this task's *first* cycle is done
and reviewed, this briefing is superseded by the standing
`expansion/README.md` — delete it then.
