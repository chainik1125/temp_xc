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

## Preconditions (the user satisfies these BEFORE launch — the agent never asks mid-run)

This is a hands-off run; it does not pause for input. At Stage 0 verify all of
these and, if any is missing, **STOP with a clear report** rather than improvise:

- **Claude API key** at `/workspace/.tokens/anthropic_key` or `ANTHROPIC_API_KEY`
  in the env. Never hardcode. Absent ⇒ stop.
- **Judge models reachable** (probe each once): bulk `claude-haiku-4-5-20251001`,
  validation/adjudication `claude-sonnet-5`, hypotheses + skeptic
  `claude-opus-4-8`. If a string is stale, substitute the nearest available tier
  and record which.
- **Cost cap = $25 / cycle** (default; user may override). Meter spend; hard-stop
  at the cap and report partials — never exceed it.
- **Text-corpus source**: a fineweb (or similar) sample streamable via `datasets`
  (cached to the volume). The reasoning-trace domain needs **no model** in Cycle 1
  (Stage 0).

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
5. **Adversarial skeptic pass on every PROCEED (structured rubric, not vibes).**
   Before a spec is frozen, a *separate* Opus call fills a fixed kill-rubric, each
   item yes/no + evidence: (a) is the ordered−shuffled gap within the labeler
   noise floor? (b) could the labeler be leaking the target? (c) is this
   composition/marginal, not order (the topic_switching trap)? (d) does the mirror
   match the statistic by construction (circular)? (e) is the effect an artifact
   of window/segmentation choice? A PROCEED survives ONLY if every item clears;
   store the filled rubric in the calibration record.
6. **Cost cap — $25 / cycle (default).** Bulk labeling on Haiku
   (`claude-haiku-4-5-20251001`); Sonnet (`claude-sonnet-5`) only for
   validation/adjudication; Opus (`claude-opus-4-8`) only for hypotheses + the
   skeptic. Meter spend; hard-stop at the cap and report partial results.

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

**Calibration is text-only.** It measures the temporal signature of the *label
stream* (autocorrelation, dwell, MI vs lag, nulls) over text — it needs **no
activations, no local model, and no architecture.** `real_lm.py` (the
activation-cache builder) belongs to the *later, blind* eval stage, NOT this loop.
Do not build activation infra here.

- **Claude API client:** small wrapper over the anthropic SDK with per-model
  routing (Haiku/Sonnet/Opus) + a spend meter enforcing the $25 cap.
- **Reasoning-trace domain — reuse, don't generate.** Cycle 1 runs on the **300
  stored traces** at `results/c7_backtracking/stage_a/sentence_labels.json` —
  already sentence-segmented text (`sentences[].sentence`, with char offsets).
  Re-label those sentences with a Claude judge for each NEW candidate property.
  **No local model, no generation, no `vllm` in Cycle 1** — generating fresh
  traces on the A40 is a *later scaling* step (more/diverse data), not a Cycle-1
  prerequisite.
- **Text-corpus domain — the real Cycle-1 build.** The stored data is
  reasoning-trace only, so this half is built from scratch: a fineweb (or similar)
  sample streamed via `datasets` (cached to the volume) + a Claude-judge labeler
  for it. This is the heavier half of Cycle 1 — budget for it.
- **The factory harness:** generalize `backtracking/measure.py` — its toolkit is
  already reusable: `acf`, `fano`, `mi_vs_lag`, `self_excitation`,
  `inter_event_cv`, `markov_order_test`, and the null generators `null_permute`
  (=N1) / `null_homog` (=N3); add an N2 trend-preserving null. Parameterize the
  **label field** (currently `is_backtracking`) → any per-span signal. Plus a
  labeler-runner (Claude judge → per-span signal + noise-floor validation) and the
  mirror menu (Appendix B) fit+validate (from `mirror.py`). Never edit
  `temp_bench/core/`.

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
   the Claude-judge labeler (noise floor) → measure the signature on the **labeled
   text stream** (stored traces / corpus, held-out) → **run the null battery** →
   temporal-ness verdict.
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
version-pin the corpus snapshot + the judge model in every record (as backtracking
pinned its Sonnet judge). **Calibration is text-only** — no activations, no
`real_lm.py`, no architectures (those are the later blind eval). Prime directive:
a sound verdict, never a win. When this task's *first* cycle is done and reviewed,
this briefing is superseded by the standing `expansion/README.md` — delete it then.
