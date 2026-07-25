---
status: queued
created: 2026-07-25
for: runpod-b
venue: runpod (32C CPU)
---

# QUEUED — panel support + rebuttal audit (start ONLY after mirror-probe-truth closes)

**You are `runpod-b`. DO NOT START THIS until
`briefings/mirror-probe-truth.md` reaches its acceptance gate and is
pushed** (Stage-3 grid + mix arms + Test B + final receipt/figure/
scorecard). That campaign stays your priority; this is the queue behind
it so you never idle. If the campaign is still running, keep running it.

Your increment-2 receipt is read and its ADOPT-consistent finding is
noted; **the methods decision stays mine and is not yet taken** — it
waits on your mix arms and the trained ladder. Nothing below assumes
either outcome.

Both H100s are committed overnight to Stage-2 panels
(`stage2-oprate.md` = runpod-d, `stage2-fineweb.md` = runpod-e), and
`runpod` is on the round-3 factory. Every item here is CPU-only and
serves those panels or the rebuttal. **Ship incrementally, one LOG line
each — items 1 and 2 are time-critical to the panels; 3 and 4 are not.**

## 1. FIRST, and time-critical: pre-flight the variance harness against BOTH new panels

Both briefings instruct d and e to produce variance receipts with
`support_stats/stage2_variance.py` — the harness you made
probe-agnostic. **There is a specific, predictable way it breaks on
their data, and you are the one who found it:** you fixed a loader that
keyed rows on (arch, T, seed, kind) with no `k_pos` filter and aborted
on 24 duplicate cells when the post-matched `k_pos = 8·T` rows landed.
**Both new panels run TXC-post at k = 8·T *from the start*** — so
`k_pos` varies within a single panel by construction, on two new
datasources, in one night, with no one downstream positioned to debug
it at 3am.

Do this without their results (you do not need them):
- Construct the row populations both panels will emit — new
  datasources (`oprate` case on Ward; `punctint` q on fineweb/gemma
  primary + gpt2/llama replication cells), 5 archs, T {2,4,8,16},
  seeds {1,2,42}, trained + untrained, post at `k_pos = 8·T`, and rows
  carrying BOTH v1 and v2 columns — as synthetic fixtures.
- Run the harness over them. It must select the right population, not
  abort, not silently drop the post arm, and not mix v1/v2 columns.
- **Fix whatever breaks, with tests**, and make the invocation explicit:
  commit the exact command line each pod should run for its panel
  (`--ds`, `--probe`, `--k-pos`, `--out-prefix`) into a short
  `support_stats/PANEL_RECIPES.md`, and post ONE LOG line pointing d
  and e at it. If it already works unchanged, say so in one line and
  move on — a five-minute negative is a good outcome here.
- Replication-cell wrinkle for e's panel: gpt2/llama cells exist at
  only two T values, so any trend statistic over the T ladder is
  undefined there. Make the harness degrade honestly (report the
  cells, skip the trend, say why) rather than emitting a trend from
  two points.

## 2. `PROBE_V2_SPEC.md` — carry your own caveat

Your increment-2 entry states it directly: v2 is biased LOW by up to
0.18 exactly where the real panels live (low truth + dense code +
p/n ≥ 1), so adopting v2 **tightens a lower bound; it does not make
reported recovery an estimate of truth** — and the spec does not
currently say this. Add it to the spec as a first-class limitation
(not a footnote), including the numbers and the arms they came from,
so that whatever I decide, the adopted document cannot be read as
claiming more than your measurement supports. This is the single most
important sentence in that file.

## 3. The claim→artifact receipt index (rebuttal insurance)

Build `experiments/explorations/task_hunt/RECEIPTS.md`: for every
number the program currently considers rebuttal-quotable, one row
giving the claim as we would state it, the exact artifact path + JSON
key it comes from, the commit that produced it, and a
**recomputed-now** value with PASS/FAIL against the quoted one. Seed
list (extend it — finding a quotable claim that is NOT on this list is
itself a deliverable): the λ̂ Stage-2 panel cells + the exact
permutation p = 0.0093; the trained−untrained margin p = 0.0046; the
pre/T8 n = 6 CI [0.179, 0.235]; **the pre-vs-T-SAE margin's NOT-bounded
status** (paired LB −0.041, Welch LB −0.016, p = 0.082 — this one must
never be quoted as significant); the shuffle/anticipation receipt; the
T-SAE fairness receipt (max |paired D| 0.011 vs the 0.05 bar); your
split-forensics zero-leakage result; the five Stage-1 KEEP screen
numbers **with their training-corpus size attached** (runpod's
estimator finding); the amended order finding's g_order band AND
dialevel's counterexample. Flag every mismatch loudly; a FAIL here two
days before a deadline is worth more than any new result.

## 4. If the night is still long: the analysis is pre-staged, not improvised

Write (do not run) the analysis each new panel will need the moment it
lands: the exact commands, the expected row decomposition, and a
skeleton scorecard. Aim for "d or e pushes results → one command
produces the receipts" instead of anyone writing analysis code while
tired against a deadline.

## Acceptance gate — stop for review

Items 1 and 2 complete (or an honest one-line negative for 1); LOG line
per item; STATUS rewritten. Items 3–4 as far as the night allows —
partial with a coverage statement beats rushed. Briefing stays until
mac-local review.
