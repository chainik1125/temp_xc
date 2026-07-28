# mac-local — STATUS · SNAPSHOT #5 (2026-07-28 ~22:0x BST)

**Supersedes SNAPSHOT #4 (16:1x).** Rewritten because the day turned:
a *delivered* exhibit is under challenge, and the challenge spread to
three sections. I am the hub — review, ratify, rulings, the binding
LOG, ledger oversight, and the handover surfaces
(`REBUTTAL_HANDOFF.md`, `REBUTTAL_CODE_GUIDE.md`,
`REBUTTAL_CELL_CENSUS.md`). **No compute of my own.**

---

# RESUME HERE

## The one paragraph

**ITEM 6 CLOSED — challenge answered, claim RESTORED but NARROWED.**
Dmitry's agent showed TXC losing to pooled/stacked SAE baselines we had
never run; the proper comparison now exists as a **recovery-vs-budget
frontier** (not a matched point, which does not exist). **vs pooled:
TXC above 2/4 T (T=8, T=16), INDISTINGUISHABLE 2/4 (T=2, T=4), never
below** — and pooled **saturates**, never reaching TXC at up to
**4.06× its budget**, so it is not a cutoff artifact. Stacked's 4/4 is
**refused** as probe-capacity overfitting. Deliverable:
`figs_writeup/tab_sycgen_budget_matched.md`. **Fleet idle, agent spend
$0/h. All four mac-c hunt lanes un-parked behind lever 3.**

> **⚑ THIS SAID "above 3/4" UNTIL 00:41 07-29 AND THAT WAS BIASED
> (`73f8ea388`).** The generator compared TXC against pooled's best
> point at `l0 ≤ TXC's l0`; on a coarse k grid that meant **TXC @ 5.66
> vs pooled @ 3.51 at T=2 — 38% less budget — scored as a win**, while
> the point 5% *above* TXC's budget was indistinguishable. Table now
> brackets + interpolates (rules A/B/C printed). **T=16 is the strong
> cell** (Pareto: pooled cannot operate that cheaply at all). **Do not
> re-quote 3/4 from any older surface.** Standing check: print the
> comparator's **budget ratio** — if it is not ≈1.0, "matched" is not
> earned.

## Fleet

| agent | state | lane |
|---|---|---|
| **mac-local** (me) | hub | rulings, LOG, handoff surfaces |
| **mac-d** | **0 pods, $0/h** | ⚑ **TOP: `briefings/sycgen-shuffle-sparsity-matched.md`** — issued 00:34, **NOT YET CLAIMED.** Up to 20×H100 authorized (~$60/h; authorized ≠ free, pre-spend estimate first) |
| **mac-c** | **0 pods, $0** | lever-3 rescue (`a027b7caa`, P1/P2/P3 held, P5 fired) + delivered the shuffle-brief pre-reg audit |

### ⚑ THE ONE LANE IN FLIGHT — sycgen shuffle ablation, sparsity-matched

Han (00:3x): *"we NEED sycgen proper shuffle ablation with matched
sparsity."* Brief issued; **mac-c's pre-registration audit found 4
defects + a stamp error, all ADOPTED by hub ruling `29bc6a95d`.** The
lane is **cleared to run but unclaimed** — fleet idle at $0/h, so
nothing burns while it waits.

**Before any cell runs, in this order:** (1) **A1** — the one-line
input-side `assert (tiles_sh - tiles_ev).abs().max() > 0`; the
pooled-zero gate tests *pooled*, not the shuffle, and a no-op shuffle
would pass it and read as (b), the outcome the brief pre-commits to
publishing. (2) **A2/A3** decision rules into the card: twin is a
**gate on (a)**; (a) needs TXC>stacked in **all 3 seeds** AND mean
margin > across-seed SD. (3) **A4** redraw from the `T!−1`
non-identity permutations. (4) **§2b** bracket both sides of TXC's
budget — the same coarse-grid trap that produced the 3/4 error.

**⚑ 3 unattributed non-convention pods ($3.87/h)** — `mats-gap-code-h100`,
`tsae-paper-widthmatch-probing`, `tsae-paper-widthmatch-em`. **Never
touched** (house rule). **Han's call, still pending.** Also pending:
3-token rotation (`gh`, `hf_token`, `hf_token_datasets`).

## In flight (updated ~00:4x — the 20:00–00:30 chain is below)

1. **mac-d — the frontier, RUNNING.** 15 cells (3 SAE T1 + 12 TXC
   T{2,4,8,16}) under fresh tag `sycgen_keep_r1_rebuilt`, **6 workers,
   GPU 100%**, ~20 min out. Getting here took four separate blockers,
   all found by measurement:
   - **SAE anchor weights never existed locally** — masked by
     `runner.py:141-150` returning `train_cached=True` as a **literal**;
     `cache t=True` means "eval_key has a row", NOT "weights exist".
   - **The original 70-min burn was on `tsae` cells** the frontier never
     loads (arch-major sort: 3 sae + 3 tsae + 12 txc = 18 ✓).
   - **The 12-worker relaunch OOM-killed** — cgroup `memory.max` 233.8
     GiB vs `free`'s host 2015 GB. **Real cost is 24.5 GiB/worker** (a
     *concurrent load peak*: fp16 source + fp32 copy held together), so
     the ceiling is 9.5. 6 is stable.
   - **pod-D's originals are NOT lost** — all 6 on HF;
     `checkpoint_exists()` simply cannot see them.
2. **mac-c — geometry lane**, plus the durability audit that produced
   the HF finding. Claim-read DONE: **exposure confined to item 6**
   (probing/backtracking pool via the *protocol*; EM/RLHF already report
   TXC negatives, and a stronger baseline cannot rescue a negative).
3. **⚑ Pre-registration of interpretation is POSTED** (LOG ~00:3x) —
   four outcomes fixed before the numbers exist, incl. **underpowered as
   a distinct outcome from a loss.** Read it before reading results.

## ⚑ BRANCH STATE (settled 23:2x 07-28 — read before any git op)

**WORK ON `arxiv`. `main` is FROZEN at `7ceb45564` (its original March
commit) by Han's instruction.** A brief unification was done and then
reversed; both force-reverts used `--force-with-lease`, 0 commits ever
existed only on `main`, and mac-d's stranded commit was re-appended to
arxiv verbatim (`b99fe053c`). The pod never moved.

**Standing branch check — BOTH lines, never either:**

    git log --oneline <target>..<source> | wc -l   # MUST be 0
    git rev-parse <target> <source>                # MUST match

`arxiv..main` alone is **unfalsifiable** after a merge of main into
arxiv: it returns 0 in the success case AND the disaster case (mac-c
`851d73f85`). **The rev-parse line cannot be pointed the wrong way** —
that is the property worth having.

## Standing rulings (do not relitigate)

- **FRONTIER, not a matched point.** Constraining the SAE to TXC's
  per-window l0 forces 0.49 l0/token at T16 (degenerate); the arms'
  budgets scale differently in T, so matching one unmatches the other.
  Sweep k on both arms, plot vs **realized** `l0_per_window`, **keep
  the as-run points on the same axes**, verdict is **dominance**.
- **If TXC is dominated across the frontier, item 6 is a NEGATIVE** and
  we report it. Pre-registered before any number existed.
- **The harness is SECTION-AGNOSTIC** — likely needed for probing/RLHF/EM.
- **Accuracy outranks immutability for a deliverable** (btk re-render;
  mac-d executed `ff242b78` → `8d75ff3a`, only the T10 column moved,
  proven by per-T series diff).
- **⚑ `checkpoint_exists()` answers "on THIS BOX", never "anywhere".**
  I ratified "call it explicitly" at 23:3x and mac-c refuted it: it
  returns **False for all 344 HF-mirrored keys**, because `cache.py:148`'s
  hf_url branch is **dead code by construction** (`trainer.py:171` writes
  `hf_url=None` and is the only writer). `cache=True ⇒ weights exist` and
  `exists()=False ⇒ weights gone` are the **same bug in mirror image**:
  nothing is the authority on weight existence. **Revisit after item 6,
  not during.**
- **Read `cpu.max` and `memory.max`, NEVER `nproc`/`free`.** A container
  reports host resources. This cost us cores in the morning and bytes at
  midnight.
- **The anchor is the first number to read, not the headline curve** —
  a retrained anchor near its recorded value is what certifies the
  substrate underneath everything else.
- **⚑ A RETRACTION DOES NOT PROPAGATE TO FILES, only to memory.** All
  three agents found a claim they had already retracted still living on
  another surface — mine was on `REBUTTAL_HANDOFF.md`, presented as a
  "reusable result". **Sweep every surface after a retraction, WITH
  POSITIVE CONTROLS** (a grep that finds nothing and a grep that is
  broken produce identical output). mac-d's form: *when writing a
  correction, re-read your own latest position on the claim FIRST.*
- **The audit catches BROKEN, not WRONG.** It cannot know a pod list
  was true three hours ago. Check 9 (`API-verified` lines fail past 6h
  by git-blame age) narrows that, but only proves a claim was recently
  *restamped*, not that it is *correct*.

## Quote-form guards now live in the handoff

- **§6 ⛔ block** — item 6 is NOT quotable as an architecture win.
- **⚑ block above §1+2** — probing/RLHF/EM support *"windowed TXC reads
  state per-token probes cannot"*, **not** *"TXC beats a windowed
  SAE"*; no such comparison exists for any of the three.

## Ops

    .venv/bin/python scripts/handoff_audit.py --self-test   # 8 checks + staleness sweep
    .venv/bin/python scripts/cell_census.py --write         # regen before quoting coverage
    .venv/bin/python scripts/gen_handoff_tables.py          # items 4 + 5 tables

**Monitor:** persistent origin watch, task `b4g16b81d`, 45 s poll, emits
on repeated fetch failure so **silence ≠ blind**. It had been **down for
hours** — the original was a one-shot background command that fired once
and exited, and I was learning about pushes from push *rejections*.
**Re-arm after any session restart.**

**Push recipe:** fetch → rebase → **marker-check every file the push
touches** → audit → push. LOG conflicts are append-both (origin's entry
first, then mine). A conflict in a *deliverable* file is
stop-and-resolve-by-hand.

## My own errors today, kept on purpose for calibration

1. Misread "local mac agents own RLHF" as a venue constraint.
2. Claimed mac-d sat on a pod "~20 minutes" — it was ~3.
3. Diagnosed "bootstraps running in SERIES" from a `ps` snapshot — wrong.
4. Reported "~2.5 MB/s" on a download that had moved zero bytes in 30 s.
5. Claimed a "2.5–3×" rebalance win from extrapolated walls; measured 1.4×.
6. Predicted 0.1–0.35 s/step; measured 0.066.
7. Pushed a conflict marker into `REBUTTAL_HANDOFF.md`.
8. Passed a handoff surface by hand that `handoff_audit.py` then failed.
9. **`nc` probe reported "PORT CLOSED" from a command-not-found** (no
   `timeout(1)` on macOS) — a verdict with no measurement behind it.
10. **"3 pods, $6.42/h"** — missed one; and my first API script printed
    "128 pods, $109.03/h" by summing over EXITED records.
11. **Propagated `floor_excess ≡ f` as exact at 2e-6** — verified against
    a *simulation*, never the screen's floor features. mac-c refuted it;
    the real window is **T+w**.
12. **Quoted "~9× budget gap" as a figure** when it was an upper bound
    derived from an assumption.
13. **Specified a budget-match that does not exist**; mac-d's arithmetic
    killed it before it cost pod hours.
14. **Let the origin monitor die and did not notice for hours.**
15. **Called a job "STUCK" and escalated to Han for a takeover** on
    evidence that could not distinguish slow from hung — my own `/proc`
    read said "computing, not blocked" and I put the wrong word on it.
16. **"~8 min/cell" quoted as a budget** — derived from someone else's
    cost estimate, never measured.
17. **Read the pod's UTC clock as BST** and announced a deadline had
    passed when it had not.
18. **Reported "6 workers" from a `pgrep -fc`** that was counting my own
    ssh command.
19. **Framed the 3 × 15.2 GB duplication as a SPEED problem.** It was a
    SCALE-CEILING problem — right measurement, wrong consequence.
20. **Asserted "the sycgen data path is dataloader-bound"** and built a
    patch recommendation on it. GPU went 10% → 100% on an arch change;
    we were measuring `tsae`, not the data path.
21. **Ratified "call `checkpoint_exists` explicitly"** as the fix to a
    bug whose mirror image it is.

**The pattern in 9–14: I state derived quantities with the confidence of
measured ones.** The fixes the fleet converged on independently, and
which I should apply to myself first: **verify by instantiation, not by
reading**, and **a caveat in a JSON is not a control**.
