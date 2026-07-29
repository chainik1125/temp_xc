# mac-local — STATUS · SNAPSHOT #5 (2026-07-28 ~22:0x BST)

**Supersedes SNAPSHOT #4 (16:1x).** Rewritten because the day turned:
a *delivered* exhibit is under challenge, and the challenge spread to
three sections. I am the hub — review, ratify, rulings, the binding
LOG, ledger oversight, and the handover surfaces
(`REBUTTAL_HANDOFF.md`, `REBUTTAL_CODE_GUIDE.md`,
`REBUTTAL_CELL_CENSUS.md`). **No compute of my own.**

---

# RESUME HERE

## ⚑⚑ OVERNIGHT MODE (Han asleep from ~01:5x 07-29) — read this first

### WHERE ITEM 6 STANDS, as of 02:53 — it narrowed THREE times tonight

1. **"a clean TXC win"** → **above 2/4 T at matched budget**
   (`73f8ea388`; the comparator rule was biased toward us).
2. → **the shuffle gap is (b) ARCHITECTURAL, NOT LEARNED**
   (`90e0a4e2a`; a random-init TXC is *more* order-sensitive, 11/12
   cells, pre-registered and pre-committed to publishing).
3. → **"per-token probes sit at chance" was FALSE** (`2b2fc4266`;
   `tok_best` 0.50–0.53 against a ~0.33 **3-class** null — someone read
   0.50 as chance). **sycgen is NOT per-token-silent.**

**Each narrowing came from running a comparison we had not run. None
came from a reviewer.** **T=16 is the one unambiguous cell** (Pareto:
pooled cannot operate at TXC's budget at all).

**What item 6 supports:** per-token 0.50–0.53, windowed 0.62–0.65, a
gain of **+0.11–0.12** at matched sparsity **against baselines reading
the same activations.** Not per-token blindness. **The reviewer
response reflects only what survives** — a level claim, no order claim.

### PROGRAM-LEVEL: the hunt's aim was WITHDRAWN twice tonight

- **`floor_reach` demoted** to a kill filter (`c6a6c756f`) — my own
  ratified screen rule; it ranks our only KEEP behind a WEAK.
- **Per-token silence WITHDRAWN as a positive criterion**
  (`7d8a8a18d`, propagated into `hunt-safety-gold-clew.md`) — measured
  on two corpora: `arm_excess` tracks `tok_excess` (+1.000 / +0.943),
  so **screening for silence selects AGAINST the arm.**
- **Replacement:** rank by `tok_excess × amplification` **and by
  `arm_excess`**; the amendment **yields to one weak-tok/strong-arm
  counterexample.**
- **⚑ Flagged for Han, NOT concluded:** this governs the HUNT's screen,
  not the paper's ambience claim — but it points the other way.

### TOOLS BUILT TONIGHT (use them; the rules alone did not hold)

    scripts/pod_inventory.py       full fleet, never tail
    scripts/claim_sweep.py         live-vs-quoted retraction sweep + control
    scripts/check_response_math.py 6 LaTeX/render checks, self-tested
    scripts/check_response_sync.py arxiv vs dmitry branch drift
    scripts/check_response_numbers.py quoted table vs frontier.json

**Every one exists because a written rule failed and a script did not.**


**⚑ FORMAT + CO-EDITING CHANGE (02:23):** the reviewer-response
tables are now **markdown pipe tables** (Dmitry's agent converted them;
it retires the LaTeX/backslash-escaping problem entirely). **And
`dmitry-txcwins-10h` is ACTIVELY EDITED** — it moved twice in twenty
minutes. **Never copy your file over theirs: rebase onto their current
tip and graft only your block.** Both copies are byte-identical now
(434 lines, 0 drift).

**Han: *"i'm going to head to sleep, continue orchestration of agentic
work"*, plus two standing instructions: keep `REBUTTAL_HANDOFF.md`
updated as results land, and keep
`docs/dmitry/reviewer_responses/reviewer_responses_1.md` updated **on
both `arxiv` and `dmitry-txcwins-10h`**.

**`briefings/OVERNIGHT-STANDING-ORDERS.md` is the operative document**
— pre-authorised spend defaults, hard stops, what needs Han and must
WAIT, and §6's results-landing procedure across all three surfaces.

**Hub beat while he sleeps:** pull → ratify worker pushes in the LOG →
if a sycgen number moved, run §6 → `handoff_audit.py`,
`check_response_math.py`, `check_response_sync.py` → push. Fleet is
mac-d's single A40 at $0.44/h; **hard stop 3h after pod-up.**

**Do NOT wake him for:** anything covered by the standing orders.
**Do leave for him:** token rotation, the 4 unattributed pods
($9.41/h ≈ $226/day), and coordination with Dmitry — `stacked-em-steer`
is recomputing the Stacked-SAE-on-EM number the Reviewer-1 response
quotes, so **that section is frozen overnight.**



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
| **mac-d** | **1 pod `mac-d-sycshuffle-0729` (A40, $0.44/h)** | ⚑ **TOP: `briefings/sycgen-shuffle-sparsity-matched.md`** — **CLAIMED, card frozen, sharding built, running.** Sized down from my 20×H100 authorization because the lane trains nothing (encode-and-probe) |
| **mac-c** | **0 pods, $0** | lever-3 rescue (`a027b7caa`, P1/P2/P3 held, P5 fired), shuffle-brief pre-reg audit (A1–A5), A1 receipt, band review, and the floor-aim correction |

### ⚑ THE ONE LANE IN FLIGHT — sycgen shuffle ablation, sparsity-matched (01:2x)

Han (00:3x): *"we NEED sycgen proper shuffle ablation with matched
sparsity."* **CLAIMED BY mac-d, card frozen before any cell ran**
(`SHUFFLE_MATCHED_CARD.md`, `62fd1536d` + `ab415af18`). **mac-d
corrected the scale down from my 20×H100 authorization to 1 GPU** and
verified weight existence first (0/15 train_keys on HF, 5 repos,
positive control firing) — so it is a retrain, sized to the work.

**Review loop closed through three rounds, all $0:**

1. **mac-c audit** (A1–A5) → hub adopted all (`29bc6a95d`). **A1 was
   mine and blocking**: the pooled-zero gate tests *pooled*, not the
   shuffle; a no-op shuffle passes it and reads as **(b)**, the
   pre-committed headline. mac-c then made it an executable receipt —
   the gate returns PASS on a dead shuffle at every T.
2. **mac-d strengthened A1** past the minimal assert: gate the measured
   shuffled-row fraction against the predicted `1 − 1/T!`, catching
   partial application and wrong-axis permutation.
3. **mac-c reviewed the strengthening** — it converts a deterministic
   check into a statistical one, and **at T=8 an equality gate
   spuriously VOIDS 9.66% of HEALTHY runs.** Exact mirror of A1: A1
   could not fire when it should; its fix fires when it should not.

**BINDING BANDS (hub `33a5c72d8`, independently re-derived):**
`Binomial(n, 1/T!)`, `n = n_windows·(L//T)` — **T=2 7936..8448 · T=4
268..414 · T=8 0..3 · T=16 0..0.** Print observed count, band, and `n`
per cell. *(My own E±4σ construction was WRONG at T=8/16 — a σ band is
meaningless at λ≈0.1; use tail probability. mac-c's numbers stand.)*

**Standing lesson: state the false-pass AND false-fail rate of any gate
before it guards a run.**

**Also binding:** §7 withdraws my §2b "refine the k grid" guidance —
**the grid is already tight (1.4–7.7% at T=2/4/8; T=16's k=1 floor is
structurally unreachable). Add no k cells; spend on SEEDS (n=3→5),
since outcome (d) is still unsized.**

**⚑ UNATTRIBUTED PODS — RE-QUERIED 01:2x 07-29, THE PICTURE CHANGED.**
Now **4 non-convention pods at $9.41/h** (was 3 at $3.87/h):

| pod | $/h | note |
|---|---|---|
| `reviewer-btk-tsae-300k` | 2.99 (H100) | **NEW** |
| `reviewer-headline-multiseed` | 2.99 (H100) | **NEW** |
| `stacked-em-steer` | 2.99 (H100) | **NEW** |
| `tsae-paper-widthmatch-probing` | 0.44 (A40) | was already up |

**⚑ CORRECTED 02:0x — an earlier read used `tail -8` and lost a row.**
**5 unattributed pods = $12.40/h ≈ $298/day** (`mats-lenctl-h100`,
`reviewer-btk-tsae-300k`, `reviewer-headline-multiseed`,
`stacked-em-steer`, `tsae-paper-widthmatch-probing`). Ours: $2.20/h.
Total fleet **$14.60/h**. Use `scripts/pod_inventory.py` — never `tail`. **Never touched** (house rule:
agents do not modify pods they did not spin up). **Han's call, still
pending** — and it is now ~**$226/day**, not ~$93.

**⚑ COORDINATION SIGNAL, bigger than the cost:** those three names are
**rebuttal work by someone outside this fleet**. `stacked-em-steer` in
particular is **reviewer 1's requested Stacked-SAE baseline on EM
steering** — the one task where our own response says stacked beats the
TXC. Whoever is running it may be about to change a number the
Reviewer-1 response quotes. **Worth a word with Dmitry before we freeze
that section.**

Also pending: 3-token rotation (`gh`, `hf_token`, `hf_token_datasets`).

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
