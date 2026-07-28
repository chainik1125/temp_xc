# mac-local STATUS — SNAPSHOT #4 (16:1x London 07-28, date-checked)

**Supersedes PRE-COMPACT #3 (13:13), which is stale in almost every
particular.** I am the hub: review, ratify, the binding LOG, ledger
oversight, and the handover surfaces (`REBUTTAL_HANDOFF.md`,
`REBUTTAL_CODE_GUIDE.md`, `REBUTTAL_CELL_CENSUS.md` — regen:
`scripts/cell_census.py --write`). **No compute of my own.** Stamps
from `date` at write, interpolated, never pre-written.

---

## THE HEADLINE

**Deliverables 1–7 are all delivered.** Item 7 is delivered *as a
diagnosed negative*, and **Han has reopened the hunt** — *"the task
hunt must continue — a second safety relevant task would be platinum"*
(16:1x, binding).

---

## Fleet (API-verified 16:12)

| what | state |
|---|---|
| `mac-d-rlhfpf-0728-5` | **1×H100, $2.99/h** — running the three deferred btk gap cells |
| mac-c's L40S | **TERMINATED ~16:05** (lane closed before my hold order landed) — a re-screen needs a fresh pod |
| everything else | terminated, guarded + API-verified |

**Burn $2.99/h**, from a $17.94/h peak. **The whole 15-cell pf grid
cost ~$20** against a morning projection of ~213 GPU-h.

**Agents:** `mac-d` (RLHF + pods + renderer), `mac-c` (the whole hunt).
**All three of us are sessions on ONE MacBook** (M5 Pro / 48 GB) —
agent count is not machine count.

---

## Deliverables

| # | item | state |
|---|---|---|
| 1,2 | probing k5/k20 | delivered, both arms, FINAL |
| 3 | RLHF | **delivered, both arms** — pf grid **15/15 COMPLETE**, HANDOFF § 3 final (mac-d's block) |
| 4 | λ̂ backtracking | delivered |
| 5 | dq question-marks | delivered |
| 6 | sycgen | delivered (KEEP, first hunt gold) |
| 7 | hunted task #4 | **delivered AS A NEGATIVE** — `retryesc_gen` WEAK 3/3 |

**pf RLHF verdict (binding):** 15 cells, uniform 3 seeds at
T{2,4,6,8,10}; whole-grid gap **−0.00279, t = −1.29, df 14, p = 0.219,
NOT significant**, 0.13× the anchor seed-scatter. **No order effect at
any T — matching the btk arm. Two independent arms, one null.** No
T-trend (T8 sign-mixed mid-range; two all-negative of five is P = 0.12).
Resolved against a threshold **pre-registered before the deciding cell
landed**.

**T1/T16 absent from pf BY DESIGN** — upstream's archs are
`t2,t3,t6,t7,t8,t10,t15,t20`; neither exists. Both covered by btk at
3 seeds. **Every T Han asked for is covered by at least one arm.**

**Item 7's negative, in one line:** gain bar CLEARED every leg
(+0.063…+0.069) but the FLOOR clause killed every leg — the task
detects, it does not discriminate; cause = corpus density overshoot
(0.261 vs 0.185 aimed) from a biased estimator, now corrected.

---

## In flight

1. **btk gap cells** — T6/s2, T10/s1, T10/s2 on the surviving pod;
   keys pre-verified by me against the 26 existing btk rows. Brings btk
   to uniform 3 seeds. **Open suggestion, not yet a directive: 3 cells
   on 3 pods costs the SAME as 3 on 1 and is 3× faster.**
2. **mac-c attempt 2** — regenerate `retryesc_gen` at a corrected
   density target. **Gate before spend: the corrected `claim_zone` must
   RETRODICT today's miss (0.185 + 0.076 == 0.261) or it is a fudge
   factor and may not be used to aim.** Pre-registered: bars unchanged,
   **attempt 2 of a cap of 2** for this face family (a third needs a new
   *mechanism*, not a new density), re-aim disclosed with any KEEP.
   Gold-visibility arms only at a KEEP on the screen.
3. **⚑ TOKEN ROTATION — Han's call, still pending.** Rotate **`gh`,
   `hf_token`, `hf_token_datasets`** (pod-staged, therefore 666-exposed).
   *Not* the runpod / claude / s2 keys — never seeded to pods.

---

## Today's load-bearing findings

- **Substrate: `gemma_2_2b_base_l12_phase7`**, not l13-IT. Settled twice
  by measurement (anchor FVU 0.0036 vs 0.0367; step-0 init within 4% of
  upstream vs 84% high).
- **Upstream training contract recovered from source** (`94119bc08`,
  `experiments/phase5_downstream_utility/train_primary_archs.py`): **no
  scheduler, no warmup**, constant Adam 3e-4, grad_clip 1.0, plateau
  stop (<2% over a 5-point window, min 3 000) — which is why upstream
  runs ended at 4 200/4 600/5 200 steps, not 25 000.
- **The "818 MiB/s feed-bound" cost was a PORT BUG**: `consumes="sequence"`
  ships whole 128-token sequences (**1152 MiB/step**) to use T+1 of them
  (**27 MiB at T2**) = **42.7× over-transfer** — 814 computed vs 818
  measured, and it explains the otherwise-strange T-independence. Fix =
  `TEMP_BENCH_BUFFER_RESIDENT=1` (opt-in, default off), bitwise-receipted,
  **219.7× on CUDA**.
- **Measured ~0.066 s/step at T2** — a cell is **~10 min**, not hours.
- **`chmod` is a SILENT NO-OP on the RunPod MooseFS FUSE volume** —
  every pod-staged token sat at 666, not 0600.

## House rules adopted today

1. **Liveness = `/proc` receipts**, never GPU point-samples + log size.
2. **Receipts cite the ARTIFACT** (bytes/manifest), never process state.
3. **Timing claims come from the production path AND production
   hardware** — a fix must be measured on hardware that HAS the defect.
4. **Derive, don't estimate** — including the clock. A rounded number
   sampled twice is not a rate.
5. **Never `set -x` in a script touching a secret** (`GIT_ASKPASS` is
   the pattern); **`chmod` on FUSE must be verified by reading the mode
   back**.
6. **Marker count is necessary, not sufficient** — after any conflict
   resolution on a deliverable file, diff the block against what you
   intended to keep. **`--ours` is INVERTED under rebase.**
7. **Don't edit a file concurrently with its owner.**
8. **Push recipe:** explicit-path `git add`, retry loop with the
   keep-both LOG healer, and a **marker check on EVERY file the push
   touches** — a conflict in a deliverable file is
   stop-and-resolve-by-hand, never `rebase --continue`.

## My own corrections today (calibration for whoever reads this next)

Wrong about: a "20-minute" pod claim (3 minutes); a "serial bootstrap"
(they were parallel — I read `ps` instead of artifacts); a "2.5–3×"
rebalance win (1.4×); T6/T8/T10 walls (40/60/90 extrapolated vs
20/28/21 measured); the step-time band (0.1–0.35 vs 0.066); "~2.5 MB/s"
on a download that was transferring **zero**; and I pushed a conflict
marker into `REBUTTAL_HANDOFF.md`. **Pattern: asserting from an
insufficient read.** The pre-registered T10 threshold is the one place I
bound myself first, and it paid — it forced the null against the reading
I preferred.

## Ops

- **Watcher:** `scratchpad/watch_origin.sh`, armed ONLY via its own
  `run_in_background` call, liveness-checked (wrapper+child = 2 pids =
  ONE instance). **Never inline `&`.**
- **Pods:** `agents/mac-d/podctl.sh {mine,ssh,status,terminate}`.
  **Look, don't touch** — mac-d owns them.
- **Cache:** RLHF activations are **local** at
  `~/caches/rlhf/txcdr-base-data/` (14.16 GB), installed at
  `results/data_cache/44b72320bc3a56e2`. Durable origin: HF dataset
  `han1823123123/txcdr-base-data`.
- **Key reproduction** (for any orphan check): merge
  `arch_hparams_override` into the ArchSpec **and** pass
  `section='rlhf'`, exactly as `runner.py:116-122` does. Omit either and
  every cell looks orphaned.
