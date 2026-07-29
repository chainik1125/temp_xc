# mac-d STATUS — RunPod-API executor agent

**RESUME HERE FIRST.** Updated 2026-07-29 02:3x BST.

## Headline

**SHUFFLE LANE DELIVERED. ZERO PODS. $0.00/h. Nothing unpushed.**

Verdict **(b) ARCHITECTURAL, NOT LEARNED** — a randomly-initialised TXC
is *more* order-sensitive than the trained one in **11 of 12** (T, seed)
cells. **Not one cell returns (a).** PTR — pending team review.

Nothing is mid-flight.

> **Earlier item-6 headline superseded (hub `73f8ea388`).** My delivered
> "ABOVE pooled at 3/4 T" is **above 2/4** (T=8, T=16), INDISTINGUISHABLE
> at T=2/T=4, never below. Not my arithmetic — the hub's comparator rule,
> which I implemented as briefed: it picked pooled's best point at
> `l0 ≤ TXC's l0`, which on a 40–75%-step grid compared **TXC @ 5.66
> against pooled @ 3.51 at T=2 (38% less budget)**. T=16 remains the
> strong, unambiguous cell — **Pareto dominance**, pooled cannot operate
> at TXC's budget at all.

## What was delivered (shuffle lane)

- **`figs_writeup/tab_sycgen_shuffle_matched.md`** — the acceptance gate.
- **Card** `experiments/explorations/task_hunt/sycgen/SHUFFLE_MATCHED_CARD.md`,
  frozen **and amended before any cell ran** (git history is the receipt).
- **Scripts** `sycgen/{run_shuffle_matched,report_shuffle_matched,gen_shuffle_table}.py`
- **Data** `sycgen/results/shuffle_matched.shard{0..7}.json` — 624 rows,
  24 gate receipts, 24 cells.
- **Ledger closed**; both pods terminated and **API-verified gone**.

## The verdict in one screen

    gates: shuffle-live 24/24   pooled gap = 0 (6.53e-09)   SAE l0 invariant (0 viol)

    T      trained gap   twin gap   trained>twin
    2        +0.1114     +0.1671        0/3
    4        +0.0231     +0.1375        0/3
    8        +0.0504     +0.0820        0/3
    16       +0.0618     +0.0267        1/3

15 of 16 (T × draw × probe) return (b) or (c). None returns (a).

**This does NOT say TXC fails.** Ordered recovery **0.499 → 0.578**
across T vs the twin's 0.222 → 0.058 — training works. It says the
**ordered−shuffled gap is the wrong evidence for it.**

## Live caveats — do not let these get lost

1. **The twin barely does the task** (0.058 ordered at T=16 vs 0.578),
   so its raw gap is a difference of two near-chance numbers and raw
   gaps may not be commensurable across a 10× base-recovery difference.
   **This is a limitation of the rule I pre-registered.** Reported, not
   used to overturn the verdict. The post-hoc *relative* gap — checked
   *because* it was the reading that might have favoured us — makes the
   negative **stronger** (twin loses 76–79%, trained 4.5–22%).
2. **Budget confound:** twin runs at `l0`=8.00 vs trained 5.44–7.86 (up
   to 1.47×), plausibly inflating the twin's gap. Smallest at **T=16
   (1.02×)** — and T=16 is where the twin gate is *least* decisive (1/3
   seeds). **The cleanest cell is the least conclusive one.**
3. **Stacked cannot be budget-matched at T=8/16** — structural: its
   `l0` is a sum over positions so its floor is `T·1` = 8.00/16.00 vs
   TXC's 7.22/7.86. No finer `k` closes it. **"Matched" is not claimed
   at T=8/16** (ratios 0.64–2.05 printed).
4. **`mono NO` is expected, and my precondition was mis-specified.** The
   *gap* need not be monotone in budget; I inherited that precondition
   from item 6 where the interpolated quantity was *recovery*. The
   bracket should be read as a **range**, not an interpolated point.
   **Not yet fixed.**

## Traps that cost real time — re-read before trusting a check

- **`_key_from_manifest` resolves 6 of 15 cells on a fresh box**, and
  the 9 it misses are exactly the TXC cells — **the whole claim arm**.
  Manifest entries are written *on the pod* and **containers never
  push**. Use the **leaderboard** (it is in git). The sweep skipped the
  claim arm per-cell and still **exited 0**.
- **`repo_type` is part of the search space.** Searching 5 HF *model*
  repos (1506 keys, positive control firing) returned a confident
  **0 of 15**; the mirror is the **dataset** repo `temp-bench-data`
  under `ckpts/<train_key>/`. All 15 were there.
- **Read `cpu.max`/`memory.max`, never `nproc`/`free`.** 1×A40 pod:
  cgroup **7.7 CPUs / 46.6 GiB** while `nproc` said 96, `free` 503 GB.
- **A sigma band is meaningless at λ≈0.1 or 1e-10.** My "wider of
  binomial and 4σ" rule gave **0..1 at T=16** where the binding band is
  **0..0** — I verified the *component*, not the composed rule.
- **`oom_kill` in `memory.oom_control` is a config line, not a
  counter** — `grep -c` returns 2 on a healthy pod. Real counters:
  `memory.failcnt` and the `oom_kill` *value*.
- **macOS `wc -l` emits leading whitespace**, so `head -$(wc -l < f)`
  breaks. Pipe through `tr -d ' '`.
- **A rejected `git checkout` leaves the pod on the OLD pin** and the
  next run silently uses stale code. Check `git rev-parse` after.

## Sizing rule earned here

**Measure which resource binds before buying GPUs.** GPU utilisation
during a real cell: **mean 1.4%, idle in 94% of samples**. Three
concurrent shards on 1×A40: **one OOM-killed, survivors 207 s → 381 s,
zero gain** — each process materialises its own **15.2 GB** copy of the
activations. **RAM binds — not GPU, not CPU.** 4×A40 was bought for its
**186.3 GiB / 32.3 CPUs**; **8×H100 declined** as $23.92/h of 94%-idle
silicon. Lane cost **≈ $2.12** against a **$60/h** authorization.

**24 cells is a hard parallelism ceiling** (partition-tested) — beyond
24 shards, shards go empty whatever the budget allows.

## Standing obligations

- **Post-compact: run `agents/mac-d/PAPER_FAITHFUL_CHECK.md`** — Han's
  standing instruction, every compact, no exceptions.
- Pods: `mac-d-<purpose>-<mmdd>`; ledger at spin-up **and** teardown;
  terminate > stop; verify by API query. **Never touch a pod I did not
  spin up** (4 unattributed at $9.41/h — untouched all night).
- Secrets: RunPod + Claude keys env-injected from keychain only, never
  echoed / written to a file / passed as argv, and **never seeded to
  pods**. Pods get gh + hf×2 only — no Modal, no Anthropic. `chmod` is
  a **silent no-op** on the `/workspace` MooseFS FUSE volume.
- Stamps only from a separate preceding `date` call.
