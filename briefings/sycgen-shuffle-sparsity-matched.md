---
status: active
owner: mac-d (executor) + mac-c (review/pre-reg audit)
issued-by: mac-local (hub)
issued: 2026-07-29 01:2x London
priority: TOP — closes the gap Han found in the item-6 exhibit
scale: AUTHORIZED up to 20 simultaneous H100s (Han)
---

# sycgen shuffle ablation, SPARSITY-MATCHED — and the design trap in it

Han: *"we need the proper sycgen shuffle ablation sparsity matched"* —
after noticing the two sycgen exhibits rest on **different
comparisons** and nothing crosses them.

- `fig_sycgen_shuffle_tsweep.*` — ordered vs shuffled, against a
  **per-token anchor + untrained twin**. The framing Dmitry's challenge
  undermined.
- `tab_sycgen_budget_matched.md` — TXC vs pooled/stacked at matched
  budget, **no shuffle dimension at all** (`frontier.json` has no
  shuffle key).

**Nobody has asked: does the ordered−shuffled gap survive when the
comparator is a sparsity-matched SAE rather than a per-token probe?**

---

## 1. ⚑ READ THIS FIRST — pooled CANNOT have a shuffle gap, by construction

**Verified before this brief was written:**

    pooled  ordered vs shuffled : IDENTICAL   max|diff| 5.96e-08 (float noise)
    stacked ordered vs shuffled : DIFFERENT   max|diff| 4.12

**Mean-pooling per-token codes over a window is permutation-invariant.**
Shuffling tokens inside the window cannot change the mean. So pooled's
ordered−shuffled gap is **exactly zero, always, on any data, for any
model.**

**Three consequences, and they define the experiment:**

1. **Pooled is NOT a comparator for this question — it is an INSTRUMENT
   CHECK.** Its gap must come out 0. **If pooled shows a non-zero gap,
   the shuffle instrument is broken and every number in the run is
   void.** This is the single most valuable cell in the design.
2. **Reporting "TXC's shuffle gap beats pooled's" would be a
   TAUTOLOGY dressed as a result** — beating zero by construction. **Do
   not do it, and do not let the table imply it.**
3. **Stacked is the real baseline.** Concatenation is position-dependent,
   so stacked gets order information **for free from its architecture,
   with no temporal learning at all.** That is exactly the null the
   claim has to beat.

**⇒ THE CLAIM UNDER TEST: does TXC's ordered−shuffled gap exceed
STACKED's at matched measured budget?** Not pooled's. Anyone who
reports against pooled has measured a mathematical identity.

## 2. Pre-registration — write into the card BEFORE any cell runs

State, before numbers exist:

- **(a) TXC gap > stacked gap at matched budget** ⇒ the windowed
  architecture uses order beyond what concatenation supplies.
- **(b) TXC gap ≈ stacked gap** ⇒ the gap is **architectural
  position-sensitivity, not learned temporal structure.** This is the
  live hypothesis: **sycgen's original shuffle result already dissolved
  once under exactly this pressure** — untrained twins showed *larger*
  gaps than trained models.
- **(c) TXC gap < stacked gap** ⇒ report as a negative.
- **(d) INDISTINGUISHABLE at n=3** ⇒ a distinct outcome; **not a win,
  not a loss.**

**Binding controls:**

- **Untrained twins are MANDATORY, not optional.** They are what killed
  the original claim. A trained-only gap is not evidence.
- **pooled's zero is the instrument gate** — check it first, abort on
  failure.
- **Sparsity matched on MEASURED `realized_l0_per_window`**, never
  nominal k, never the derived per-token axis.
- Report `l0_unit` per arm; stacked's `T·d_sae` probe-capacity
  advantage disclosed and **never netted out** — and remember it is
  **uninformative at T≥8** (32768 features vs 1024 windows).

## 2b. ⚑ AMENDMENT 01:0x — the budget table's comparator rule was
## biased, and the same trap is waiting for you here

A hub sanity check on the *delivered* budget table found the bug this
run would have repeated. The generator compared TXC against **the best
SAE point with `l0 ≤ TXC's l0`**. Sounds conservative. It is not: k is
swept on a **coarse grid** (1,2,4,8,16,32), consecutive points differ
by **40–75% in budget**, so no point lands at TXC's budget and the rule
silently selects a **much cheaper** baseline. **At T=2 it compared TXC
@ 5.66 against pooled @ 3.51 — 38% less budget — and returned a win.**
The point 5% *above* TXC's budget was indistinguishable. **Headline
went from above 3/4 to above 2/4.**

**Binding for this run:**

- **Bracket, never single-sided.** Report the best point **below** and
  the cheapest point **above** TXC's measured budget, and interpolate
  to TXC's exact `l0`. A verdict from one side of a coarse grid is not
  a matched-budget verdict.
- **Sweep k finely enough to bracket tightly.** Add intermediate k
  where the grid straddles TXC's budget — cheap, and it is the
  difference between a defensible number and an artifact. **A tight
  bracket is worth more pod-minutes than another seed.**
- **State the bracket width** in the card. If the two ends give
  different verdicts, that IS the finding — report it, do not pick.
- This applies to the **gap** as much as the level: a gap difference at
  mismatched budget is not a matched-budget gap difference.

**The general lesson, and it is why this brief exists twice over:
a selection rule that is defensible in words can be biased in
arithmetic, and it will bias toward whoever wrote it.** Check the rule
against the grid it runs on, not against its own description.

## 3. Grid and scale

`T {2,4,8,16}` × `seeds {1,2,42}` × arms `{txc, pooled, stacked}` ×
`{ordered, shuffled}` × k-sweep × `{trained, untrained-twin}`.

**Han has authorized up to 20 simultaneous H100s.** Notes from
tonight's run, so the scale is usable rather than aspirational:

- **Read `cpu.max` and `memory.max`, NEVER `nproc`/`free`** — a
  container reports host resources. This cost us cores in the morning
  and an OOM at midnight.
- **`max_tasks_per_child=1`** — `_SYNTHETIC_CACHE` is per-process, so
  long-lived workers crossing seeds accumulate ~3× datasource RAM.
- Peak per worker was **24.5 GiB** (concurrent load: fp16 source + fp32
  copy held together), not the ~19 GB steady RSS.
- The 15 sycgen checkpoints exist under tag `sycgen_keep_r1_rebuilt`;
  the eval arms are **post-hoc transforms of the frozen T=1 SAE**, so
  much of this is encode-and-probe, not retrain. **Verify weight
  existence FIRST** — `checkpoint_exists()` answers "on this box", not
  "anywhere", and `cache t=True` is a literal that means "a row exists",
  not "weights exist".

**Ledger both ends; terminate at lane end and API-verify.** 20×H100 is
~$60/h — **it is authorized, which is not the same as free.** Post the
pre-spend estimate before launching.

## 4. Acceptance gate

`figs_writeup/tab_sycgen_shuffle_matched.md` + a card carrying the
pre-registration written before the run, the pooled-zero instrument
receipt, the untrained-twin columns, realized `l0_per_window` per cell,
and a plain-words verdict naming which of (a)–(d) fired.

**If the answer is (b) — architectural, not learned — that is the
result and we publish it.** The prime directive has not moved: *a sound
verdict, never a win.*

**Delete this file when the lane closes.**
