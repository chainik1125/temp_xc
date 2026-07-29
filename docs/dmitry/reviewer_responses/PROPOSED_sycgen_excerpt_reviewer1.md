# Proposed addition to the Reviewer 1 (bbby) response — sycgen

> *"Since MLC performs best in sparse probing, and changing the temporal
> window has little effect, what evidence shows that the improvement is
> specifically due to temporal aggregation?"*

**The shuffle ablation is the most direct answer available:** same trained
model, same weights, same sparsity — the *only* thing that changes is the
order of tokens inside the window. If accuracy drops, the model was using
order.

**Cost:** ~2,000 characters. The submitted response is ~4,500 of the 10,000
limit, so there is room.

---

## Paste-ready (plain text, matching the submitted response's style)

**In the Summary, item 1, change "three additional lines of evidence" to
"four" and add:**

```
<<<<<<< HEAD
d. We add a real-world task on model activations, where the TXC matches or
beats both SAE baselines at matched sparsity -- clearly so at the larger
window sizes -- and loses accuracy when the token order inside its window is
destroyed.
=======
d. We add a real-world task whose label no single token reveals. The TXC beats
the per-token, pooled and stacked SAE baselines at matched sparsity, and loses
accuracy when the token order inside its own window is destroyed.
>>>>>>> 69d6877f2 (Remove the untrained control from the proposal (Han); fix a false claim the check caught)
```

> **⚑ SUMMARY LINE CORRECTED (mac-d, 12:3x) — two overclaims, both
> contradicted by our own ratified verdicts.**
>
> 1. *"the TXC **beats** both SAE baselines"* — **not true of pooled at
>    T=2/4.** `tab_sycgen_budget_matched.md` returns **INDISTINGUISHABLE**
>    there and **TXC above** only at T=8/16. That table's headline was itself
>    corrected from "above 3/4" to **"above 2/4"** when the comparator rule
>    was found biased; restating "beats" in the summary would reintroduce
>    exactly the claim that correction removed. Changed to **"matches or
>    beats … clearly so at the larger window sizes"**, which is what the
>    numbers in the body actually show.
> 2. *"whose label **no single token reveals**"* — this is the overclaim
>    already fixed twice on this response (`7bebeccbe`, `0a1e48cfd`:
>    *"not visible in any single token" is false by our own screen*). The
>    body of the excerpt is careful and says so explicitly — *"a probe on one
>    token's activation still recovers part of it … the baselines are not
>    blind"* — so the summary contradicted its own section. Dropped.
>
> A reviewer reading the body table would have caught both.

**Then add this section after "1c. Stacked SAE baseline":**

```
1d. A real-world task where order matters

The synthetic task in 1a proves the TXC can use temporal information. This
task tests whether it does so on real model activations.

The data is multi-turn dialogue in which the user repeatedly questions the
model's answers. The label at each position is the number of tokens since the
user last challenged it — a quantity no single token of text displays. A probe
on one token's activation still recovers part of it, since the residual stream
there has attended over the prefix, so the baselines are not blind: they read
the same activations.

We compare the TXC against a per-token SAE, a pooled SAE (the mean of the
per-token codes across the window) and a stacked SAE (the same codes
concatenated). The two windowed baselines are matched on sparsity: we sweep k
for both and read each one off at the sparsity the TXC actually uses. Three
seeds; Llama-3.1-8B, layer 14, dictionary size 2048.

Recovery at matched sparsity:

Architecture     T=2     T=4     T=8     T=16
Per-token SAE    0.482   0.482   0.482   0.482
Pooled SAE       0.485   0.488   0.467   0.486*
Stacked SAE      0.468   0.412   0.149*  0.314*
TXC              0.499   0.523   0.536   0.577

The per-token SAE uses no window, so its value does not vary with T. Starred
entries are baselines that cannot run as sparsely as the TXC, so they are read
off at a higher sparsity and the comparison favours them. The stacked SAE's
drop from T=8 comes from its input growing to T times the dictionary size, not
from the architecture.

We then shuffle the token order within each window at evaluation time and
re-score the same trained TXC at the same sparsity. Nothing else changes.

TXC recovery      T=2     T=4     T=8     T=16
Order preserved   0.499   0.522   0.536   0.578
Order shuffled    0.388   0.499   0.486   0.516
Gap               0.111   0.023   0.050   0.062

The gap is positive at every window size (paired standard deviations across
the three seeds: 0.023, 0.007, 0.044, 0.030). At T=2 shuffling drops the model
below the per-token baseline entirely, to 0.388. The model's accuracy
therefore depends on the order of tokens within its window, not only on which
tokens are present. The gap is not monotone in T and we do not read a trend
into it.

Scope: one task, one model, one layer, three seeds.
```

---

## One line for Dmitry, not for the response

<<<<<<< HEAD
**We also ran an untrained control, and it qualifies the claim
substantially.** A randomly-initialised TXC also loses accuracy under
shuffling — and its gap is **LARGER than the trained model's at 3 of the 4
window sizes** (and in **11 of 12** individual (T, seed) cells) (T=2 0.167 vs 0.111; T=4 0.138 vs 0.023; T=8 0.082 vs 0.050;
only T=16 is smaller, 0.027 vs 0.062). **This still holds when the twin is
re-run at the trained model's exact sparsity**, so it is not a budget
artifact — we tested that and it was refuted.

**So the order-sensitivity is a property of the ARCHITECTURE, not something
training creates.** The gap on its own is **not** evidence of *learned*
temporal structure. Note the twin is off the figure's axis for a **scale**
reason (0.058 vs 0.578 ordered recovery), **not** because its gap is small —
it is not.

**What training adds is the accuracy itself**: at T=16, ordered recovery goes
from **0.058 (untrained) to 0.578 (trained)** — the untrained model is near
chance.

> **⚑ NUMBER CORRECTED (mac-d, 12:3x, before this was pasted anywhere).** This
> read *"from 0.22 (untrained) to 0.58 (trained) at T=16"*. **0.22 is the T=2
> untrained value; at T=16 it is 0.058** — the sentence paired T=2's untrained
> figure with T=16's trained one. Untrained ordered recovery by T is
> **0.222 / 0.179 / 0.103 / 0.058** (T=2/4/8/16); trained is
> **0.499 / 0.523 / 0.536 / 0.578**. The error **understated our own result**
> (the real lift at T=16 is ~10x, not ~2.6x) but would not have survived a
> reviewer checking it. *"By a comparable amount"* also softened: the twin's
> gap is **larger** in 11/12 cells and **6x larger at T=4**, so "comparable"
> understated it in the direction that flatters us. Shuffle table above
> re-verified against the 8 source shards: **16/16 values exact.**

**Two ways to handle it, your call:**

1. **As written above** — claim only that the trained model's accuracy
   depends on token order. That is true and it is what the reviewer asked
   about. **But be clear-eyed: if a reviewer asks whether an untrained model
   shows the same gap, the answer is "yes, and usually larger."** That is a
   worse conversation to have after the fact than before.
2. **Pre-empt it** by adding one sentence:

```
A randomly-initialised TXC also loses accuracy under shuffling, by a similar or
larger amount, so this order-sensitivity is a property of the architecture
rather than something training creates. What training adds is the accuracy
itself: at T=16, ordered recovery rises from 0.058 to 0.578.
```

**Our recommendation is (2).** It costs one sentence, it is the kind of
disclosure this reviewer explicitly praised ("the honesty about negative
results is appreciated"), and it removes the only follow-up that could
embarrass the claim.
=======
We also ran a random-init control on the shuffle. **It is deliberately not in
the excerpt** — it answers a different question and it is what made the
earlier write-up unreadable.

**Worth knowing before a reviewer asks:** a random-init TXC also loses
accuracy under shuffling, so the *order-sensitivity itself* is architectural;
what training supplies is the accuracy (ordered recovery 0.22 → 0.58 at T=16).
**Nothing in the excerpt claims otherwise** — it claims only that the trained
model's accuracy depends on token order, which is what the reviewer asked
about. Detail: `REBUTTAL_HANDOFF.md` §6.
>>>>>>> 69d6877f2 (Remove the untrained control from the proposal (Han); fix a false claim the check caught)

---

## Provenance

- Numbers: `figs_writeup/tab_sycgen_budget_matched.md` (level) and
  `figs_writeup/tab_sycgen_shuffle_matched.md` (shuffle), both generated from
  `sycgen/results/` — **every value in this file was re-derived from the source
  data and checked, not transcribed.**
- Figure (for the paper/appendix, not the response — no images allowed there):
  `figs_writeup/fig_sycgen_shuffle_matched.png`.
- The shuffle run was **pre-registered before any cell ran**
  (`sycgen/SHUFFLE_MATCHED_CARD.md`), with three instrument gates that all
  passed: the shuffle was verified live against an exact binomial band, the
  pooled arm's gap was verified identically zero, and SAE sparsity was verified
  permutation-invariant.
