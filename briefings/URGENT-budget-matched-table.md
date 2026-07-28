---
status: active
owner: mac-d (primary executor) + mac-c (§6 only, $0)
issued-by: mac-local (hub)
issued: 2026-07-28 19:4x London
priority: ⛔ TOP — ahead of the geometry matrix, the rescue lane and the
  clew sweep. This is a live threat to a DELIVERED exhibit.
---

# ⛔ URGENT — the budget-matched TXC vs pooled-SAE vs stacked-SAE table

Han: ***"we need to get the budget matched comparison TABLE DONE ASAP."***

Dmitry's agent ran the baseline we never ran and **TXC loses at every T**:

    T        TXC          pooled SAE     stacked SAE
    2    .501 ± .015     .506 ± .009    .506 ± .009
    4    .529 ± .011     .537 ± .006    .544 ± .002
    8    .543 ± .007     .563 ± .022    .565 ± .028
    16   .600 ± .010     .633 ± .006    .619 ± .003

**Their TXC column is OUR column** (.498/.524/.541/.592) — this is not a
measurement dispute, it is a missing-baseline dispute, and we are the
ones who were missing it. Verified: the sycgen retrain ran **three**
archs and the only SAE is `batchtopk_sae_btkonly` **at T=1**. We have no
SAE that gets to use the window.

---

## 1. The fast path — these arms are EVAL-ONLY. Do not retrain an SAE.

**Pooled and stacked SAE are post-hoc transforms of the already-trained
per-token SAE.** Encode with the T=1 SAE, then over each T-window:

- **pooled** — mean (or sum) the per-token feature vectors → `d_sae` dims
- **stacked** — concatenate them → `T × d_sae` dims

Nothing needs training that is not already trained. That is why Dmitry's
agent produced this quickly, and it is why this table can exist today.

## 2. The budget match — constrain the SAE arms, never inflate TXC

Measured `l0_per_window` from our own rows:

    TXC:  T2 5.53-5.98   T4 6.27-6.54   T8 6.73-7.25   T16 7.81-7.83
    SAE (T=1, trained):  4.37-4.75 per token

A pooled SAE over T tokens can carry **up to ~4.5·T nonzeros per
window** — ≈72 at T16, roughly **9× the TXC's 7.8**.

**⚑ Correction to my own earlier framing: that 9× is an UPPER BOUND, not
a measurement.** Pooling collapses features that fire at several
positions, so the realized union may be well under 4.5·T. **Measure it —
do not quote my bound.**

**⚑ SUPERSEDED — HUB RULING 19:5x, mac-d's objection is upheld.**

I originally specified: top-k truncate the pooled/stacked vector to the
TXC's realized `l0_per_window`. **mac-d showed the arithmetic before
spending any pod hours (`e3c16764d`) and it defeats the prescription:**
constraining the SAE to TXC's per-window budget forces **0.49 l0/token
at T16** — under one feature per two tokens, almost certainly
degenerate. **The two arms' budgets scale differently in T, so matching
per-window necessarily unmatches per-token, and vice versa.** There is
no single "matched point"; my brief asked for a quantity that does not
exist.

**A table where we hobble the baseline into degeneracy and then win is
worth less than no table at all.**

**RULING — do mac-d's version: report the recovery-vs-budget FRONTIER.**

- **Sweep k on both arms** and plot recovery against **realized
  `l0_per_window`** (measured per cell, never nominal).
- **Plot the as-run points on the same axes**, labelled — Dmitry's
  result must be locatable on the figure, not replaced by it.
- **The verdict is dominance, not a point comparison:** does the TXC
  curve lie above the pooled/stacked curves anywhere in the budget
  region of interest, and in particular at equal `l0_per_window`?
- **If TXC is dominated across the whole frontier, item 6 is a
  negative** and we say so — that is a cleaner and more honest result
  than any single matched point could have given.

This also puts item 6 in the same frame the probing section already
uses (*"probe-budget-dependent, no monotone window win at any k"*),
which is house style rather than a special pleading invented for this
challenge.

**mac-d: this is a decision, not a preference — proceed.** You raised
it as "stated as preference not decision"; the hub is ruling, and the
ruling is yours.

**Every cell must report its realized `l0_per_window` as a receipt.**
A budget-matched table with no realized-l0 column is not evidence.

## 3. What the table must contain

Rows `T ∈ {2,4,8,16}`, **3 seeds**, mean ± sd, and **both budget
conditions side by side**:

| T | TXC | pooled (as-run) | stacked (as-run) | pooled @matched l0 | stacked @matched l0 | l0/win TXC | l0/win pooled |
|---|---|---|---|---|---|---|---|

**The as-run columns are Dmitry's result and they STAY IN THE TABLE.**
Dropping them and showing only the matched version would look exactly
like hiding the inconvenient number, and would deserve to.

Also carry the two controls the exhibit already has, so the table is
self-contained: the **per-token anchor** (0.4819) and the **untrained
twin** (≤0.227).

## 4. Pre-registration — write this into the card BEFORE you run it

**If TXC still loses at matched budget, item 6 is a NEGATIVE and we
report it as one.** No re-cutting, no "but at T8", no post-hoc arm
selection. Under the prime directive — *a sound verdict, never a win* —
a gold task that dissolves under the right baseline is precisely what
this program exists to catch.

State in the card, before numbers exist: what result would count as TXC
winning, what would count as a null, and what would count as a loss.

## 5. ⚑ FIRST BLOCKER — check this before anything else

**The pods are gone** (mac-d terminated the fleet; 0 pods, $0/h) and the
**ckpt mirror stamp is 02:0x while the sycgen retrain ran ~03:40–04:30**
— so the sycgen checkpoints may never have been mirrored to HF. The
retrain results are `eval_extra`-namespaced and **carry no `train_key`**,
so they cannot be looked up the usual way. `checkpoints/` holds 20 dirs
locally.

**Step 0: establish that the trained TXC and SAE weights still exist.**
If they do → this is an eval-only job, hours not days. If they do not →
say so immediately and loudly, because the cost changes completely and
Han needs to know before he plans around this table.

## 6. mac-c — one $0 question, and it may be bigger than item 6

Do not drop the geometry lane; this is a single check.

**Does the probing / RLHF `tsae_btkonly` baseline actually run at T>1,
and at a matched budget?** If it does, the paper sections are defended
against this objection and only the hunted task is exposed. **If it does
not — if T-SAE is also effectively a per-token comparator — then the
same missing-baseline criticism applies to the headline probing and RLHF
exhibits**, and that is a far larger problem than item 6.

I am asking, not asserting. Report either answer plainly.

## 7. Venue

Local workers, per Han. Prefer local/MPS if the eval fits — it is an
encode-and-probe job over cached activations, not a training run. If a
pod is genuinely faster, spin one (`mac-d-bmatch-0728`), terminate at
lane end, API-verify, ledger both ends.

## 8. Acceptance gate

`figs_writeup/tab_sycgen_budget_matched.md` + a `RESULT.md` carrying the
pre-registration, the realized-l0 receipts, and a one-line verdict in
plain words. Then `REBUTTAL_HANDOFF.md` §6's ⛔ block gets replaced by
the outcome — **whichever way it falls.**

**Delete this file when the table lands.**
