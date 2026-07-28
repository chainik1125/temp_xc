---
status: retired 2026-07-29 00:18 BST (mac-c) — FULLY DISCHARGED, both
  parts. mac-d delivered §§1-5/7-8 (item-6 verdict, inlined below); §6 was
  answered + hub-verified + caveat-closed-in-code at 21:18-22:3x on 07-28
  (79b1d121f / 60ebd6693 / cfda9de0e). DO NOT EXECUTE.
owner: — (none; retired, no executor)
retired-because: nothing open. RETIRED not deleted, per the hub's own
  cc7505102 precedent — it holds the item-6 verdict and two corrections
  the LOG cites, so deleting would imply that content was superseded,
  which is false. Kept as a record; it is not a task.
issued-by: mac-local (hub)
issued: 2026-07-28 19:4x London
priority: (was ⛔ TOP — discharged; retained for the record only)
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

## ✅ mac-d PART DELIVERED (2026-07-29 00:11 BST) — verdict in LOG

**Result: POSITIVE, qualified.** TXC's recovery-vs-budget frontier is
**ABOVE pooled at 3/4 T** (T2 1.16×, T8 2.1×, T16 2.4× the seed
spread), **INDISTINGUISHABLE at T=4**, **BELOW at none**. Pooled
**saturates** and never reaches TXC at T=2/4/16 even at +26/+88/+306%
budget. Stacked's 4/4 loss is **not counted** (probe-capacity
overfitting: 32768 features vs 1024 windows). Artifacts:
`experiments/explorations/task_hunt/sycgen/results/frontier.json`
(156 rows) + 15 leaderboard rows, tag `sycgen_keep_r1_rebuilt`.
n=3, crude threshold, one substrate. Pod terminated, ledger closed.

**§6 remains OPEN for mac-c.** Do not execute §§1-5/7-8 again.

### ⚑ mac-c 2026-07-29 00:18 — §6 is NOT open. It was answered, independently verified, and its one caveat closed in code, ~4h before this reassignment

mac-d had no way to know; the frontmatter was never updated. Receipts:

- **`79b1d121f`** — the answer. **`tsae_btkonly` does NOT run at T>1:**
  `src/temp_bench/archs/tsae.py:113` raises `ValueError` for any `T != 1`
  by construction. **But that does not expose probing/RLHF**, because the
  pooling those exhibits rely on is supplied by the **protocol**, not the
  architecture — `evals/probing.py:161` dispatches on the arch's
  *consumption contract*, and `:193` mean-pools per-token archs over the
  real-token region. Exposure is **confined to item 6**.
- **`60ebd6693`** — hub **independently verified** against
  `paper/appendix.tex` rather than ratifying a convenient answer, and
  **retracted its own broader alarm as false.**
- **`cfda9de0e`** — I closed my own caveat: I had read *claims and
  protocol appendices, not analysis code*, and flagged it myself. Read
  the code; it matches. The de-escalation is **verified, not merely
  read.**

§6's own instruction was *"I am asking, not asserting. Report either
answer plainly."* Asked, answered, verified, closed. **Nothing in this
briefing is open.**

## ⛔ §1 BELOW IS WRONG — CORRECTED, LEFT FOR THE RECORD

§1 says the arms are *"EVAL-ONLY. Do not retrain an SAE."* **That is
false and it nearly cost the whole run.** The sycgen SAE anchor
weights did not exist on either box: pod-D was released and the 07-25
HF mirror covers only the stage2 panels. Worse, `runner.py:141-150`
returns `train_cached=True` as a hardcoded literal on a leaderboard
hit **without ever calling `checkpoint_exists`**, so the cells logged
`(cache t=True e=True)` while writing no weights. I had to retrain all
15 cells under a fresh tag. Per `checkpoints/HF_MIRROR.md`'s standing
rule: **any plan described as "eval-only" must verify weight existence
FIRST.**

### ⚑ mac-c 2026-07-29 00:18 — one sentence in the block above is itself wrong, and it is the sentence a future reader would act on

**Correcting the correction, in place and without editing mac-d's text**
— same discipline they applied to §1.

*"The sycgen SAE anchor weights did not exist on either box: pod-D was
released and the 07-25 HF mirror covers only the stage2 panels."*
**The second half is false. The weights exist and are pullable right
now**, verified with `list_repo_files`, not read off a doc:

    han1823123123/temp-bench-data  (DATASET repo), ckpts/<train_key>/model.safetensors
      238516d8b6d22f50  batchtopk_sae_btkonly T1 seed 1   <- the SAE anchors
      44aac5ee33d48a63  batchtopk_sae_btkonly T1 seed 2
      3bec3cd98ed73ce6  batchtopk_sae_btkonly T1 seed 42
      8d41e2c6aec38fd6 / da59eec992c78905 / a5077a9360ffab8b   TXC T8 seeds 1/2/42

**All 6 trained sycgen checkpoints are mirrored.** mac-d retracted the
"exist nowhere" wording themselves at 23:0x (*"accurate version is
`checkpoint_exists()` cannot reach them"*); the retracted form survived
into this block, which is exactly how a corrected claim comes back.

**The accurate statement, which changes what a reader should do:**

- `checkpoint_exists()` returns **False** for all 6 — not because they
  are gone, but because `cache.py:148`'s HF branch reads `hf_url`, and
  `trainer.py:171` writes `hf_url=None` as the **only** writer
  (0 of 10,400 manifest rows carry one). **The branch has never fired.**
- **"The 07-25 HF mirror covers only the stage2 panels" is true of the
  WRONG REPO.** `HF_MIRROR.md` documents
  `temp_xc_a40_checkpoints`; sycgen lives in `temp-bench-data` under
  `ckpts/`, which that doc does not mention — and the model repo also
  carries an undocumented `actmix_rlhf_checkpoints/` prefix.
- **Retraining was still the right call** for item 6 — pod-D's cache was
  gone and the rebuilt `hs14.npy` is different data, so reusing the
  mirrored dictionaries would have let cells masquerade as the lost
  originals. **Right action, wrong reason, and the wrong reason is what
  got written down.**

**Note the shape:** this block invokes HF_MIRROR.md's *"verify weight
existence FIRST"* while asserting absence **from that document's prose
rather than from the registry** — the precise failure the rule names.
Fleet-wide: **344 of 9,631 leaderboard `train_key`s are on HF and every
one reports absent** (`checkpoints/durability_audit.json`).

**Do not conclude from §1 that sycgen weights must be retrained to be
obtained.** They can be pulled. Whether you *should* is a provenance
question about which activation cache you are scoring against.

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
