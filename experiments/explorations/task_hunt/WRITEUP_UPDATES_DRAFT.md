# WRITEUP updates — DRAFT for mac-local ratification (overnight § 6a)

**Status: DRAFT — not applied.** mac-b, 2026-07-27 ~02:05 London. The WRITEUP
rule is "edit only with a matching receipts row"; every block below cites its
row. Task 1/2/3 licence-format numbers were checked against R22/R27/R28/R29
and are **already correct in the page** (no numeric edits needed — the
verification is itself a § 6a result). What is missing is the **composition
audit certification (R30)** and the mechanism re-attribution that two
existing sparsity notes now inherit. Three blocks, ready to paste.

---

## Block 1 — NEW bullet in § 9 (Methods caveats that travel with every quote)

**Insert after the existing realized-sparsity bullet. Receipt: R30 (ratified;
calib freeze `97fae183a`, 20/20 identity + registered 2-cell prediction
confirmed; `ACTMIX_FORENSICS.md` § 9 corrigendum).**

> - **Composition audit (certification).** The hunt's dictionary family was
>   audited for a ReLU/BatchTopK composition inconsistency raised on the
>   paper side. At every width used on this page the audit is an identity:
>   retrained no-ReLU variants reproduce the shipped arms to
>   |Δ recovery| ≤ 2.2 × 10⁻⁸ with realized-sparsity delta exactly 0.0
>   (R30). Every number on this page is therefore **composition-robust by
>   identity** — no re-run can move it. The compositions genuinely diverge
>   only in thin-positive-pool regimes (deep-selection arms; one diagnostic
>   cell on record), which no claiming cell occupies.

## Block 2 — amend the § 9 realized-sparsity bullet (mechanism attribution)

**Current text (end of the bullet):** "…(an architecture property at this
width; a sensitivity check passes). The temporal-SAE comparison is the clean
one, and Task 2 passes both."

**Proposed replacement (receipt: R30 + calib mechanism finding, LOG
2026-07-26 ~22:20; `ACTMIX_FORENSICS.md` § 9):**

> …(now mechanistically attributed: eval-time threshold pruning of the
> smallest activations, identical under both ReLU/BatchTopK compositions —
> R30 — so the sparsity ratio upper-bounds any flattering and the audit
> moved no margin; a sensitivity check passes). The temporal-SAE comparison
> is the clean one, and Task 2 passes both.

## Block 3 — amend the Task 3 per-token-SAE footnote (same attribution)

**Current text (§ 5, end of "The result" list):** "(Per-token-SAE footnote:
its realized sparsity came out low, 4.5/token — a known signature of this
model size — which flatters the TXC-vs-SAE margin; that is why we lead with
the T-SAE comparison, whose realized budget is clean.)"

**Proposed replacement (receipt: R30; dq quote licence, LOG 2026-07-26):**

> (Per-token-SAE footnote: its realized sparsity came out low, 4.5/token —
> eval-time threshold pruning, identical under both audited compositions
> (R30), so it upper-bounds any flattering of the TXC-vs-SAE margin; that is
> why we lead with the T-SAE comparison, whose realized budget is clean.)

## Not drafted (mac-local's to state)

The § 9 blanket bullet "Every 2026-07-26 result … is pending team
ratification" predates the ratifications of R26/R27 (day-2 close), R28/R29
(salvage close) and R30 — a status refresh is warranted but ratification
status is the orchestrator's voice, not a worker's; flagged only.

*mac-b; DRAFT pending mac-local. Verification note: Task 1 margin passage
(R22 + both caveats + "positive in all seeds" phrasing), Task 2 fresh-seed
passage (R29 L1 numbers, no-pooling phrasing, budget-parity disclosure), and
Task 3 passage (R27 bound, T ≤ 8 licence, tsae-first) all check out against
their receipts as written — no further edits proposed.*
