# Benchmark registry — the one-stop index

The single human-facing index of **every** benchmark in the program — the
abstract/original suite **and** the grounded-expansion program — with
unambiguous status. It exists because "registered" and "convincing" each mean
several different things, tracked in different places; this table joins them.

**Hand-maintained.** The authoritative machine sources are cross-linked below;
update this file on every expansion cycle and every stage-6 run.

## How to read the columns

- **Provenance** — which generator produced it (README § "The two generators,
  one substrate"): `grounded` = PhenomenonBench (data-first, measure→mirror) ·
  `theorem-first` = FreqBench (constructed, proven ceilings) · `hybrid` = mixed.
- **Spec status** — measure→mirror *validity* (is it a sound benchmark?):
  `SPEC` graduated & valid · `SPEC*` provisional (real but flagged) · `ABORT`
  tried, didn't clear the gates · `prop` frozen idea, not yet calibrated ·
  `n/a` theorem-first construct (its validity lives in proofs, not
  measure→mirror).
- **Reg.** — *framework-registered*: `✓` has a datasource plugin in
  `configs/data.yaml` (runnable in the B×A grid) · `✗` spec only, not yet built.
- **Arch verdict** — the *head-to-head result* (`REPORT.md`): POSITIVE /
  NEGATIVE / SPLIT · `—` not yet run.

These are independent: `signed_motion` is fully **registered + valid** yet its
arch verdict is **NEGATIVE** (no arch recovers the sign). "Registered" ≠
"convincing"; "aborted" (measure→mirror) ≠ "negative" (architecture).

---

## A. Live benchmarks (valid — a real or staged benchmark)

| benchmark | provenance | domain | class (DC/AC lens) | spec | reg. | arch verdict |
|---|---|---|---|---|---|---|
| **backtracking** | grounded — hand-run measure→mirror (Ward traces) | reasoning-trace | bursty / self-exciting (AC) | SPEC | ✓ | **POSITIVE** |
| **changepoint** | hybrid — topic dwell measured; emission constructed | abstract emission | change-point / equality (DC+AC) | SPEC (semi) | ✓ | **SPLIT** |
| **frequency** | theorem-first — Dmitry FreqBench port | abstract | periodic / cyclic-tone (AC) | n/a | ✓ | **POSITIVE** |
| **signed_motion** | theorem-first — FreqBench *ac_sign*, forked w/o its proofs | abstract | order-sensitive step (AC) | n/a | ✓ | **NEGATIVE** |
| **assumption_consequence** | grounded — expansion C1 → C2 (g7 re-exam) | reasoning-trace | directed-transition (AC) | SPEC | ✓ | **NEGATIVE** *(frozen windows>per-token prediction failed: order-1 mirror ⇒ s_i sufficient, per-token reads the directed latent; needs order-2+ mirror to separate archs)* |
| **hedging_drift** | grounded — expansion C1 → C3 (`hier_ar1` mirror) | reasoning-trace | slow-drift (DC) | SPEC | ✓ | **SPLIT** *(per-token R² 0.73 of 0.77 ceiling — drift is ambient per token; window edge ≤ +0.04 with weak T-trend)* |
| **recipe_instruction_phase_runs** | grounded — expansion C3 → C4 (`hier_categorical` re-freeze) | text-corpus | interaction/equality (**regime 2/3-mixed** — measured at § 8) | SPEC | ✓ | — *(stage-6 § 8 STOP, no grid: the equality-variant gate found raw-linear access to `e_t` at 0.61 per-token / 0.72 window ≫ chance — a class-conditional continuation-rate leak the frozen regime-3 claim missed; nonlinear-only residual 0.77→1.00 is real; awaiting review re-scope — see `bench_record.md`)* |
| **list_item_parallelism** | grounded — expansion C3 (re-freeze) | text-corpus | bursty / self-exciting (AC) | SPEC | ✗ | — *(pending; ⚠ redundant class + weak mirror — low value)* |
| **self_reference_echo** | grounded — expansion C2 (re-filed) | reasoning-trace | bursty / self-exciting (AC) | SPEC\* | ✗ | — *(not planned; redundant, marginal labeler)* |

*Datasource plugins (the `✓`s): `toy_backtracking_selfexcite_d64`,
`toy_changepoint_modes_d64`, `toy_cyclic_{circle,random}_M101_d128`,
`toy_signed_motion_M19_d40`, `toy_assumption_consequence_d64`,
`toy_hedging_drift_d64`, `toy_recipe_instruction_d64`. Remaining SPECs
without a plugin: `list_item_parallelism`, `self_reference_echo` (both
deliberately unbuilt).*

**Stage-6 outcome (2026-07-22):** both grounded benches built, gated, and run
through the uniform fair-backbone grid (495 cells each, 0 failures) with the
frozen predictions checked blind — see the two `bench_record.md`s. Both
verdicts are "the substrate doesn't separate architectures" findings: the
order-1 AC mirror collapses the directed dependency into the current state,
and the ambient DC magnitude is per-token-readable. The *next* discriminating
benchmarks need order-2+ (AC) or integration-required (DC) structure.
`list_item_parallelism` stays low priority (redundant class, weak mirror).

---

## B. Tried & set aside (the "tried the idea, wasn't convincing" record)

Every candidate that was calibrated and did **not** graduate — with the specific
gate it failed. An ABORT is a *success* of the process (prime directive), not a
failure; kept here so the same idea isn't re-tried blindly.

| candidate | cell (domain · class) | why set aside | cycle |
|---|---|---|---|
| topic_switching | text · change-point | measure-stage: 82% per-doc *composition*, not order; labeler inadequate | pre-expansion |
| question-answer-adjacency | text · AC | skeptic: **definitional labeler leakage** (ANSWER requires a preceding Q → circular) | C1 |
| quotation-burst | text · bursty | skeptic: **circular mirror** (validated only on its fitted moment) | C1 |
| operator-alternation | reasoning · int/eq | preregistered NEGATIVE sign **falsified** (real is +clustering, not alternation) | C2 |
| greeting-signoff-mirror | text · int/eq | gate-8: mis-keyed mirror (zero MI vs real 0.027) | C2 |
| pronoun-referent-recurrence | text · long-memory | dies at the **noise floor** (perturbed falls inside the null band) | C2 |
| computation-verification-alternation | reasoning · periodic | gate-8 mirror fail (C2), then skeptic **circularity** on the over-expressive hybrid (C3) | C2/C3 |
| enumeration-cadence | text · periodic | gate-8: rhythmic **and** bursty; periodic mirror can't hold the Fano | C3 |
| proof-operation-phase-runs | reasoning · int/eq | **signal REAL** (self-match ACF ≫ nulls); gate-8 MI(2) fail — categorical *plateau* the menu can't make | C3 |
| recipe-instruction-phase-runs | text · int/eq | **signal REAL**; gate-8 ACF(4) fail — *rescued in C4 by the `hier_categorical` re-freeze → SPEC (see section A)* | C3 |
| proof-operation-phase-runs-r2 | reasoning · int/eq | **signal REAL**; even `hier_categorical` fails both hardened gate-8 moments — reasoning phase streams hold **three timescales** (run/segment/doc); C5 needs a segment-level regime layer | C4 |
| goal-restatement-recurrence | reasoning · long-memory | skeptic: **composition** (cross-trace rate mixture); thin margin, κ near floor | C3 |

*Frozen ideas never calibrated (prop):* `hedge-to-assertion-drift`,
`error-correction-cascade` — cards frozen in C1, never selected.

---

## Where the prize stands

**interaction/equality is half-claimed (C4).** The text-corpus cell holds the
program's first int/eq SPEC — `recipe_instruction_phase_runs`, also its first
**grounded regime-3 candidate** — after the C4 `hier_categorical` mirror
passed the hardened two-moment gate-8 exactly where the C3 semi-Markov died.
The reasoning-trace cell stays open with a sharper diagnosis: proof-phase
streams hold **three timescales** (run / segment / doc) and even the
doc-level hierarchy fails; the C5 build is a **segment-level regime layer**,
to be validated under the hardened gate so added expressiveness stays
structurally checked. **Stage-6 update (2026-07-22):** the new SPEC's
equality-variant STOP-gate *fired* — the grounded dwell heterogeneity leaks
`e_t` into raw-linear readouts (0.61 per-token / 0.72 window vs additive
ceiling 0.77; nonlinear residual to 1.00 intact), so the bench is
regime-2/3-mixed as frozen and the grid was withheld pending a review
re-scope (re-normalize the primary axis to the additive ceiling, or demote
the claim — options in the bench record). The prize's text-corpus half is
therefore *valid as a phenomenon* but **not yet a certified regime-3
architecture test**.

## Authoritative sources (this file is the human index over them)

- **Grounded coverage grid** (proposed/abort/SPEC, per cell): [`expansion/LEDGER.md`](expansion/LEDGER.md).
- **Architecture head-to-head** (auto-generated, evaluated benches only): [`REPORT.md`](REPORT.md) + [`registry.py`](registry.py).
- **Program methodology + the evaluated-benchmark index**: [`README.md`](README.md).
- **Per-benchmark detail**: `synthetic/<name>/bench_spec.md` (spec) + `bench_record.md` (arch results, evaluated ones).
- **Living program state**: [`STATUS.md`](STATUS.md).
