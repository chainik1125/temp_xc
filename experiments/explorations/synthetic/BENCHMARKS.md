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
| **multilane** | theorem-first — FB-C1 card FB-2 (superposition) | abstract | 3 simultaneous cyclic tones (AC, regime 3) | n/a | ✓ | **POSITIVE** *(spectral 0.79 > post 0.46 ≫ additive family ≈ token ≈ 0 at T=8; memorization-immune \|Ω\|³M³; the sprint's multiband>vanilla headline FAILED its frozen T=8 bar (+0.019 < 0.03) — band advantage is scarcity/coarse-window-bound, and spectral COLLAPSES at k_pos=8 (−0.583 margin flip))* |
| **colored_sources** | theorem-first — FB-C1 card FB-3 (CS-1/CS-2) | abstract | lag-D covariance / direction-recovery-primary (order-2) | n/a | ✓ | **POSITIVE** *(weak realization: CS-1 floor holds over all 261 T≤D cells; the W=D+1 transition is realized at ≤21% of the provable +0.96 ceiling, and the ordering INVERTS the tone benches — txc-pre ≥ spectral > post ≈ floor)* |
| **phasepair** | theorem-first — FB-C1 card FB-1 (phase-vs-power) | abstract | rotation direction within ± pairs (phase-only, regime 3) | n/a | ✓ | **POSITIVE** *(the sharpest dissociation: txc-post reads sign PERFECTLY (1.000 at T=8, all seeds) while spectral is sign-blind at T≤4 — singleton DCT bands admit no quadrature — recovering at T=8 (0.936); additive family ≈ 0 on both components; retro-explains signed_motion's NEGATIVE as a substrate defect, not panel phase-blindness)* |
| **assumption_consequence** | grounded — expansion C1 → C2 (g7 re-exam) | reasoning-trace | directed-transition (AC) | SPEC | ✓ | **NEGATIVE** *(frozen windows>per-token prediction failed: order-1 mirror ⇒ s_i sufficient, per-token reads the directed latent; needs order-2+ mirror to separate archs)* |
| **hedging_drift** | grounded — expansion C1 → C3 (`hier_ar1` mirror) | reasoning-trace | slow-drift (DC) | SPEC | ✓ | **SPLIT** *(per-token R² 0.73 of 0.77 ceiling — drift is ambient per token; window edge ≤ +0.04 with weak T-trend)* |
| **recipe_instruction_phase_runs** | grounded — expansion C3 → C4 → stage-6 #3b re-scope | text-corpus | interaction/equality (**regime-3 residual** over the measured additive floor) | SPEC | ✓ | **POSITIVE** *(the first grounded regime-3 architecture verdict: on the re-scoped residual axis [additive ceiling 0.771 → exact 1.0], only Spectral-TXC exposes the latent — T=2 +0.60/+0.90/+0.96 across d, k-robust to 4, untrained ≈ 0 — TXC-post caps at the ceiling (+0.26 best), all additive families pinned at the DC-leak line (−0.76…−0.86); § 5-r predictions frozen pre-grid in `241845d2`, checked blind — see `bench_record.md`)* |
| **list_item_parallelism** | grounded — expansion C3 (re-freeze) | text-corpus | bursty / self-exciting (AC) | SPEC | ✗ | — *(pending; ⚠ redundant class + weak mirror — low value)* |
| **self_reference_echo** | grounded — expansion C2 (re-filed) | reasoning-trace | bursty / self-exciting (AC) | SPEC\* | ✗ | — *(not planned; redundant, marginal labeler)* |

*Datasource plugins (the `✓`s): `toy_backtracking_selfexcite_d64`,
`toy_changepoint_modes_d64`, `toy_cyclic_{circle,random}_M101_d128`,
`toy_signed_motion_M19_d40`, `toy_assumption_consequence_d64`,
`toy_hedging_drift_d64`, `toy_recipe_instruction_d64`,
`toy_multilane_circle_M101_d24`, `toy_colored_sources_N32_D2_d32`,
`toy_phasepair_M101_d24` (FB-C1). Remaining SPECs
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
| proof-operation-phase-runs-r3 | reasoning · int/eq | **signal REAL, three timescales CONFIRMED** (real-vs-permuted ACF(4) gap 0.056); the `seg_hier_categorical` layer closes lag-2–8 — MI(2) passes for the first time, ACF(4) flips −55% → +21% (marginal fail) — but the preregistered **insertion control** catches the estimator hallucinating on run-permuted streams; C6 needs calibrated segment-composition extraction, not a new family | C5 |
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
ceiling 0.77; nonlinear residual to 1.00 intact) — and the review adopted
re-scope option 1 (primary axis = the regime-3 residual over the measured
additive floor). **Stage-6 #3b (2026-07-23): the re-scoped head-to-head ran
blind and the verdict is POSITIVE** — the text-corpus half of the prize is
now **fully claimed**: a grounded, gate-validated, regime-3 benchmark that
separates architectures (Spectral-TXC linearizes the residual to +0.97;
coincidence-via-squash caps at the additive ceiling; additive families sit
at the DC-leak line). The regime-3 row of the 4-regime table has its first
grounded confirmation: order-2 structure separates window *mechanisms*, not
window *presence*.

## Authoritative sources (this file is the human index over them)

- **Grounded coverage grid** (proposed/abort/SPEC, per cell): [`expansion/LEDGER.md`](expansion/LEDGER.md).
- **Architecture head-to-head** (auto-generated, evaluated benches only): [`REPORT.md`](REPORT.md) + [`registry.py`](registry.py).
- **Program methodology + the evaluated-benchmark index**: [`README.md`](README.md).
- **Per-benchmark detail**: `synthetic/<name>/bench_spec.md` (spec) + `bench_record.md` (arch results, evaluated ones).
- **Living program state**: [`STATUS.md`](STATUS.md).
