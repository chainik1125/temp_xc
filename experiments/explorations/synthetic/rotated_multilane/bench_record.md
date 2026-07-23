# Rotated multilane (FB-4) — bench record

**Status: ABORT at T2 (symmetry-triviality / redundancy) — 2026-07-23,
runpod-b, FB-C2 Phase 3. The abort is the pre-registered expected outcome of
the frozen card and is double-witnessed (T2 decision rule + skeptic). An
ABORT is a success of the process (prime directive): the acid-test question
the card carried is real, but this construction's knob is provably inert.**

Frozen card: [`../freqbench/cards/FB-4.md`](../freqbench/cards/FB-4.md)
(committed pre-build `adc6bb28`, mac-local's prediction directions verbatim +
the § 3 absorption obligation added at freeze). Build:
`multilane_tones_rotated` + `toy_multilane_rotated_M101_d24` + contract tests
(`ca2cebac`). Gates committed pre-run (`6e627593`, LOOP T3 strict
commit-then-run). No uniform grid was spent (that is what T2 is for).

## 1. What was built

FB-2's generator composed with ONE fixed Haar-random `Q ∈ O(24)`
(rotation_seed=777, shared across data seeds), labels untouched, exposed
ground truth (planes, codebook) rotated consistently. Contract tests pin:
Q orthogonal and seed-independent; all fields exact Q-images; per-lane
oracle decisions *identical* to base; per-token means degenerate.

## 2. The absorption theorem (why the knob is inert)

FB-2's embedding isometry `P` is Haar-random **and re-drawn per data seed**
(the runner passes the run seed into the generator). Left-invariance of the
Haar measure gives `Q·P =d P` for any fixed `Q`, so the composed generator is
**distribution-identical to FB-2, jointly over data and every exposed ground
truth**. Every panel statistic — trained, untrained, per-cell — therefore has
identical distribution; the FB-2 reference numbers (e.g. untrained spectral
+0.298) are already means over three independent embedding draws.

## 3. Gate evidence

- **T1 (`gating.py`) PASS** after one amendment
  ([`results/rotated_multilane_gating_stats.json`](results/rotated_multilane_gating_stats.json)):
  P5 restated exact (oracle decision equality, zero tolerance; oracle matches
  FB-2's recorded 0.42/0.75/0.906 at T=2/4/8); per-token floors at chance,
  falsifier T=1 recovery 0.0007 (clean). **Amendment (own commits, first-pass
  FAIL preserved at `d9e00a5b`, re-key at `c5e2554c`):** the window-concat
  linear floor first ran against FB-2's absolute bar and read 0.115–0.137;
  the diagnostic found *numerically identical* values on the unrotated base
  (a linear probe is exactly invariant under an orthogonal feature map) — a
  substrate-level variance leak this probe's sample size surfaces equally on
  FB-2 (P2 bounds means only). Re-keyed to the card's actual obligation:
  rotation-invariance of the floor (paired identical-probe gap ≤ 0.005;
  measured ≤ 1e-4). **Datum left for the program:** FB-2's raw-window-linear
  "≈ chance" reading is probe-protocol-conditional at the margins
  (0.10 → 0.13 with a larger-sample probe).
- **T2 (`t2_battery.py`) → ABORT_T2_SYMMETRY** per the pre-registered
  decision rule
  ([`results/rotated_multilane_t2_stats.json`](results/rotated_multilane_t2_stats.json)):
  - *Arm A* (8-seed ensembles, permutation two-sample): coordinate kurtosis
    p=0.37, plane–coordinate alignment p=0.27 — no separation. (Disclosed:
    the third statistic, per-channel DCT high-band energy, is analytically
    Q-invariant — its 4e-11 agreement is a wrapper-purity check, not an
    alignment probe.)
  - *Arm B* (canonical runner, FB-2 anchor cells, seeds {1,2,42}): untrained
    spectral **0.290 vs FB-2 0.298** (inside band — the frozen "+0.298 → ≈0
    collapse" direction is REFUTED); trained spectral 0.794 vs 0.794; post
    0.464 vs 0.461. Nine cells, all inside the FB-2 seed bands.
  - Bag control on rotated data: 0.39 balacc vs oracle 0.906 (order route
    required, as FB-2). Memorization budget and shuffle semantics inherited
    (basis-independent; card § 5).
- **Skeptic (Fable 5) — ABORT confirmed 5-item**
  ([`../freqbench/results/skeptic_verdict_FB-4.json`](../freqbench/results/skeptic_verdict_FB-4.json)):
  kills exactly `b_triviality` + `d_redundancy`; `a_proof_circularity` PASS
  ("the absorption argument is sound and non-circular");
  `e_substrate` PASS (amendment judged an honest re-key). Raw persisted
  pre-parse.

## 4. Scoring the frozen directions (what the card decided anyway)

| frozen direction (mac-local) | outcome |
|---|---|
| per-token / stacked / pre stay ≈ 0 | HELD (trivially — P1/P2 rotation-invariant; gating floors at chance) |
| spectral untrained +0.298 → ≈ 0 | **FAILED** — 0.290, inside FB-2's band. The spectral access prior is *temporal* (DCT-over-τ); a spatial rotation cannot touch it |
| trained spectral recovery = open question | **Decided by theorem, not bench**: recovery is guaranteed by absorption (measured 0.794 = FB-2's 0.794). The intended "subtype rule survives" reading is NOT licensed — the test had no power against the alternative |
| falsifier: any arch > 0.1 at T=1 | did not fire (max 0.0007) |

## 5. What survives for the program (the salvage)

1. **The acid-test question stands unanswered**: is spectral's power/equality
   dominance generic order-2-even conversion or DCT-alignment? The live knob
   is **temporal** — a fixed orthogonal mixing of the within-window time
   basis (which destroys tone structure in the DCT frame while preserving
   order-2 information content) — a candidate **FB-5**, deliberately NOT
   frozen here (briefing hard line: no cards beyond FB-4); left for
   mac-local review.
2. **A card-design checklist item** (skeptic's note, proposal to the program
   — requires mac-local sign-off, not adoptable in-session): for any
   generator whose embedding is Haar-random and seed-re-drawn, ANY fixed
   spatial orthogonal knob is provably inert — kill such cards at freeze.
3. **The probe-protocol datum** on FB-2's raw-window-linear floor (§ 3
   amendment note).

## 6. Review (2026-07-23, mac-local) — APPROVED; ABORT stands

The absorption theorem is sound; the inert construction was mac-local's
card-design defect, and the frozen collapse direction is rightly scored
REFUTED. Gate amendment verified genuine (exact orthogonal-invariance of
linear probes, proven by identical base readings; commit pair fully
diff-visible under the strict rule). The skeptic's checklist item is
ADOPTED into LOOP.md card item 1 as the **non-absorption obligation**.
The salvage carries forward: the acid-test question goes to FB-5
`permuted_tones` (temporal knob; `briefings/freqbench-fb5.md`). Audit:
`../freqbench/PORT.md` § I review.
