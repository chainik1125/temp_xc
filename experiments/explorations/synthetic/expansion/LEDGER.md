# Grounded-benchmark expansion — coverage ledger

The **anti-drift invariant** for the autonomous expansion loop (see
[`README.md`](README.md), the standing pipeline doc). Every candidate temporal
property lands in exactly one `domain × temporal-class` cell. Selection each cycle
must (a) keep a **per-domain floor** (≥⌊N/2⌋ calibrated from each domain) and
(b) **prioritize under-covered cells** (empty / abort-only) over cells that
already hold a PROCEED. Update this file at the end of every cycle and report the
grid — an empty domain is then a visible, prioritized gap, not silent drift.

**Cell status:** `—` empty · `prop` proposed (prereg frozen) · `calib:ABORT` ·
`calib:PROCEED` · `SPEC` (PROCEED graduated to a `synthetic/<name>/` benchmark) ·
`SPEC*` (PROCEED but **provisional** — flagged, not stage-6-runnable).

| temporal-class | reasoning-trace | text-corpus |
|---|---|---|
| **DC-slow-drift** (state persists, slow) | **`SPEC`** — uncertainty-hedging-drift (**mirror VALID C3**: `hier_ar1` menu extension holds the ACF plateau — gate-8 ACF(2) \|err\| 0.003 ≤ 0.033, ACF(4) 0.018 ≤ 0.028, matched ACF(1) 0.013; SPEC*→SPEC, canonical mirror swapped by dated spec amendment → `synthetic/hedging_drift/`) | `calib:ABORT` — topic_switching (composition, labeler inadequate) · `prop` — hedge-to-assertion-drift (C1, unselected) |
| **AC-order-sensitive** (depends on order) | **`SPEC`** — assumption-then-consequence (**g7 re-exam RESOLVED C2**: strict per-sentence labeler, ctx=0 → asym **0.297** (was 0.135), gate-8 PASS, fresh skeptic 5/5 → `synthetic/assumption_consequence/`, canonical mirror = g7 fit) | `calib:ABORT` — question-answer-adjacency (skeptic kill: definitional leakage) |
| **periodic** (rhythmic/cyclic) | `calib:ABORT` ×2 — computation-verification-alternation (C2: peak real, periodic_rate can't make Fano 2.29; **C3 re-freeze with `periodic_hawkes`: gate-8 PASSED (Fano \|err\| 0.10 ≤ 0.46) but skeptic killed on circularity** — the joint period+kernel fit inserts everything measured; any future re-freeze needs a genuinely non-inserted moment, e.g. gap-distribution shape or cross-doc period stability) | `calib:ABORT` — enumeration-cadence (C3: spec_peak **4.10** real ≫ null 1.13, κ=0.56, but gate-8 fano fail 3.52 vs 0.97 — the C2 comp-verif failure replicated in text: enumeration is rhythmic AND bursty; same hybrid-vs-circularity tension as the reasoning cell) |
| **bursty/self-exciting** (clustered events) | **`SPEC`** — backtracking (the hand-run anchor) · **`SPEC*`** — self-reference-echo (re-filed from interaction/equality, C2 review; redundant with backtracking + marginal κ=0.30 → low eval priority) · `prop` — error-correction-cascade (C1, unselected; cell PROCEEDed) | **`SPEC`** — **list-item-parallelism** (**C3 re-freeze PROCEED — the program's first text-corpus SPEC**: ACF(1)=0.52 [0.48,0.56] ≫ N1 hi 0.21, κ=0.64, gate-8 Fano \|err\| 0.163 ≤ 0.781 (±20% rel), skeptic 5/5 → `synthetic/list_item_parallelism/`; **re-filed from interaction/equality**: measured binary run-clustering, `logistic_ar` family) · `calib:ABORT` — quotation-burst (C1 skeptic kill: circular mirror validation) |
| **interaction/equality** (cross-position compare) | **— still unfilled** · `calib:ABORT` ×2 — operator-alternation (C2: NEGATIVE sign falsified) · **proof-operation-phase-runs (C3, categorical recipe: signal REAL — self-match ACF(1)=0.286 ≫ N1 hi 0.047, 5-class marginal [.22,.33,.08,.08,.29], κ=0.59 — but gate-8 MI(2) fail 0.030 vs 0.065: semi-Markov halves the two-step structure)** | **— still unfilled** · `calib:ABORT` ×3 — greeting-signoff-mirror (C2: mis-keyed mirror) · list-item-parallelism (*C3: re-filed → bursty/self-exciting, where it PROCEEDed*) · **recipe-instruction-phase-runs (C3, categorical recipe: signal REAL — self-match ACF(1)=0.479 ≫ N1 hi 0.204, 5-class marginal [.29,.51,.06,.07,.07], κ=0.61 — but gate-8 ACF(4) fail 0.201 vs 0.294: the self-match tail is a plateau semi-Markov can't hold — the categorical analogue of the hedging plateau ⇒ C4 menu gap: hierarchical categorical mirror)** |
| **long-memory** (renewal / heavy-tail) | `calib:ABORT` — goal-restatement-recurrence (C3: gap-CV 1.48 > N1 hi 1.01, gate-8 PASS, but **skeptic kill on composition**: pooled gap-CV inflated by cross-trace rate mixture; perturbed margin thin (1.14 vs N2 hi 1.06); κ=0.42 near floor) | `calib:ABORT` — pronoun-referent-recurrence (C2: gap-CV dies at the noise floor) |

## Notes / provenance

- **backtracking** is the hand-run anchor; the automation imitates it.
- **Cycle-1 lessons became design-time gates 7–8** (no-leakage labeler;
  non-fitted-moment mirror) — and Cycle 2 shows them working *before* the
  skeptic: 3 of 5 C2 aborts were cheap gate-8 kills (skeptic skipped).
- **Gate-8 tolerance-scaling lesson (C2):** tolerances preregistered as raw
  absolutes get mis-scaled when the statistic's magnitude is unknown
  (list-item-parallelism died at 4% relative error). Cycle 3 should preregister
  tolerances **relative to the statistic's magnitude or null-band width**.
- **Mirror-menu gap (C2):** two real phenomena the menu can't generate —
  long-memory plateaus (per-sequence levels / slow regimes; hedging) and
  periodic+bursty hybrids (verification). Both are concrete Appendix-B
  extension proposals for Cycle 3.
- **The g7 re-exam vindicated the strict-labeler discipline**: removing the
  relational clause *strengthened* the assumption→consequence asymmetry 2.2×.
- **C2 review (mac-local) — measured-class re-filing (new loop rule, README).** A
  candidate's cell is its **measured** temporal class, not the proposed one.
  `self-reference-echo` was *proposed* interaction/equality but *measured* pure
  self-excitation (ACF/Fano/excite + `logistic_ar` mirror = backtracking's
  process), so it re-files to **bursty/self-exciting** (redundant, marginal
  labeler) and **interaction/equality is empty again — still the top target.**
- **interaction/equality is structurally hard under gate 7** (equality is
  inherently relational, gate 7 forbids relational labels). The gate-7-clean path
  (README): a **categorical per-sentence content label** (sub-goal / operation /
  claim-topic) whose **equality-adjacency `[c_t=c_{t-1}]`** is the *measured*
  signature — exactly how the synthetic changepoint mode works. Two binary-label
  attempts (self-reference-echo, operator-alternation) both measured as clustering
  instead.
- **C3 mirror-menu resolution + the next gap.** The two C2 gaps were built and
  validated (`hier_ar1` rescued hedging; `periodic_hawkes` passed gate 8 on
  verification) — but C3 exposed two successors: **(i)** real categorical
  streams hold self-match *plateaus* (recipe ACF: 0.48→0.28 over 8 lags) that
  dwell+jump processes can't generate — a **hierarchical categorical mirror**
  is the C4 extension both int/eq aborts point at; **(ii)** a mirror rich
  enough to fit a rhythmic+bursty stream (periodic_hawkes) leaves nothing
  non-inserted for gate 8 — future periodic cards must preregister a moment
  the hybrid does NOT fit (gap-distribution shape, cross-doc period
  stability) or the skeptic will (rightly) kill on circularity.
- The abstract benches (signed_motion, frequency) are out of this ledger's scope.

## Cycle log

- **Cycle 1 — 2026-07-14 (runpod, autonomous).** 10 cards frozen (5+5); 4
  calibrated (2+2): assumption-then-consequence PROCEED→SPEC,
  uncertainty-hedging-drift PROCEED→SPEC, question-answer-adjacency ABORT
  (skeptic: leakage), quotation-burst ABORT (skeptic: mirror circularity).
  Spend $9.55/$25. Review (mac-local): approved; gates 7–8 added;
  assumption→SPEC* provisional; both mirrors to be gate-8 rechecked.
- **Cycle 3 — 2026-07-19 (runpod, autonomous).** Menu extended per the C2
  review (`hier_ar1`, `periodic_hawkes` + harness tests); the **uniform
  relative gate-8 rule** preregistered (±20% of held-out magnitude + floors);
  4 categorical interaction/equality cards frozen under the gate-7-clean
  recipe (blind selection picked 2); 6 calibrated (3/domain) + the hedging
  re-fit rider. Verdicts: **list-item-parallelism-r2 PROCEED→SPEC** (first
  text-corpus SPEC; re-filed to bursty/self-exciting by measured class);
  **hedging rider PASS → hedging-drift SPEC*→SPEC** (hier_ar1 holds the
  plateau: ACF(2) err 0.003, ACF(4) 0.018). 5 ABORTs: computation-
  verification-r2 (hybrid mirror gate-8 PASSED but skeptic circularity kill —
  joint fit inserts everything measured), proof-operation-phase-runs (gate-8
  MI(2): semi-Markov too short-memory), recipe-instruction-phase-runs (gate-8
  ACF(4): categorical plateau), enumeration-cadence (gate-8 fano 3.52 vs
  0.97 — rhythmic+bursty, the C2 comp-verif failure in text), goal-
  restatement-recurrence (skeptic composition kill: cross-trace gap mixture).
  Spend **$8.20/$25**. **interaction/equality target NOT met** (both
  categorical attempts were real signals killed on mirror fidelity — the
  measurement recipe works; the categorical menu lacks a long-memory
  process). Systemic C3 lessons for C4: (i) a **hierarchical categorical
  mirror** (per-doc phase propensities + jump chain — the categorical
  hier_ar1) would plausibly convert BOTH int/eq aborts; (ii) periodic
  phenomena that are also bursty face a **hybrid-vs-circularity tension** —
  the hybrid passes gate 8 but the skeptic (correctly) demands a non-inserted
  moment; preregister gap-shape / cross-doc period stability as the gate-8
  moment before any further periodic re-freeze; (iii) the relative-tolerance
  rule worked as intended (no spurious tolerance kills; enumeration's err was
  363% of magnitude — a genuine mismatch, not mis-scaling).
- **Cycle 2 — 2026-07-14 (runpod, autonomous).** 4 new interaction/equality
  cards frozen (2+2) under gates 7–8; 6 calibrated (3+3, deterministic
  selection) + the g7 re-exam rider + the gate-8 recheck rider. Verdicts:
  self-reference-echo **PROCEED→SPEC**; operator-alternation ABORT (sign
  falsified); computation-verification-alternation ABORT (gate-8: bursty≁periodic
  mirror); greeting-signoff-mirror ABORT (gate-8: zero-MI mirror);
  list-item-parallelism ABORT (gate-8 near-miss, re-freeze candidate);
  pronoun-referent-recurrence ABORT (noise floor). Riders:
  **assumption-consequence g7 re-exam → SPEC upgraded** (asym 0.297, 2.2× the
  contextual labeler); **hedging mirror gate-8 recheck FAILED** (+ preregistered
  semi-Markov attempt also failed) → SPEC downgraded to SPEC* pending a
  hierarchical-AR(1) menu extension. Spend **$14.06/$25**. **Text-corpus PROCEED
  target NOT met** (0/3 — two mirror-fidelity kills + one noise-floor kill; the
  underlying text signals were real in 2 of 3 cases). Next cycle should target:
  re-freeze list-item-parallelism (scaled tolerance) + computation-verification
  (hybrid mirror) for the text/periodic wins; the hierarchical-AR(1) +
  periodic-Hawkes menu extensions; relative gate-8 tolerances; long-memory ×
  reasoning (goal-restatement, frozen) and periodic × text (enumeration-cadence,
  frozen).
