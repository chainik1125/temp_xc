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
| **interaction/equality** (cross-position compare) | **— still unfilled** · `calib:ABORT` ×3 — operator-alternation (C2: NEGATIVE sign falsified) · proof-operation-phase-runs (C3: signal REAL but gate-8 MI(2) fail — semi-Markov halves the two-step structure) · proof-operation-phase-runs-r2 (C4, `hier_categorical`: BOTH hardened gate-8 moments still fail — MI(2) 0.030 vs 0.065, ACF(4) 0.058 vs 0.127; lag-curve diagnosis: lag-12 doc floor matches, lags 2–8 carry a SEGMENT-scale layer ⇒ THREE timescales, corr(len, conc) = −0.60) · **proof-operation-phase-runs-r3 (C5, `seg_hier_categorical`: the segment layer CLOSES the lag-2–8 gap — MI(2) PASSES for the first time (0.075 vs 0.065, err 0.010 ≤ 0.013) and ACF(4) flips from −55% undershoot to a marginal +21% OVERSHOOT (0.154 vs 0.127, err 0.0263 vs tol 0.0255) — but the new preregistered INSERTION CONTROL fails BOTH moments: re-fit on run-permuted streams the estimator hallucinates +0.018 MI(2) / +0.039 ACF(4), so part of the captured structure is winner's-curse artifact. Diagnosis: the three-timescale structure is REAL (real ACF(4) 0.127 ≫ permuted 0.071 — segment share ~0.056) and the family reaches it; the ESTIMATOR over-extracts. C6 gap: calibrated segment-composition extraction — cut hallucination ~2–3× at preserved sensitivity; the C5 shrinkage campaign (documented in mirrors.py) shows which estimator families fail and why)** · **C6 (estimator battery, card frozen pre-build `be8e2b6d`): NEITHER calibrated candidate passes gates 1–3 — NO r4; the r3 ABORT stands.** Null-calibrated global shrinkage (`_cal`) is defeated by the deconvolution (shrink-then-deconvolve is NON-monotone — the fixed point re-amplifies shrunk compositions; on real material the only null-clean point is λ=1.0 = zero extraction, and even that undershoots the permuted streams' own mid-lag floor ⇒ the null-clean extraction window is EMPTY); per-doc quantile deflation (`_deflate`) collapses 75% of real segments yet still leaks +0.012/+0.016 through the surviving tails. Both cancel the winner's curse in the WEAK regime (raw +34% overshoot → 3%/0.2%; permanent pytest rail) at 35–45% retained contrast on strong signal. Diagnosis sharpened: a resolution/power limit — per-segment null concentration fluctuations sit at the scale of the genuine contrast at 287×~85 corpus size. C7 direction (proposed): deconvolve-first-then-shrink (monotone) + variance-aware in-loop margins; if still inert on real material, close the reasoning half as unreachable at this corpus resolution (`expansion/results/estimator_battery_c6.md`) | **`SPEC`** — **recipe-instruction-phase-runs-r2 (C4 re-freeze PROCEED — the program's FIRST interaction/equality SPEC and first grounded regime-3 candidate:** `hier_categorical` mirror holds BOTH hardened gate-8 moments — ACF(4) \|err\| 0.018 ≤ 0.059 (the C3 killer, now passed), MI(2) \|err\| 0.029 ≤ 0.036 — skeptic 5/5, signal unchanged from C3 (ACF(1) 0.479 ≫ N1 hi 0.204, κ=0.61) → `synthetic/recipe_instruction_phase_runs/`; measured class IS multi-class run/segment equality — no re-filing) · `calib:ABORT` ×2 — greeting-signoff-mirror (C2: mis-keyed mirror) · list-item-parallelism (*C3: re-filed → bursty/self-exciting, where it PROCEEDed*) |
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
- **C4 mirror-menu resolution + the next gap.** The hierarchical categorical
  mirror was built, hardened-gate-8-validated, and **split the two C3 aborts
  by domain**: text instruction phases ARE doc-level hierarchical (recipe →
  SPEC), reasoning proof phases are NOT — they need a third, segment-scale
  layer (run / segment / doc; the lag-curve and length-concentration evidence
  is in the C4 cycle-log entry). The C5 extension is a segment-level regime
  layer — build it under the hardened ≥2-non-fitted-moment gate to keep the
  over-expressiveness tension (C3 circularity lesson) structurally controlled.
  **C5 resolution:** built (`seg_hier_categorical`) with the
  over-expressiveness tension made EXPLICIT — a preregistered insertion
  control on run-permuted streams. Outcome: the layer closes the lag-2–8 gap
  but the estimator over-extracts (control FAIL, cycle log). **The C6 gap is
  no longer the model family — it is the extraction estimator**: segment
  compositions need calibrated (not raw, not naively shrunk) estimation.
  **C6 resolution:** two calibrated candidates built under a pre-build
  frozen estimator card; NEITHER passes the frozen battery (no r4 — the
  r3 ABORT stands). The failure is now mechanistic: the C4 deconvolution
  makes shrink-then-deconvolve non-monotone, and quantile deflation leaks
  through its tails; both DO cancel the weak-regime winner's curse. The
  open C7 lever is a monotone (deconvolve-first) calibration — or an
  honest close at this corpus resolution (cycle log; battery record).
- The abstract benches (signed_motion, frequency) are out of this ledger's scope.

## Cycle log

- **Cycle 6 — 2026-07-23 (runpod, autonomous; zero API spend — battery ran
  on committed r3 labels + harness toys; skeptic never reached).** The
  briefing-mandated estimator fix, run exactly by the pre-build frozen
  card (`prereg/estimator-card-c6-segment-extraction.md`, commit
  `be8e2b6d`): TWO candidates with a-priori calibration principles —
  **(A) `seg_hier_categorical_cal`**, null-calibrated global shrinkage
  with the C5 insertion control moved IN-LOOP (λ* = smallest λ whose fit
  on run-permuted train round-trips both gate-8 moments within a
  null-referenced tolerance strictly tighter than the recorded control;
  real held-out moments never enter any objective), and
  **(B) `seg_hier_categorical_deflate`**, per-doc length-matched null
  deflation (each real segment keeps only its concentration excess over
  the 75th percentile of 20 same-doc run-permuted replicas). **Verdict:
  NEITHER passes the frozen gates 1–3 → NO r4 run; the r3 ABORT stands**
  (`results/estimator_battery_c6.{json,md}` + deterministic λ-scans).
  Mechanistic findings: (1) shrink-then-deconvolve is NON-monotone — the
  C4 self-consistency fixed point re-amplifies shrunk compositions, so
  λ is flat on [0, 0.9] then cliffs at 1.0; on real material the only
  null-clean point is λ=1.0 (zero extraction) and even the inert limit
  undershoots the permuted streams' own mid-lag floor ⇒ the null-clean
  extraction window is EMPTY at card tolerance; (2) quantile deflation
  collapses 75% of real segments and still leaks +0.012 MI(2) /
  +0.016 ACF(4) through the surviving tails; (3) BOTH candidates cancel
  the weak-regime winner's curse the raw estimator exhibits (+34%
  overshoot → 3% / 0.2%; pinned as a permanent pytest rail) at the cost
  of 35–45% retained contrast on strong signal; (4) boundary verdicts are
  generation-noise brittle (gate-2 seed flip 0.022→0.040 vs bound 0.035).
  Interpretation: a resolution/power limit of extraction at this corpus
  size, not a family failure and not evidence against the (confirmed)
  three-timescale structure. **C7 direction (proposed, unfrozen):**
  monotone deconvolve-first-then-shrink calibration + variance-aware
  in-loop margins; if a monotone estimator still calibrates to inert on
  real material, close the reasoning int/eq half as unreachable at this
  corpus resolution. Spend unchanged: **$10.82/$25** (C6 $0.00).

- **Cycle 5 — 2026-07-22 (runpod, autonomous overnight; measure→mirror only,
  zero API spend — labels cached from C3, skeptic skipped on ABORT).** The
  LEDGER-mandated build: **`seg_hier_categorical`** menu extension — the
  three-timescale mirror (per-symbol dwell / within-doc segments from a
  run-aware BIC changepoint DP with the C4 deconvolution applied per
  segment / doc-marginal tilt; segment+doc tilt weights by joint MLE; all
  objectives likelihood-based, no pooled lag statistic fitted). Null-safety
  is a **preregistered INSERTION CONTROL**, not automatic shrinkage: the
  mirror re-fit on run-permuted streams (no-adjacent-repeat shuffle — a
  plain run permutation merges same-type runs and distorts the dwell
  material, measured +58% ACF(4) on the permuted moments themselves) must
  not hallucinate either gate-8 moment beyond the real-data tolerance. A
  full campaign of automatic winner's-curse-safe estimators was measured on
  the committed harness toys and abandoned — complementary in-block halves
  (hypergeometrically anti-correlated), interleaved splits (confounded by
  dominant/excursion alternation), analytic multinomial floors (no-self-jump
  deflates real variance below them), permutation-matched DP split-half
  (segment-size mismatch), posterior-mean shrinkage (under-disperses,
  −49% ACF(4)) — each either drowned genuine signal or leaked; the campaign
  is documented in `mirrors.py` docstrings and the harness tests pin both
  control behaviors (passes on a strong three-timescale toy, catches a
  doc-homogeneous heavy-dwell null). **Verdict:
  `proof-operation-phase-runs-r3` ABORT — doubly informative** (cell log
  above): the segment layer closes the lag-2–8 gap (MI(2) passes for the
  first time; ACF(4) flips undershoot→marginal overshoot) but the insertion
  control catches the estimator hallucinating ~half the overshoot on
  exchangeable data. The three-timescale DIAGNOSIS is confirmed
  (real-vs-permuted ACF(4) gap 0.056 ≫ tolerances); the C6 target is a
  calibrated extraction estimator, with the measured hallucination
  magnitudes (+0.018 MI(2) / +0.039 ACF(4)) as the bar. Cosmetic done: the
  skeptic record header now names the actual judgment model
  (`_judge_model`, C4-review item); pre-C5 records annotated "untracked".
  Spend **$10.82/$25** cumulative (C5 $0.00).

- **Cycle 4 — 2026-07-22 (runpod, autonomous; loop judgment roles on
  `claude-fable-5`, bulk unchanged on Haiku).** The LEDGER-mandated build:
  **`hier_categorical`** menu extension (per-doc phase propensities with a
  self-consistency deconvolution — raw doc marginals flatten every
  fit→generate round — + empirical dwell + MLE-tilted global jump chain;
  harness tests show the fit round-trips ACF(4)/MI(2) within ±20% while
  `semi_markov` misses by 25–70%), and **gate-8 HARDENED to ≥2 non-fitted
  moments, all must pass** (guardrail 8; multi-moment support in calibrate).
  Both C3 int/eq real-signal aborts re-frozen by dated amendment (cards carry
  3-axis coordinates + regime-3 claim + design-time discriminability per the
  revamped rules; C3 labels + validation reused). Verdicts — **1 win, 1
  informative abort: (1) recipe-instruction-phase-runs-r2 PROCEED → SPEC**
  (`synthetic/recipe_instruction_phase_runs/`) — **the interaction/equality
  prize (text domain) and the program's first grounded regime-3 candidate**:
  both hardened gate-8 moments pass (ACF(4) err 0.018 ≤ 0.059 — the exact
  moment that killed C3's mirror; MI(2) err 0.029 ≤ 0.036), skeptic 5/5
  (caveats carried: MI margin ~17%, heterogeneity *level* inserted via the
  propensity list — hier_ar1 precedent). **(2)
  proof-operation-phase-runs-r2 ABORT** — even the doc-level hierarchy
  fails both moments on reasoning traces; lag-curve diagnosis pins the miss
  to lags 2–8 while the lag-12 floor matches ⇒ **reasoning-trace phase
  streams hold three timescales (run / segment / doc)**; corr(length,
  concentration) = −0.60 independently confirms within-doc drift. **C5
  targets:** a segment-level regime layer (hierarchical semi-Markov) for the
  reasoning int/eq cell — with the over-expressiveness tension noted (the
  hardened 2-moment gate is the structural control); stage-6 build of the
  new SPEC needs the equality-latent variant of the discriminability
  STOP-gate (both raw-linear readouts may sit at chance — verify the
  nonlinear access route instead, per the changepoint § 8 treatment).
  Ops notes: two Fable skeptic calls burned on a truncated-JSON parse crash
  before the fix (raw verdict now always persisted pre-parse; rubric-key
  validation + deterministic repair); the recorded skeptic verdict was
  recovered from the persisted raw text, never re-rolled. Spend
  **$10.82/$25** cumulative (C3 $8.20 + C4 $2.62).
  **Reviewed + APPROVED (2026-07-22, mac-local):** commit order verifies
  freeze-before-calibration; both hardened gate-8 moments preregistered with
  non-fittedness arguments and demonstrated teeth (recipe PASS with honest
  ~17%-margin caveat; proof FAIL 2.7× tolerance ⇒ rule-forced ABORT); skeptic
  transcript genuine (5/5, caveats carried into the spec); the
  `hier_categorical` deconvolution is sound and α is likelihood-fit
  (non-moment-inserting); spend + scope clean; 127 tests pass. Findings baked
  in: the **equality-latent variant of the discriminability STOP-gate** is
  now in README (both raw-linear readouts at chance + nonlinear access
  verified — the C4 flag, adopted); cosmetic for C5: the skeptic record
  header hardcodes "Opus" (`render_records.py` template) while the skeptic
  ran on Fable — fix the template next cycle. Briefing `expansion-c4.md`
  retired.

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
