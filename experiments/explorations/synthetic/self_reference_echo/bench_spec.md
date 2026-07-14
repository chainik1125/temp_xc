# Synthetic benchmark spec — self-reference echo (problem-anchoring recurrence)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the loop) —
frozen by the expansion Cycle-2 PROCEED, staged for a *later, deliberately
blind* B×A evaluation. No architecture has seen or touched this spec.

**Provenance.** Expansion-loop Cycle 2 (autonomous, gated): prereg card
[`../expansion/prereg/self-reference-echo.md`](../expansion/prereg/self-reference-echo.md)
(frozen 2026-07-14 before any data, under the Cycle-2 design gates), calibration
record
[`../expansion/records/self-reference-echo/calibration.md`](../expansion/records/self-reference-echo/calibration.md)
— verdict **PROCEED**: cleared the N1/N2/N3 battery, the noise floor, the
**preregistered gate-8 non-fitted-moment check** (MI(1): |err| 0.004 ≤ 0.015),
and all five skeptic items. Labeled at **ctx = 0** (gate-7 strict per-sentence).

## 1. What it tests

Whether a dictionary code linearly exposes the **recurrent re-anchoring
state** of reasoning: sentences that restate the problem being solved ("we
need to find…", "the problem asks…", "we are given…") cluster in runs and
recur throughout a trace. Measured on real R1-Distill traces (base rate
0.168): indicator ACF(1) = **0.311** [0.285, 0.340] vs N1 hi **0.058** / N2 hi
**0.068**, noise-robust at ε̂ = 0.102 (perturbed 0.151), split-half stable
(0.299/0.322); Fano 2.12, excite-ratio 2.46, decaying multi-lag tail
(0.31 → 0.08 over lags 1–8).

**Class-assignment caveat (flagged for review):** the card sits in the
`interaction/equality` ledger cell (match/echo framing), but the *measured*
signature is smooth run-clustering — mechanically closer to
bursty/self-exciting than to discrete echo peaks. The verdict is unaffected;
the taxonomy may deserve a reviewer relabel.

## 2. Generative process (two layers)

### Layer 1 — logistic-AR re-anchoring dynamics (the fitted mirror)

Per sequence of length `L`, a binary re-anchoring stream (fit on 210 train
traces, validated on 90 held-out; full precision in
[`mirror_params.json`](mirror_params.json)):

```
logit P(b_i = 1) = a + c·(i/L) + Σ_{l=1..8} w_l · b_{i−l}
a = −2.817   c = +0.678   w = [1.57, 0.52, 0.14, 0.23, …]
```

**Matched:** the lag-1 self-excitation (ACF(1)). **Gate-8 verified
non-fitted moment:** MI(1) (held-out 0.0349 real vs 0.0390 syn).
**Deliberately NOT matched:** which specific problem facts get restated (the
label's content), and any echo-at-fixed-lag fine structure beyond the AR
kernel.

### Layer 2 — emission into activations

The standard emission pattern (backtracking's): sentence `i` →
`x_i = b_i·m·u_ref + Σ_{j∈content_i} m_j·u_j + σ·ε_i`, with `u_ref` the
re-anchoring feature, `K_c` content features sized so `F = 20`, `d_in = 64`
(pinned at datasource-plugin time; none added this cycle).

## 3. Ground truth / capacity / metrics

Per Part II conventions and the uniform design (identical to the
backtracking-bench treatment): `F` = 1 + K_c directions; hidden latent = the
conditional intensity λ_i (linear in history; chance = base rate, oracle =
the generating λ); `d_sae ∈ {F//2, F, 2F}`, `k_pos ∈ {1,2,4,8,16}`, seeds
{1,2,42}, untrained control; `L = 32`, `T ∈ {2,4,8}` tiled; cosine-AUC /
linear λ-probe (example-split) / windowed NMSE.

## 4. Predictions (frozen in the card, before any run)

- **per-token SAE:** detects the re-anchoring lexicon per sentence; blind to
  the history-driven intensity (the DPI-floor argument of the backtracking
  bench applies).
- **window families:** backward-looking windows recover λ from the event
  history; predicted to behave like the backtracking bench's window result
  given the closely matched dynamics class.

## 5. Caveats carried from calibration

- **Labeler is marginal:** inter-judge κ = 0.304 (barely above the 0.30
  adequacy floor); the independent heuristic has recall 0.10 (judge's notion
  of re-anchoring is much broader than the lexicon). The skeptic flagged
  this explicitly (not a kill, given the 5× gate margin). A stricter/better-
  validated labeler is a reasonable pre-stage-6 improvement.
- Dynamics overlap with backtracking's class (see the class caveat above) —
  the *property* is new; the *mirror family* is the same.
- One model, one domain: R1-Distill math/logic traces.

## 6. Reproduction

```bash
.venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate self-reference-echo
.venv/bin/python -m experiments.explorations.synthetic.expansion.render_records
```

_Frozen 2026-07-14 by the runpod agent (expansion Cycle 2). Stage-6 must go
through the canonical runner with a registered datasource plugin; nothing here
has been run against any architecture._
