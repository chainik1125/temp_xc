# Synthetic benchmark spec — list-item parallelism (enumeration run-clustering)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the loop) —
frozen by the expansion Cycle-3 PROCEED, staged for a *later, deliberately
blind* B×A evaluation. No architecture has seen or touched this spec.

**Provenance.** Expansion-loop Cycle 3 (autonomous, gated): prereg card
[`../expansion/prereg/list-item-parallelism.md`](../expansion/prereg/list-item-parallelism.md)
(frozen 2026-07-14 under the Cycle-2 design gates; C2 calibration ABORTED
solely on a mis-scaled absolute gate-8 tolerance — Fano |err| 0.163 vs ±0.15,
a 4.2% relative miss; the C2 review mandated the re-freeze), dated re-freeze
amendment 2026-07-19 (uniform ±20%-relative tolerance, preregistered before
any C3 run), calibration record
[`../expansion/records/list-item-parallelism-r2/calibration.md`](../expansion/records/list-item-parallelism-r2/calibration.md)
— verdict **PROCEED**: cleared the N1/N2/N3 battery, the noise floor, gate-8
(Fano |err| 0.163 ≤ 0.781), and all five skeptic items. Labeled at **ctx=0**
(gate-7 strict per-sentence). **The first text-corpus PROCEED of the
program.**

**Ledger cell (measured-class filing).** Proposed under
`interaction/equality`; the **measured** signature is binary event
run-clustering — ACF(1) 0.52, Fano 3.9, excite-ratio 3.97, gap CV 2.07, and a
`logistic_ar` mirror (backtracking's process family). Filed under
**`text-corpus × bursty/self-exciting`** per the C2-review re-filing rule;
the interaction/equality reading (same-template positions MATCH) remains the
*interpretation*, but the dynamics class is self-excitation.

## 1. What it tests

Whether a dictionary code linearly exposes the **run/burst state of
enumerated structure** in ordinary web text: list-item / parallel-template
sentences (base rate 0.149) come in strong consecutive runs. Measured on the
pinned 400-doc fineweb sample: indicator ACF(1) = **0.520** [0.477, 0.560] vs
N1 hi **0.212** / N2 hi **0.013** — the widest real-vs-null margin the
expansion loop has produced — noise-robust at ε̂ = 0.053 (perturbed 0.342),
split-half stable (0.515/0.524), MI(1) 0.101 nats, spectral peak 4.6. κ =
0.644 (well-validated labeler).

Note the N1 band is itself elevated (ACF ≈ 0.20): docs differ strongly in
list density (listicles vs prose), so composition alone induces pooled
autocorrelation — the gate margin is measured *above* that band, and the
skeptic's composition item cleared on exactly this comparison.

## 2. Generative process (two layers)

### Layer 1 — logistic-AR run dynamics (the fitted mirror)

Per document of length `L`, a binary list-item stream (fit on 280 train docs,
validated on 120 held-out; full precision in
[`mirror_params.json`](mirror_params.json)):

```
logit P(b_i = 1) = a + c·(i/L) + Σ_{l=1..8} w_l · b_{i−l}
a = −3.029   c = +0.155   w = [1.97, 1.04, 0.41, 0.41, 0.18, 0.28, 0.16, 0.40]
```

**Matched:** lag-1 self-excitation (+ position). **Gate-8 verified
non-fitted moment:** Fano(w=10) (held-out real 3.903 vs syn 3.740, |err|
0.163 ≤ ±20% rel = 0.781). **Deliberately NOT matched — documented fidelity
limits (skeptic note):** higher-order clustering the AR kernel under-fits —
excite-ratio (real 3.65 vs syn 4.51), spectral peak (4.11 vs 5.31). The
mirror is a *weak lower bound* on the real run structure.

### Layer 2 — emission into activations

The standard emission pattern: sentence `i` →
`x_i = b_i·m·u_list + Σ_{j∈content_i} m_j·u_j + σ·ε_i`, with `u_list` the
list-item feature, `K_c` content features sized so `F = 20`, `d_in = 64`
(pinned at datasource-plugin time; none added this cycle).

## 3. Ground truth / capacity / metrics

Per Part II conventions and the uniform design (identical to the
backtracking-bench treatment): `F` = 1 + K_c directions; hidden latent = the
conditional intensity λ_i (linear in the 8-lag history; chance = base rate
0.149, oracle = the generating λ); `d_sae ∈ {F//2, F, 2F}`,
`k_pos ∈ {1,2,4,8,16}`, seeds {1,2,42}, untrained control; `L = 32`,
`T ∈ {2,4,8}` tiled; cosine-AUC / linear λ-probe (example-split) / windowed
NMSE.

## 4. Predictions (frozen in the card, before any run)

- **per-token SAE:** fires a list-item feature per token but cannot expose
  run-length coupling (the DPI-floor argument).
- **window families:** windows spanning a run capture the match-cluster;
  per-token cannot.

## 5. Caveats carried from calibration

- **The heuristic cross-check is weak** (F1 0.09; judge rate 0.149 vs regex
  0.022): the judge's notion of "parallel-template unit" is far broader than
  leading-enumerator regexes. The inter-judge floor (κ 0.644, ε̂ 0.053) is
  the binding validation; the skeptic reviewed and cleared this explicitly.
- Mirror under-fits higher-order clustering (§ 2) — weak validation only,
  and the gate-8 pass is under the C3 relative tolerance, not the C2
  absolute one (both preregistrations and both outcomes are on record).
- Same dynamics family as backtracking/self-reference-echo (logistic-AR);
  the *domain* (web text) and *property* (structural enumeration) are new —
  this is the program's first text-corpus benchmark.
- One corpus snapshot: fineweb sample-10BT, seed 0, 400 docs, pinned
  splitter.

## 6. Reproduction

```bash
.venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate list-item-parallelism-r2
.venv/bin/python -m experiments.explorations.synthetic.expansion.render_records
```

_Frozen 2026-07-19 by the runpod agent (expansion Cycle 3). Stage-6 must go
through the canonical runner with a registered datasource plugin; nothing
here has been run against any architecture._
