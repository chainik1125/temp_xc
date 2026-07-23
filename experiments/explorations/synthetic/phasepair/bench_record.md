# Phasepair (FB-1) — bench record

**Status: DONE — verdict POSITIVE: the suite's sharpest phase-vs-power
dissociation, with a second ordering inversion (2026-07-23, runpod-b,
FB-C1 Phase 3).** Frozen card:
[`../freqbench/cards/FB-1.md`](../freqbench/cards/FB-1.md) (frozen before
construction). Gates: T1/§8 PASS
([`results/phasepair_gating_stats.json`](results/phasepair_gating_stats.json)),
T2 PASS ([`results/phasepair_t2_stats.json`](results/phasepair_t2_stats.json)
— the exact bag null holds to 0.007; shuffle is a FULL null for sign),
skeptic PROCEED 5/5
([`../freqbench/results/skeptic_verdict_FB-1.json`](../freqbench/results/skeptic_verdict_FB-1.json)).
Provenance `theorem-first`. Grid: **636/636 cells, 0 failures, 76 min.**
Stats: [`results/phasepair_bench_stats.json`](results/phasepair_bench_stats.json),
figure [`figs/phasepair_bench.png`](figs/phasepair_bench.png).

## 1. The task (frozen)

`cyclic_tones` (the frequency substrate — P1–P5 attached) at Ω = ±3, ±12,
±30 (mod 101): 6 classes = 3 magnitude pairs × sign. Within a pair the
trajectories are time reversals — identical per-channel power spectra and
identical bag-of-symbols distribution (**exact** null) — so sign is
phase-only (cross-channel quadrature). `d_in=24, σ=0.10, seq_len=64, L=32`;
F anchor = M = 101. Primary `sign_recovery` (per-pair binary probes,
normalized from ½); companion `pair_recovery` (3-class magnitude).
Memorization: 606 templates vs d_sae ≤ 202 (the signed_motion confound,
capacity-controlled). `c_relevance`: GPT-2 day-stride direction (real-model
phase code; sprint § 4.7).

## 2. Proof obligations — discharged

- **P5 signed ceiling:** sign oracle 0.968 / 1.000 / 1.000 at T = 2/4/8
  (6-class oracle 0.952 → 1.000); raw floors at chance (one-sided
  documented amendment — below-chance multiclass probes are degenerate-
  probe artifacts); bag-MLP sign 0.497–0.503 (the exact null, measured).
- **T2 chirality finding (recorded):** reflection ∘ column-flip exchanges
  the sign classes — sign is chirality w.r.t. the realized R; well-defined
  per seed, not poolable across seeds. Skeptic-judged benign.

## 3. Frozen predictions under test (card § 6 summary)

Token ≈ 0; stacked sign ≈ 0, vel < 0.1; pre sign ≈ 0 with pair leak
0.1–0.3; post sign 0.2–0.6 @T8; spectral sign 0.3–0.7 with untrained sign
≈ 0; dissociation pair ≥ sign everywhere; spectral's edge SHRINKS on sign.
Falsifiers: T=1 sign > 0.1; bag off chance; sign oracle ≈ ½; winner
trained ≈ untrained.

## 4. Blind verdict vs the frozen predictions

3-seed means at d_sae = F = 101, k_pos = 2 (sign | pair).

**4.0 Falsifiers — none fired.** T=1 sign max **0.0091**; bag exact-null
held (gating); sign oracles saturated; the winner's trained−untrained gap
is +0.40 (1.000 vs 0.597) — learning on top of a real access prior.

**4.1 Token + stacked — HELD.** Sign −0.013…+0.006, pair ≤ 0.022; both
components at zero.

**4.2 txc-pre — sign HELD at exactly 0; the predicted pair leak MISSED
LOW.** Pair +0.022/+0.036 at T=4/8 (frozen 0.1–0.3). Same lesson as FB-2:
the variance/bag route is far weaker off the single-tone setting than the
frozen transport assumed. The additive family is fully blind on BOTH
components, not dissociated.

**4.3 txc-post — MISSED HIGH, spectacularly.** Sign **+0.758 / +0.982 /
+1.000** at T = 2/4/8 (frozen band 0.2–0.6). The coincidence crosscoder
converts cross-channel quadrature *perfectly* at the canonical cell — its
best result anywhere in the suite, exceeding its own tone-magnitude
numbers. Untrained access is substantial (0.36–0.60) and trained learning
completes it.

**4.4 spectral — the structural discovery of the cycle.** Sign
**−0.001 / −0.004** at T = 2/4 while pair reads **+0.956 / +1.000** — the
pure dissociation, in a TRAINED arch: at T ≤ 4 `multiband` degenerates to
singleton DCT bands, and a singleton band has one real temporal basis
function — **no quadrature partner exists inside a branch, so the
per-branch code is sign-blind by construction**. At T = 8 the bands become
multi-index ([1,2], [3,4], [5,6,7]) and sign snaps on: **+0.936** (each
seed 0.91–0.96), untrained +0.673. The frozen "untrained sign ≈ 0"
prediction FAILED at T=8 (the multi-index band prior already carries
phase access); it held at T ≤ 4 for the structural reason above.

**4.5 Dissociation (pair ≥ sign everywhere) — HELD**, with spectral T ≤ 4
as the extreme case (gap 0.96 → 1.00 vs ≈ 0).

**4.6 "Spectral's edge shrinks on sign" — HELD in inverted form.** Post
BEATS spectral on sign at T=8 for k ≥ 2 (1.000 vs 0.936; and spectral
again degrades with budget: k1 0.954 → k8 0.576, the FB-2 collapse
pattern), while spectral ≫ post on pair at every T. The phase axis and
the power axis order the two mixing archs OPPOSITELY.

**Verdict: POSITIVE — a regime-3 phase separator with provable and
measured power-blindness, completing the panel's triple dissociation:**
across the three theorem-first benches, **spectral wins power
(multilane), txc-pre wins covariance-eigenstructure (colored_sources),
txc-post wins phase (phasepair)** — no window architecture dominates, and
which one wins is predicted by where the task sits on the coordinate
axes. This is the program's acid-test currency. It also retro-explains
the `signed_motion` NEGATIVE: on the same panel, the phase latent is
readable (post 1.000 here) — signed_motion's failure was its substrate
(random embedding + 38 templates), not the panel's phase-blindness.

## 5. Coordinates (axis 1, FreqFrac at bench time)

`freqfrac_report phasepair` (seed 1; stats under
`../freqbench/results/freqfrac_stats_phasepair_s1_T{4,8}.json`) — see the
merged table; the substrate's spectral coordinates match the frequency
bench (same generator); the sign latent adds the chirality/phase
dimension invisible to FreqFrac (a per-atom POWER lens — consistent with
sign living outside every power statistic).

## 6. Review (2026-07-23, mac-local) — APPROVED

Verdict stands — the suite's sharpest dissociation and the third leg of
the triple. The one-sided-floor amendment verified genuine (below-chance
probes at 0.112–0.115 vs chance 0.167 are degenerate-classifier
artifacts; the above-chance tolerance never moved; the failing first-pass
stats are committed at `f2f4128c`, the flip at `4f2f2c98`, disclosed to
the skeptic pre-grid). 636/636 cells, 0 dup keys; misses (pre pair-leak
low, post sign high, untrained-spectral at T=8) labeled as misses.
Structural singleton-band sign-blindness feeds the subtype rule (README
coordinates). Audit: `../freqbench/PORT.md` § H.
