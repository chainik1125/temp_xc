# Multilane superposition (FB-2) — bench record

**Status: DONE — verdict POSITIVE (2026-07-23, runpod-b, FB-C1); the
sprint-transported band headline PARTIALLY FAILS (see § 4.7).** Frozen card:
[`../freqbench/cards/FB-2.md`](../freqbench/cards/FB-2.md) (commit
`f0e6778f`, BEFORE construction). Gates: T1/§8 PASS
([`results/multilane_gating_stats.json`](results/multilane_gating_stats.json)),
T2 PASS ([`results/multilane_t2_stats.json`](results/multilane_t2_stats.json)),
skeptic PROCEED 5/5
([`../freqbench/results/skeptic_verdict_FB-2.json`](../freqbench/results/skeptic_verdict_FB-2.json)).
Provenance `theorem-first`. Grid: 708 cells (uniform 30k-step design +
the frozen band-partition addendum), canonical runner.

Grid outcome: **708/708 cells, 0 failures, 80 min** (28 workers). Stats:
[`results/multilane_bench_stats.json`](results/multilane_bench_stats.json),
figure [`figs/multilane_bench.png`](figs/multilane_bench.png).

## 1. The task (frozen)

3 simultaneous circle tones in mutually orthogonal 2-planes (one Haar
isometry into `R^24`), per-lane hidden velocity `Y_k ~ Unif(Ω)` +
phase `B_k ~ Unif(Z_M)`; `M=101`, `Ω` the 10-tone ladder, `σ=0.25`,
`seq_len=64`, `L=32`. `F` anchor = per-lane alphabet `M=101`
(`d_sae ∈ {50,101,202}`). Primary metric `multilane_recovery` (mean
per-lane normalized logistic probe on the shared per-tile code).

## 2. Proof obligations — discharged

- **P5 ceiling (per-lane periodogram = ML):** verified numerically on the
  built generator — per-lane oracle EQUALS the matched single-lane oracle
  (worst gap 0.0071 across T ∈ {2,4,8,16}); T=16 reproduces the sprint's
  0.995. Oracle at the design frontier: 0.421 (T=2), 0.750 (T=4),
  **0.906 (T=8)**.
- **P1/P2 floors:** raw per-token linear probes at chance (worst dev
  0.0005); raw-linear window-concat at chance (worst dev 0.0039) — the
  equality-variant configuration. T2: additive linear on stacked trained
  token codes 0.101 (chance), bag-MLP 0.102 vs oracle 0.906.
- **P6 immunity:** `|Ω|³M³ ≈ 1.03e9` templates ≫ every capacity; no
  memorization route at any cell.
- **Shuffle semantics (stated):** per-window independent permutations
  destroy phase progression, keep the symbol multiset — shuffled oracle
  degrades 0.906 → 0.208 (NOT a full null; the spread cue survives).

**Documented gate amendment:** the § 8 info-presence check keys on the
ORACLE witness (0.906 on raw tiles), not a generic MLP (0.173 recorded as a
datum — probe capacity, not information). Skeptic examined and accepted.

## 3. Frozen predictions under test (card § 6, verbatim summary)

1. Token archs ≈ 0 (< 0.05); stacked < 0.10; txc-pre 0.05–0.30 flat-S(f).
2. txc-post positive 0.3–0.7 at T=8; spectral best trained 0.6–0.9 at T=8;
   spectral untrained access ≫ other archs' untrained.
3. Ordering: spectral > txc-post > {pre, stacked, token} at T ∈ {4,8};
   positive T-trend for mixing archs.
4. **Sprint-transported headline:** 4-band > 1-band (`spectral_txc` >
   `spectral_txc_full`) by ≥ 0.03 at T=8, d=101, k_pos=1, no seed overlap —
   MAY FAIL under BatchTopK (an informative negative about the sprint's
   plain-TopK result).
5. k_pos structure: winner margin largest at k_pos ∈ {1,2}, shrinking by 8.
6. Falsifiers: any arch > 0.1 at T=1 (P1 bug); trained ≈ untrained for the
   winner (access, not learning); per-lane oracle ≪ single-lane (P5 fail).

## 4. Blind verdict vs the frozen predictions

Written against the card § 6 predictions, falsifiers checked first. All
numbers are 3-seed means of `multilane_recovery` (normalized [0, 1];
per-lane periodogram oracle references: 0.418 / 0.748 / 0.906 at T=2/4/8).

**4.0 Falsifiers — none fired.** Worst T=1 trained cell **0.0043** (bar
0.1; P1 holds in the trained grid). The winner's trained/untrained gap is
large (0.794 vs 0.298 at T=8, d=F — learning on top of a real access
prior, exactly the frequency-bench pattern). P5 verified at gating.

**4.1 Token archs ≈ 0 — HELD, exactly.** batchtopk_sae +0.000, tsae
−0.000 (untrained −0.001). The provable P1/P2 floor, measured.

**4.2 stacked < 0.10 — HELD.** Max over the whole frontier **+0.024**
(T=8). With memorization dead by construction this number is clean — the
per-position family reads essentially nothing, no A5-style caveat needed.

**4.3 txc-pre 0.05–0.30 — MISSED LOW (prediction failed, informatively).**
Measured **+0.009 … +0.047** (best: T=8, k=4). The frozen band transported
the single-tone bag level (frequency: 0.27); under 3-lane interference the
bag/variance route collapses nearly to chance. The qualitative claim
(additive family ≪ mixing family) holds everywhere; the magnitude
prediction was wrong — superposition is *harder* on the additive route
than single-tone, not equally hard. Flat per-lane S(f) confirmed
(recalls 0.09–0.26, no resolvable-band structure).

**4.4 txc-post 0.3–0.7 at T=8 — HELD.** +0.461 at the canonical k=2
(+0.762 at k=8). T-trend positive (0.096 → 0.226 → 0.461). Its per-lane
S(f) is the Rayleigh high-pass (recalls ≥ 0.87 for f ≥ 0.04, ~0.37 on the
unresolvable low cluster). Capability cost visible: NMSE 0.575 vs 0.362
per-token — the scarcity-forced specialization price (changepoint's
pattern, reproduced under superposition).

**4.5 spectral best, 0.6–0.9 at T=8 — HELD.** **+0.794** (T=8, d=F, k=2),
+0.561 at T=4. Untrained access +0.298 — 4× the next arch's untrained
(post +0.075): the DCT-band access prior, as frozen. And the capability
gate passes decisively: spectral's NMSE **0.293** is the best in the
panel (beats per-token 0.362) — the win is representation, not artifact.

**4.6 Ordering + T-trend — HELD everywhere.** spectral > post ≫ {pre,
stacked, token} at every T ∈ {2, 4, 8}; both mixing archs rise
monotonically in T.

**4.7 The sprint-transported headline — FAILS ITS FROZEN CRITERION
(direction survives, magnitude does not).** Frozen: 4-band
(`spectral_txc`) > 1-band (`spectral_txc_full`) by **≥ 0.03** at T=8,
d=101, k_pos=1, no seed overlap. Measured: **+0.776 [0.773–0.780] vs
+0.757 [0.752–0.762]** — seed-disjoint but the margin is **+0.019 <
0.03**. The sprint's W=16/plain-TopK magnitude (0.96 vs 0.91) does NOT
transport to the fair BatchTopK backbone at T=8 — the batch-pooled budget
already does part of the anti-crowding work the sprint attributed to
per-band budgets. The band advantage is real but **T-localized**: at T=4
the same comparison is **+0.468 vs +0.381 (+0.087, seed-disjoint)** — the
edge peaks where the Rayleigh cell is coarsest relative to the ladder and
shrinks as T resolves the tones (extrapolating to the frequency bench's
T=16 tie). Two unfrozen observations for the record (flagged as such):
- **2-band (dcac) is the WORST partition** (T=4: +0.351; T=8: +0.742,
  below full-band): its DC branch locks half the atoms + budget onto a
  band that carries nothing for pure tones — band *placement*, not band
  *count*, is what matters.
- **The band prior is a scarcity prior on k too**: spectral − post margin
  is +0.544 at k_pos=1, +0.333 at k=2, +0.053 at k=4, **−0.583 at k=8**
  (spectral collapses to +0.179 at k=8/T=8 while post reaches +0.762).
  The frozen "margin largest at scarce k" prediction HELD, in the
  strongest possible form (sign reversal at dense budgets).

**Verdict: POSITIVE — a regime-3 architecture separator under
superposition, memorization-clean by construction.** The full panel
ordering (spectral > post ≫ additive family ≈ token ≈ 0) appears with no
capacity caveat anywhere in the sweep; the winner also wins
reconstruction. The transported sprint claim survives only in sign: its
magnitude was a plain-TopK artifact at the frozen T=8 criterion, and the
honest reading is that band-partitioning matters at scarce budgets and
coarse windows, converging to the 1-band crosscoder as either resource
grows.

## 5. Coordinates (axis 1, FreqFrac at bench time)

`freqfrac_report multilane` at the canonical cells (seed 1; stats under
`../freqbench/results/freqfrac_stats_multilane_s1_T{4,8}.json`), firing-
weighted dc_frac / concentration, trained (init):

| arch | T=4 dc | T=4 conc | T=8 dc | T=8 conc |
|---|---|---|---|---|
| token archs | 1.000 (1.000) | 1.000 | 1.000 (1.000) | 1.000 |
| stacked | 0.263 (0.246) | 0.561 (0.567) | 0.129 (0.125) | 0.318 (0.313) |
| txc-pre | 0.271 (0.242) | 0.604 (0.560) | 0.138 (0.116) | 0.339 (0.316) |
| txc-post | — | — | 0.322 (0.117) | **0.837 (0.315)** |
| spectral | — | — | 0.321 (0.387) | 0.960 (0.944) |

The weight-space image matches the recovery story: stacked/pre stay at
their init spectra (blind); **txc-post's per-atom concentration jumps
0.315 → 0.837 at T=8** — under superposition it learns *sharper* tone
atoms than on the single-tone bench (0.47 there), because three lanes
force per-tone specialization; spectral tilts its firing off DC below
init. Axis-1 coordinate: high-band / multi-line AC, order-2, stationary —
as the card declared.

## 6. Review (2026-07-23, mac-local) — APPROVED

Verdict stands. Freeze order proven (card `9e6427be` → build/gating
`d3d6cc1a` → grid); the info-presence amendment verified genuine (ML
oracle is the correct presence witness; MLP datum kept); 708 grid cells +
2 disclosed 300-step smoke cells reconcile the leaderboard exactly; 0
duplicate eval_keys; misses (txc-pre band; the sprint's multiband>vanilla
headline) framed honestly. Cycle-level audit: `../freqbench/PORT.md` § H.

## T=16 frontier addendum (FB-C2, 2026-07-23, runpod-b)

**Design:** as the frequency addendum — window archs + the band pair at
`T=16`, uniform cells, seeds {1,2,42} + untrained (162 cells, driver commit
`32851ee8`, 870/870 grid ok, 0 failures). Frozen prediction (mac-local,
`briefings/freqbench-t16-fbc2.md`, frozen 2026-07-23 pre-run): *"the 4-band >
1-band margin (T=8: +0.019) vanishes or inverts at T=16 (≤ +0.01) — the band
advantage is a coarse-window scarcity effect."* Scored blind post-run.

**Verdict on the prediction — HELD at every capacity (one inversion).**
Matched-budget margins (spectral_txc 4-band minus spectral_txc_full 1-band,
k_pos=1, 3-seed means):

| d_sae | T=8 margin | T=16 margin |
|---|---|---|
| 50  | +0.014 | **+0.008** |
| 101 | +0.019 | **−0.015** (inverts) |
| 202 | +0.017 | **+0.005** |

All three T=16 margins ≤ +0.01; at the anchor capacity the ordering inverts
outright. Combined with the FB-C1 finding (the frozen T=8 bar +0.03 already
failed), the record now reads: **the sprint's multiband>vanilla headline is
confined to coarse windows under plain-TopK scarcity — under the fair
batch-pooled backbone the band-partition prior confers no frontier
advantage, and the last trace of it vanishes by T=16.**

**Context at T=16 (not part of the frozen claim):** spectral's frontier
rises with the resolved ladder (d=101: 0.849, d=202: 0.930 vs 0.776/0.774
at T=8) but its scarce-capacity cell collapses (d=50 k=1: 0.438 — k_win=16
against 50 atoms; and d=50 k=2: 0.161, the FB-2 budget-collapse pattern one
octave on). txc-post reaches 0.522–0.533 at k=2. The additive family stays
pinned (pre ≤ 0.075, stacked ≤ 0.021 — P2 does not move with T). Untrained
spectral access grows to 0.34–0.35 (from ~0.30 at T=8). FreqFrac T=16
(seed 1, `freqfrac_stats_multilane_s1_T16.json`): trained spectral dc_frac
0.211 vs init 0.426 — the same doubled DC-shedding as the frequency
addendum.
