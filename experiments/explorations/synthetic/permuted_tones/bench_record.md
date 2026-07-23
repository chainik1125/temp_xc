# Permuted tones (FB-5) — bench record

**Status: DONE — verdict POSITIVE (weak realization): the temporal-knob
acid test lands on the ALIGNMENT side — trained spectral is numerically
pinned to the envelope reference at every window length, txc-post is the
only arch reading temporal structure beyond it, and the subtype rule's
power leg gains its alignment qualifier (2026-07-23, runpod-b, FB-C3).**

Frozen card: [`../freqbench/cards/FB-5.md`](../freqbench/cards/FB-5.md)
(committed pre-build: "card FB-5 permuted_tones FROZEN pre-build";
mac-local's five direction predictions verbatim + the fork stated in
advance). Build: "freqbench FB-5 build" (generator + datasource +
`permuted_recovery` add-on + contract tests). Gates committed pre-run
("freqbench FB-5 gates: … COMMITTED PRE-RUN") and **passed on their first
run with ZERO amendments** —
[`results/permuted_gating_stats.json`](results/permuted_gating_stats.json),
[`results/permuted_t2_stats.json`](results/permuted_t2_stats.json).
Skeptic PROCEED 5/5
([`../freqbench/results/skeptic_verdict_FB-5.json`](../freqbench/results/skeptic_verdict_FB-5.json)).
Grid: **636/636 cells, 0 failures, 32 min** →
[`results/permuted_grid_results.json`](results/permuted_grid_results.json).

## 1. The task (frozen)

The frequency substrate with the linear phase schedule replaced by K = 10
uniformly-random permutation schedules of Z_M (per data seed):
`z_t = π_Y((t+B) mod M)`, circle codebook, `M=101, d_in=128, σ=0.10,
seq_len=64, L=32, n_steps=6000`, F anchor 101. Order-2-even structure
preserved; DCT-alignment of the trajectory destroyed (lag-1 trajectory
autocorrelation measured −0.07…+0.20 ≈ 0 ± O(1/√M) vs the tone ladder ±1
— the § 1 non-absorption statistic). Primary `schedule_recovery`
(chance 0.1); oracle = matched filter over (schedule, offset).

## 2. Gate evidence (first-run passes, no amendments)

- **T1/§ 8:** P1 marginal TV ≤ 0.018/class; per-token probe 0.0998 and
  window-concat linear 0.0987 — both AT chance (the FB-4-informed +0.05
  headroom unused); matched-filter oracle **0.43 / 0.99 / 1.00** at
  T = 2/4/8 (the card's T-resolution curve; saturation by T=4); frozen
  T=1 falsifier −0.0002 (clean). **Envelope reference** (logistic on
  circle-plane per-DCT-index energies only): balacc 0.115 / 0.143 / 0.205
  at T = 2/4/8 → **0.017 / 0.048 / 0.116 in recovery units**.
- **T2:** within-window shuffle kills the oracle (1.00 → 0.125 at T=8) —
  its route is order; the order-free set route is weak in practice
  (bag-MLP 0.151 balacc; the card's § 3 multiset honesty); bag-LINEAR
  0.099 = chance (P2, the actual claim). Memorization: K·M = 1,010
  templates ≫ d_sae ≤ 202; schedule table per-seed (no cross-seed
  pooling).

## 3. Blind verdict vs the five frozen directions

3-seed means, `schedule_recovery`, d_sae = F = 101 unless stated.

**3.1 Additive family ≈ 0 — HELD.** Token −0.000, stacked ≤ 0.011,
txc-pre ≤ 0.019 (T=8). P1/P2 exactly as on every tone bench.

**3.2 txc-post positive at T ∈ {4,8} — HELD (direction), with the
magnitude datum.** Trained 0.074 (T=4) / 0.075 (T=8, k=2), rising with
budget to **0.146** (k=8, d=101) and **0.161** (k=8, d=202) — clearly
positive and clearly learned (untrained 0.013). The indicative 0.1–0.8
band is entered only at k ≥ 4; at the canonical k=2 cell the value sits
below it (0.075). Post is the ONLY arch that exceeds the envelope
reference — at every T given budget (T=2: 0.031 > 0.017; T=4: 0.074 >
0.048; T=8: 0.122–0.161 > 0.116) — i.e. the only architecture reading
temporal structure rather than band energies. Its budget response is the
OPPOSITE of spectral's (rises to k=8, collapses only at k=16: 0.054).

**3.3 spectral below post at the canonical T=8 cell — MIXED literally,
HELD in mechanism (the operative clause).** Literal cell read: at T=8
k=2 spectral 0.096 > post 0.075 (prediction FAILS there); at k ≥ 4 it
holds (0.098 < 0.122, 0.055 < 0.146); at T=4 canonical it holds (0.042 <
0.074). The unifying fact is the frozen mechanism clause — *"residual
spectral score should track the envelope reference"* — which held with
startling precision at ALL THREE window lengths:

| T | spectral trained | envelope reference |
|---|---|---|
| 2 | 0.016 | 0.017 |
| 4 | 0.042 | 0.048 |
| 8 | 0.096 | 0.116 |

Spectral IS the envelope reader, quantitatively; its apparent "win" over
post at (T=8, k≤2) is envelope information growing with T while post is
budget-starved, not structure-reading. Its band decomposition spreads the
signal thinly (band recoveries 0.019–0.053, no dominant band — broadband,
as constructed), and its k-collapse pattern recurs (k=1: 0.115 → k=8:
0.055, the FB-2 signature).

**3.4 spectral untrained ≈ post untrained — PARTIAL.** The multilane
access-prior MAGNITUDE collapses as predicted (+0.298 → **0.045**, a 6.6×
drop; no band alignment to exploit), but exact equality fails: spectral
0.034–0.054 vs post 0.012–0.015 — a small, seed-disjoint residual gap.
Scored as a partial hold: the collapse is real, the ≈ is not exact.

**3.5 Falsifiers — none fired.** Max T=1 recovery 0.0047 across all 162
T=1 cells; winners' trained−untrained gaps ≫ seed spread (post +0.13,
spectral +0.05).

**The fork (card § 6, stated in advance) resolves to the ALIGNMENT side:**
spectral does not match or beat post *beyond the envelope reference*
anywhere — its structure-reading capability on a spectrally-generic
schedule is nil. **The subtype rule's power leg becomes:
"power/equality → spectral, when the power concentrates in few DCT
bands."** (README coordinate-section edit left for mac-local — program-
rule changes are out of session scope; proposed wording above.)

**Weak-realization datum (the FB-3 pattern, now on the tone substrate):**
the matched-filter oracle is 1.0 at T=8 and the best arch reads 0.161 —
16 % of a provable ceiling. Whole-window matched filters for arbitrary
schedules exist in txc-post's function class but training barely finds
them; the gap is the finding.

## 4. Coordinates (axis 1, FreqFrac at bench time)

`freqfrac_stats_permuted_tones_s1_T{4,8}.json` (merged table updated):
the broadband pole realized — txc-post learns concentrated atoms
(conc 0.686/0.431 vs init 0.53/0.28) with **no spectral tilt** (dc_frac
0.126 ≈ init 0.127 at T=8; contrast frequency's high-pass shift) —
matched-filter-ish atoms with flat spectral placement, exactly what a
generic-schedule detector should look like. Spectral sheds DC toward its
AC bands (0.157 vs init 0.249) — consistent with envelope-reading spread
across bands.

## 5. Review

Pending mac-local. Items for review carried in § 3.3/§ 3.5 of this record
and the cycle log (PORT § J): the proposed alignment-qualifier wording for
the README subtype rule; the FB-5 FreqFrac broadband rows as the axis-1
pole anchor.
