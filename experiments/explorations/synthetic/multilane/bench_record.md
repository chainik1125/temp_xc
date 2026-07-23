# Multilane superposition (FB-2) — bench record

**Status: GRID RUNNING (2026-07-23, runpod-b, FB-C1).** Frozen card:
[`../freqbench/cards/FB-2.md`](../freqbench/cards/FB-2.md) (commit
`f0e6778f`, BEFORE construction). Gates: T1/§8 PASS
([`results/multilane_gating_stats.json`](results/multilane_gating_stats.json)),
T2 PASS ([`results/multilane_t2_stats.json`](results/multilane_t2_stats.json)),
skeptic PROCEED 5/5
([`../freqbench/results/skeptic_verdict_FB-2.json`](../freqbench/results/skeptic_verdict_FB-2.json)).
Provenance `theorem-first`. Grid: 708 cells (uniform 30k-step design +
the frozen band-partition addendum), canonical runner.

> ⚠ THIS FILE IS A SKELETON until the blind-verdict section is written from
> `results/multilane_bench_stats.json`. No claim below the fold is final.

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

*(TO FILL from `multilane_bench_stats.json` — check falsifiers FIRST.)*

## 5. Coordinates (axis 1, FreqFrac at bench time)

*(TO FILL — `freqfrac_report` on the canonical cells once registered.)*
