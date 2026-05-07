---
author: Aniket
date: 2026-05-07
tags:
  - results
  - complete
---

## C2 review for Han: where the strongest TXC win actually lives + Effect-1-vs-Effect-2 verdict

Notes from a fresh-eyes review of `purified/docs/components/c2.md` and the
plots referenced from it on `origin/final`. Han asked for extra eyes on the
synthetic stuff while tired; this is the writeup.

Plots inspected:

- `experiments/c2_synthetic_coupled/plots/c2_headline_2panel_np10.png`
- `experiments/c2_synthetic_coupled/plots/c2_setup_d_np10_scatter.png`
- `experiments/c2_synthetic_coupled/plots/c2_setup_d_np5_scatter.png`
- `experiments/c2_synthetic_coupled/plots/c2_txc_win_gauc_vs_k_np10.png`
- `experiments/c2_synthetic_coupled/plots/c2_rho_sweep_k1.png`
- `experiments/c2_synthetic_coupled/plots/c2_rho_sweep_k5.png`
- `experiments/c2_hierarchical/plots/c2_hierarchical_gauc_vs_k.png`
- `experiments/c2_hierarchical/plots/c2_setup_e_scatter.png`
- `experiments/c1_noisy_filler/plots/c2_noisy_auc_vs_kpos.png`
- `experiments/c1_noisy_filler/plots/c2_noisy_singlelatent_scatter.png`
- `experiments/c1_noisy_filler/plots/c2_noisy_probe_scatter.png`
- `experiments/c1_noisy_filler/plots/c2_noisy_denoising_panels.png`

## Bottom line

1. **The strongest TXC win in the entire C2 suite is in Setup B (noisy
   emissions), not Setup D.** Specifically:
   - `c2_noisy_denoising_panels.png` rightmost panel: **TXC-pro T_max=10 is
     the only architecture that crosses above the perfect-denoising line**
     (denoising ratio ≈ 1.3, sustained across all k_pos). TXC-base T=2
     touches 1.0. Everyone else is at 0.45-0.7.
   - `c2_noisy_probe_scatter.png`: TXC clusters cleanly above y=x; TopK SAE
     sits *on* y=x (no denoising); T-SAE sits *below* y=x (anti-denoising);
     Stacked on y=x. Tier separation is qualitatively distinct in a way no
     other Setup achieves.
   - `c2_noisy_auc_vs_kpos.png`: at k_pos = 1-3, TopK SAE is pinned at the
     per-token noise floor (gAUC ≈ 0.39-0.53) while TXC-base T=5 is at
     0.93-0.99. Gap +0.54 / +0.42 / +0.36. Bigger and more sustained than
     the Setup D pB05_np10 gap.

2. **Setup C (ρ-sweep) is conclusive, and it kills the temporal-pattern
   claim.** At k_pos = 1, TXC-base T=5 and TXC-pro T=2 are both at
   gAUC = 0.99 *across all ρ ∈ {0.0, 0.3, 0.6, 0.7, 0.9}*, completely flat.
   The win exists at ρ = 0 — i.i.d. firing with no temporal structure to
   detect. This is **definitive Effect-1 evidence**: TXC is doing implicit
   budget extension via window aggregation, not temporal pattern detection.
   Han's own caveat (Setup A T-modulation: gAUC drops as T grows) was
   already pointing this way; the ρ-sweep nails it down.

3. **Setup A's headline framing is wrong as stated.** The c2.md text says
   "TXC ≥ 0.95 at k≤5; per-token archs saturate at gAUC ≈ 0.7-0.8" but the
   table shows TopK SAE hits gAUC = 0.99 at k=2 and k=3 (same as TXC). The
   actual TXC advantage in Setup A is at k=1 (0.99 vs 0.56) and at k≥6
   (where TopK declines faster). At k=2-3 it's a tie at 0.99. Reword.

4. **Setup E's "TXC above y=x, SAE below y=x" claim doesn't match the
   scatter.** In `c2_setup_e_scatter.png` most TXC-base T=5 points sit
   *below* y=x at low k (e.g. k=1 at eAUC ≈ 0.79, gAUC ≈ 0.64), and TopK SAE
   rises monotonically to gAUC = 0.83 at k=8 — *catching up and overtaking*
   TXC. The "engineered global-vs-local divide" doesn't really land.
   Demote to the appendix.

5. **Setup D pB05_np10 is still a clean win**, just not THE headline. TXC-base
   T=5 holds gAUC ≥ 0.90 across k_pos = 1-4 while TopK SAE drops 0.92 →
   0.56. Gap at k=3 is +0.30; at k=4, +0.34. The scatter
   (`c2_setup_d_np10_scatter.png`) shows the TXC family in the upper region
   and TopK SAE tracing a down-right line. Use as a secondary headline.

6. **Setup D pB05_np5 (Dmitry replicate) is much weaker than pB05_np10.**
   At k=1 the gap is +0.36 (TXC-pro T=2 at 0.99 vs TopK at 0.63), but at k=2
   the gap shrinks to +0.10 (TXC-pro 0.90 vs TopK 0.89). TopK SAE has a
   non-monotonic spike at k=2 (0.63 → 0.89) that the prose doesn't address.
   Lead with np10, mention np5 as the Dmitry-replicate confirmation.

## Concrete numbers to cite in the paper

Setup B, k_pos = 1 (the most striking single number, because the gap is
between TXC-base T=5 = 0.93 and TopK SAE = 0.39 at the per-token floor):

| arch                  |  T  |   gAUC at k=1   |   gAUC at k=5   | denoising ratio  |
|-----------------------|-----|-----------------|-----------------|------------------|
| TopK SAE              |  -  | 0.39 ± 0.00     | 0.73 ± 0.01     | ≈ 0.45           |
| Stacked T=2           |  2  | 0.41 ± 0.01     | 0.59 ± 0.02     | ≈ 0.50           |
| TFA-pos               |  -  | 0.82 ± 0.02     | 0.79 ± 0.01     | ≈ 0.45           |
| T-SAE                 |  -  | 0.65 ± 0.01     | 0.97 ± 0.00     | ≈ 0.70           |
| TXC-base              |  2  | 0.62 ± 0.01     | 0.99 ± 0.00     | ≈ 1.00           |
| TXC-base              |  5  | **0.93 ± 0.00** | 0.99 ± 0.00     | ≈ 0.70           |
| TXC-pro               | 10  | 0.88 ± 0.02     | 0.77 ± 0.01     | **≈ 1.30**       |

(Tier-separation lift is largest at k=1 for TXC-base T=5; denoising ratio
crosses 1.0 only for TXC-pro T_max=10.)

Setup C, ρ-sweep at k_pos = 1 (the integrity-move evidence):

| ρ      | TopK SAE gAUC      | TXC-base T=5 gAUC   | TXC-pro T=2 gAUC    |
|--------|--------------------|---------------------|---------------------|
| 0.0    | 0.45 ± 0.16        | **0.99 ± 0.00**     | **0.99 ± 0.00**     |
| 0.3    | 0.47 ± 0.16        | **0.99 ± 0.00**     | **0.99 ± 0.00**     |
| 0.6    | 0.55 ± 0.20        | **0.99 ± 0.00**     | **0.99 ± 0.00**     |
| 0.7    | 0.56 ± 0.19        | **0.99 ± 0.00**     | **0.99 ± 0.00**     |
| 0.9    | 0.46 ± 0.18        | **0.99 ± 0.00**     | **0.99 ± 0.00**     |

(Read off the ρ-sweep plots; TXC is flat across ρ; TopK is also basically
flat with one peak at ρ ≈ 0.6-0.7 that's within seed noise. The TXC win
exists at ρ = 0.)

Setup D pB05_np10, gAUC vs k_pos (the secondary-headline numbers):

| k_pos | TXC-base T=5 | TopK SAE | gap   |
|-------|--------------|----------|-------|
|   1   | 0.99         | 0.92     | +0.08 |
|   2   | 0.99         | 0.81     | +0.18 |
|   3   | 0.98         | 0.68     | **+0.30** |
|   4   | 0.90         | 0.56     | **+0.34** |
|   5   | 0.77         | 0.51     | +0.27 |
|   6   | 0.64         | 0.47     | +0.17 |

## Recommended paper framing

The defensible claim that survives all the C2 evidence:

> *Window-level sparsification recovers global feature directions from
> coupled or noisy emissions that per-token sparsification misses, because
> the window's top-K budget aggregates redundant evidence across positions
> for the same underlying latent. The mechanism is implicit budget
> extension via window aggregation, not temporal pattern detection: the
> recovery advantage exists at ρ = 0, where there is no temporal
> autocorrelation in the firing pattern for a temporal architecture to
> exploit.*

Suggested section ordering:

1. **Setup B (noisy emissions, denoising) — headline.** Use
   `c2_noisy_probe_scatter.png` as the lead visual (TXC above y=x, TopK on
   y=x, T-SAE below). Back with the denoising-ratio panel showing
   TXC-pro T_max=10 crossing above 1.0. Frame as "TXC latents act as
   denoised estimators of the hidden Markov state."
2. **Setup A + D (coupled features) — secondary.** gAUC-vs-k_pos plots
   on Setup A and Setup D pB05_np10. Frame as "the same window-aggregation
   mechanism recovers global features under deterministic OR-gate
   coupling; the gap is largest in the high-overlap noisy regime."
3. **Setup C (ρ-sweep) — integrity move.** Include explicitly. Frame as
   "we tested whether the recovery advantage depends on temporal
   autocorrelation in the firing pattern; it does not." Disclose the
   Effect-1 reading. This pre-empts a reviewer asking exactly this
   question.
4. **Setup E (engineered hierarchical) — appendix.** Demote. The scatter
   doesn't show the clean tier separation the prose claims, and TopK SAE
   catches up at k ≥ 5 because d_sae = K_g + K_l = 40.

## Robustness checks worth doing before lock-in

- Setup D's n_steps cutover (8k vs 30k) is in caveats; verify the
  pB05_np10 gAUC win at k_pos ∈ {1, 2, 3, 4} reproduces at n_steps = 30k
  on at least one seed before the paper claims this regime as headline.
- Setup B parameter mismatch with the wasteland source: confirm the
  qualitative tier separation (TXC > floor, others = floor) reproduces
  at the wasteland p_B parameter the paper cites; the absolute denoising
  ratio numbers will shift but the cross-arch ordering should be stable.
- Setup C: now that ρ ∈ {0.0, 0.3, 0.6, 0.7, 0.9} cells should have
  finished, regenerate the plot once more from a fresh leaderboard pull
  to confirm the flat-vs-ρ pattern holds with full sample size.

## Plots I have NOT inspected

- Anything outside the eight files referenced from `c2.md`. If there are
  Setup C plots at higher k_pos, or per-T-modulation plots that
  decompose the T-modulation claim, those would be useful. Same for the
  HUNT-phase per-cell tables in `hunt_summary.json` if the headline
  pB05_np10 number is sensitive to the n_steps cutover.

If Han wants me to look at any of these, ping back.
