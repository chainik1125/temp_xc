| source | arm | mode | mining score (t) | gc base | Δgc peak (coh, Sonnet) | at α | 95% CI | Δgc peak (coh rows) | at α | mean abs Δgc | Δgc peak (coh, run) | at α | Δgc peak (no floor) | at α | gc min | event-rate base → peak | coherent cells (run/Sonnet) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w6_dsm_f10440_union` | w6_dsm | union | -0.113 (-14.7) | 1.30 | +0.500 | -7 | [+0.00, +1.00] | +0.500 | -7 | 0.400 | -0.900 | -16 | -1.100 | 16 | 0.20 | 0.85 → 0.40 | 3/13 of 25 |
| `w6_recon_f10063_union` | w6_recon | union | +0.152 (16.1) | 1.45 | +0.400 | 8 | [+0.00, +0.90] | +0.800 | 12 | — | — | — | -0.900 | -16 | 0.55 | 0.85 → — | 0/19 of 25 |
| `w6_bayes_f8209_pos0` | w6_bayes | pos0 | -0.221 (-14.4) | 1.30 | -0.350 | 3 | [-0.65, -0.05] | -0.350 | 3 | 0.725 | -1.300 | 16 | -1.300 | 16 | 0.00 | 0.80 → 0.00 | 2/13 of 25 |
| `w6_dsm_f10440_pos0` | w6_dsm | pos0 | -0.113 (-14.7) | 1.40 | +0.350 | -8 | [-0.10, +0.85] | +0.350 | -8 | 0.775 | -1.200 | -16 | -1.200 | -16 | 0.20 | 0.85 → 0.20 | 2/16 of 25 |
| `w6_bayes_f8209_union` | w6_bayes | union | -0.221 (-14.4) | 1.45 | -0.300 | -1 | [-0.60, +0.00] | +0.491 | -8 | 0.967 | -1.450 | 16 | -1.450 | 16 | 0.00 | 0.85 → 0.00 | 3/14 of 25 |
| `w6_recon_f10063_pos0` | w6_recon | pos0 | +0.152 (16.1) | 1.25 | +0.300 | -8 | [-0.10, +0.70] | +0.868 | -10 | 0.075 | -0.100 | 16 | +0.800 | -10 | 0.60 | 0.80 → 0.80 | 2/16 of 25 |
| `w6_dsm_f10440_pos0_proj` | w6_dsm | pos0 | -0.113 (-14.7) | 0.00 | — | — | — | — | — | — | — | — | +0.000 | -16 | 0.00 | 0.00 → — | 0/0 of 25 |

## Symmetric/antisymmetric decomposition vs the wave-1 random control

Produced by `symmetry.py` on the six unprojected wave-2 sources against the wave-1
`control_random` rows (same 20 eval prompts, same 25-magnitude grid, same protocol —
the prompt pairing holds exactly; the paired prompt-resampling bootstrap resamples the
same prompt multiset for arm and control). Full output: `symmetry.json` in this
directory (includes the control, the `_FOLD` variants, and per-|α| rows). The
projected arm is excluded (settled projector-damage row). Components are means over
all 12 paired |α| values (see `symmetry.json` for the Sonnet-coherent-pairs view).

Sign conventions, both reported per the sign-inversion caveat: `anti > 0` means
negative α induces more backtracking (the harness convention). Two of three w6
features were mined with negative meandiff scores (`w6_dsm` f10440 −0.113,
`w6_bayes` f8209 −0.221), so their α axes are flipped in the **mined-sign-folded**
columns — folding makes magnitudes comparable across arms, but the raw (unfolded)
value is shown next to it. `w6_recon` f10063 was mined positive: folded = raw.
Wave-1 control: mean sym +0.148, mean anti −0.010, gc(0) = 1.30.

| source | mined sign | mean sym | mean anti (raw) | excess anti RAW [95% CI] | excess anti FOLDED [95% CI] | survives control? |
|---|---|---|---|---|---|---|
| `w6_bayes_f8209_pos0` | − | −0.123 | +0.073 | **+0.083 [+0.000, +0.165]** | −0.063 [−0.177, +0.052] | no (raw CI touches zero, and the direction is opposite the mined sign; folded is null) |
| `w6_dsm_f10440_pos0` | − | −0.192 | +0.071 | +0.081 [−0.035, +0.204] | −0.060 [−0.190, +0.056] | no |
| `w6_bayes_f8209_union` | − | −0.200 | +0.063 | +0.073 [−0.054, +0.192] | −0.052 [−0.165, +0.071] | no |
| `w6_dsm_f10440_union` | − | −0.048 | +0.052 | +0.062 [−0.052, +0.167] | −0.042 [−0.162, +0.058] | no |
| `w6_recon_f10063_pos0` | + | +0.119 | −0.027 | −0.017 [−0.127, +0.087] | (= raw) | no |
| `w6_recon_f10063_union` | + | −0.021 | −0.029 | −0.019 [−0.144, +0.106] | (= raw) | no |

Reading: no wave-2 source shows a directional (antisymmetric) effect beyond the
norm-matched random control once the mined sign is folded in. The one nominally
significant raw value (`w6_bayes_f8209_pos0`, CI lower bound at +0.000) points in
the direction its negative mining score says it should NOT — all four negative-mined
sources show raw excess-anti of the same sign as the positive-mined arms would,
which is the signature of a nonspecific perturbation, not of the mined feature's
direction carrying meaning. The raw Sonnet-floor peaks in the table above
(+0.30 to +0.50) sit at or below the wave-1 random control's raw peak (+0.450) and
should be read accordingly.
