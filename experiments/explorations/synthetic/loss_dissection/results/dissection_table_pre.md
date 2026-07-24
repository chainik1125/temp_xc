# Loss-dissection component table — pre family (mechanical, CARD § 5 + § 9)

| bench | metric | +matryoshka | +contrastive | +both |
|---|---|---|---|---|
| backtracking | lambda_recovery (primary) | NEUTRAL (max|D| -0.003±0.004) | NEUTRAL (max|D| +0.003±0.001) | NEUTRAL (max|D| +0.002±0.003) |
| backtracking | nmse | NEUTRAL (max|D| -0.014±0.002) | NEUTRAL (max|D| -0.009±0.009) | NEUTRAL (max|D| -0.013±0.003) |
| backtracking | eauc | NEUTRAL (max|D| -0.073±0.014) | HELPS (max|D| +0.129±0.044) | NEUTRAL (max|D| +0.060±0.028) |
| frequency | velocity_recovery (primary) | NEUTRAL (max|D| -0.019±0.004) | NEUTRAL (max|D| -0.047±0.039) | NEUTRAL (max|D| -0.016±0.003) |
| frequency | nmse | HURTS (max|D| -0.396±0.437) | NEUTRAL (max|D| -0.061±0.040) | HURTS (max|D| -0.371±0.252) |
| frequency | eauc | NEUTRAL (max|D| +0.030±0.008) | NEUTRAL (max|D| +0.030±0.016) | NEUTRAL (max|D| +0.029±0.005) |
| phasepair | sign_recovery (primary) | NEUTRAL (max|D| +0.023±0.088) | NEUTRAL (max|D| +0.022±0.015) | NEUTRAL (max|D| +0.020±0.050) |
| phasepair | pair_recovery | NEUTRAL (max|D| -0.010±0.025) | NEUTRAL (max|D| -0.058±0.073) | NEUTRAL (max|D| -0.014±0.006) |
| phasepair | nmse | HURTS (max|D| -0.026±0.010) | HURTS (max|D| -0.026±0.007) | NEUTRAL (max|D| -0.076±0.113) |
| phasepair | eauc | NEUTRAL (max|D| +0.027±0.039) | NEUTRAL (max|D| -0.029±0.025) | NEUTRAL (max|D| +0.030±0.046) |
| recipe_instruction_phase_runs | equality_residual_recovery (primary) | NEUTRAL (max|D| -0.023±0.055) | NEUTRAL (max|D| +0.038±0.032) | NEUTRAL (max|D| -0.034±0.057) |
| recipe_instruction_phase_runs | phase_recovery | NEUTRAL (max|D| -0.023±0.022) | NEUTRAL (max|D| +0.013±0.047) | NEUTRAL (max|D| -0.042±0.024) |
| recipe_instruction_phase_runs | nmse | HURTS (max|D| -0.038±0.003) | NEUTRAL (max|D| -0.006±0.002) | HURTS (max|D| -0.039±0.004) |
| recipe_instruction_phase_runs | eauc | HURTS (max|D| -0.197±0.049) | NEUTRAL (max|D| -0.042±0.041) | HURTS (max|D| -0.185±0.094) |
| multilane | multilane_recovery (primary) | NEUTRAL (max|D| -0.008±0.007) | NEUTRAL (max|D| -0.006±0.003) | NEUTRAL (max|D| -0.008±0.003) |
| multilane | nmse | HURTS (max|D| -0.047±0.001) | NEUTRAL (max|D| -0.009±0.005) | HURTS (max|D| -0.047±0.003) |
| multilane | eauc | HELPS (max|D| +0.246±0.005) | NEUTRAL (max|D| +0.056±0.063) | HELPS (max|D| +0.222±0.030) |

## Gates
- backtracking: Gate B PASS (8/8 cells); untrained guard PASS (max |diff| 0.00e+00)
- frequency: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
- phasepair: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
- recipe_instruction_phase_runs: Gate B PASS (8/8 cells); untrained guard PASS (max |diff| 0.00e+00)
- multilane: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
