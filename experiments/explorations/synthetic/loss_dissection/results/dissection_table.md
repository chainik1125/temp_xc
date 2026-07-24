# Loss-dissection component table (mechanical, CARD § 5)

| bench | metric | +matryoshka | +contrastive | +both |
|---|---|---|---|---|
| backtracking | lambda_recovery (primary) | NEUTRAL (max|D| -0.009±0.002) | NEUTRAL (max|D| +0.006±0.003) | NEUTRAL (max|D| -0.007±0.002) |
| backtracking | nmse | NEUTRAL (max|D| -0.013±0.002) | HURTS (max|D| -0.025±0.002) | NEUTRAL (max|D| -0.018±0.004) |
| backtracking | eauc | HURTS (max|D| -0.076±0.028) | HELPS (max|D| +0.099±0.009) | HURTS (max|D| -0.090±0.043) |
| frequency | velocity_recovery (primary) | HURTS (max|D| -0.094±0.036) | HELPS (max|D| +0.093±0.038) | NEUTRAL (max|D| +0.124±0.046) |
| frequency | nmse | NEUTRAL (max|D| -0.017±0.007) | NEUTRAL (max|D| +0.042±0.009) | NEUTRAL (max|D| +0.021±0.017) |
| frequency | eauc | NEUTRAL (max|D| -0.013±0.020) | NEUTRAL (max|D| +0.030±0.018) | NEUTRAL (max|D| +0.030±0.024) |
| phasepair | sign_recovery (primary) | HURTS (max|D| -0.507±0.528) | NEUTRAL (max|D| -0.081±0.091) | NEUTRAL (max|D| -0.480±0.420) |
| phasepair | pair_recovery | NEUTRAL (max|D| +0.068±0.095) | MIXED (max|D| -0.243±0.131) | NEUTRAL (max|D| -0.097±0.120) |
| phasepair | nmse | NEUTRAL (max|D| -0.018±0.005) | NEUTRAL (max|D| -0.023±0.001) | NEUTRAL (max|D| -0.022±0.011) |
| phasepair | eauc | NEUTRAL (max|D| -0.025±0.004) | NEUTRAL (max|D| -0.007±0.005) | NEUTRAL (max|D| -0.025±0.011) |
| recipe_instruction_phase_runs | equality_residual_recovery (primary) | HELPS (max|D| +0.506±0.417) | NEUTRAL (max|D| -0.293±0.681) | MIXED (max|D| +0.462±0.189) |
| recipe_instruction_phase_runs | phase_recovery | HURTS (max|D| -0.103±0.061) | NEUTRAL (max|D| -0.073±0.161) | NEUTRAL (max|D| -0.073±0.056) |
| recipe_instruction_phase_runs | nmse | HURTS (max|D| -0.027±0.004) | NEUTRAL (max|D| +0.014±0.008) | HURTS (max|D| -0.027±0.001) |
| recipe_instruction_phase_runs | eauc | NEUTRAL (max|D| -0.189±0.251) | NEUTRAL (max|D| +0.040±0.074) | MIXED (max|D| -0.262±0.195) |
| multilane | multilane_recovery (primary) | NEUTRAL (max|D| -0.049±0.009) | NEUTRAL (max|D| +0.031±0.014) | NEUTRAL (max|D| -0.053±0.002) |
| multilane | nmse | NEUTRAL (max|D| -0.011±0.002) | NEUTRAL (max|D| -0.009±0.009) | NEUTRAL (max|D| -0.013±0.003) |
| multilane | eauc | NEUTRAL (max|D| +0.044±0.021) | NEUTRAL (max|D| +0.108±0.015) | NEUTRAL (max|D| +0.054±0.027) |

## Gates
- backtracking: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
- frequency: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
- phasepair: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
- recipe_instruction_phase_runs: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
- multilane: Gate B PASS (9/9 cells); untrained guard PASS (max |diff| 0.00e+00)
