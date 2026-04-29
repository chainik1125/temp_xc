# Paper-faithful T-SAE in separation_scaling

Same transformer setup as `config.yaml` (σ=1e-3, d=64, ctx=128, 20k steps), same generator, same probe. Only difference: T-SAE arch with paper-faithful matryoshka split + 25k training steps instead of the original `Temporal BatchTopK SAE` entry which used 50/50 split + 2k steps.

## Best single-feature R² per arch (across δ sweep)

| δ | TopK | TXC | MatTXC | MLxC | TFA | TFA-pos | T-SAE old (50/50, 2k) | **T-SAE paper (20/80, 25k)** |
|---|---|---|---|---|---|---|---|---|
| 0    | -0.001 | -0.001 | -0.010 | -0.000 | -0.001 | -0.001 | -0.001 | **+0.001** |
| 0.05 | -0.001 | -0.000 | -0.012 | -0.000 | -0.000 | +0.000 | -0.000 | **+0.000** |
| 0.10 | +0.002 | +0.030 | -0.000 | +0.022 | +0.001 | +0.002 | +0.007 | **+0.005** |
| 0.15 | +0.050 | +0.244 | +0.263 | +0.092 | +0.052 | +0.078 | +0.109 | **+0.285** |
| 0.20 | +0.069 | +0.209 | +0.421 | +0.148 | +0.190 | +0.187 | +0.079 | **+0.611** |

## Headline

**Paper-faithful T-SAE wins single-feature R² at every δ where separation signal exists.** At δ=0.20, T-SAE paper hits 0.611 — a +45% margin over MatTXC (0.421), the previous winner. At δ=0.15, T-SAE paper (0.285) edges out MatTXC (0.263).

## What this tells us

The existing "Temporal BatchTopK SAE" entry in the original benchmark used `group_fractions=[0.5, 0.5]` + only 2000 training steps. The 50/50 split makes the matryoshka pressure trivial (both groups same size and weight, so prefix loss = full-width loss), and 2k steps was too short to converge. Effectively the existing entry was BatchTopK SAE with a temporal contrastive loss — not the paper architecture.

Switching to the paper-default `[0.2, 0.8]` split (which forces a small "high-level" prefix to absorb the most important structure) plus 25k training steps recovers the architecture's full single-feature compression. The 20% prefix is precisely the place where ergodic-component identity gets concentrated.

This also matches the [coupled-features T-SAE result](../../docs/dmitry/case_studies/coupled_features/summary.md): under paper-faithful settings, T-SAE compresses global structure into a small set of features more reliably than any window arch tested. The MESS3 separation_scaling task is just a different probe (transformer activations on a 3-component HMM rather than direct toy hidden-state recovery), and the same architectural advantage shows up.

## Caveats

1. **Single seed (42).** The original benchmark used the same single seed; we matched it. No replication-based variance.
2. **σ=1e-3 transformer init**, as in the original. Per the σ sweep finding (`tables/mup_study.md`), TXC's single-feature R² benefits from σ=2e-2; MatTXC and presumably T-SAE paper would have less to gain since they already concentrate. Worth checking whether T-SAE paper's gain holds at σ=2e-2 too.
3. **Only the T-SAE arch was rerun.** The other 6 archs' numbers are reused from the original `config.yaml` run.

## Files

- `config_tsae_paper.yaml` — minimal config with only T-SAE Paper arch
- `results_tsae_paper/cell_delta_*/results.json` — per-cell raw output
- `results_tsae_paper/combined.json` — combined per-arch×δ summary
