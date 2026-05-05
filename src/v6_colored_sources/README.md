# Experiment 4: delayed temporally-colored sources

Implements the "theorem-first" synthetic diagnostic from
`docs/aniket/experiments/synthetic/notes/txc_synthetic_experiment_proposal_updated.tex`
(Section "Experiment 4", lines 655–1043). The reference `.tex` is currently
untracked; the math summary below pins the equations this code targets.

## Setup

- Orthonormal basis `F ∈ R^{d×N}`, `F^T F = I_N`. Stages 1/2 use `d = N = 128`.
- Per-coordinate AR(1) latent: `z_{t+1, i} = ρ_i · z_{t, i} + sqrt(1 - ρ_i²) · η_{t, i}`,
  with `η ~ N(0, I)` and distinct `ρ_i`.
- Observation: `x_t = F z_t + σ ε_t`, `ε_t ~ N(0, I_d)`.
- Delayed variant: `D` independent residue classes mod `D`, each running its own AR(1).

## Population covariances

- `C_0 = (1 + σ²) I_d` (one-token marginal — independent of `F` when `d = N`).
- `C_ℓ = 0` for `0 < ℓ < D`.
- `C_D = F · diag(ρ) · F^T` (eigenvectors = true basis, eigenvalues = autocorrelations).

## Claim and bound

- **Local impossibility:** Any algorithm whose training data are iid samples
  from `P(x_t)` outputs directions independent of `F`. Squared recovery is
  bounded by `Rec ≲ log(H) / N` regardless of compute or sample size.
- **Temporal recoverability:** With distinct `ρ_i` and eigengap
  `γ = min_{i ≠ j} |ρ_i - ρ_j|`, the eigenvectors of `Ĉ_D` recover `F` with
  angular error `~ε / γ` once `T_eff ≫ (1+σ²)² N / γ²`.

## Recovery metric

- Sign-invariant: `Rec(F, F̂) = (1/N) Σ_i max_j |⟨f_i, f̂_j⟩|²`.
- Chance adjustment: `S_chance ≈ log(H) / N`,
  `S_adj = max(0, (S - S_chance) / (1 - S_chance))`.

## Phase transition

Window-local methods need `W ≥ D + 1` to see the informative lag. The Stage 2
headline figure plots `S_adj` vs `W` with one curve per `D ∈ {1, 2, 4, 8}`,
showing a sharp jump at `W = D + 1`.

## Module layout

| File | Purpose |
|---|---|
| `configs.py` | `ColoredSourceConfig` dataclass, `rho_schedule()` helper |
| `colored_sources.py` | basis sampling, AR-chain sampler, full dataset assembly |
| `theory.py` | population `C_lag`, spectral oracle |
| `metrics.py` | empirical lag covariance, squared recovery, chance adjustment |
| `validation.py` | five pre-training gating checks + CLI |
| `data_adapter.py` | duck-typed shim for `temporal_crosscoders.data.CachedDataSource` |
| `train_runner.py` | `d = N = 128` override + `train_stacked_sae` / `train_txcdr` driver |
| `run_experiment.py` | Stage 1 / Stage 2 sweep entrypoints |
| `plot_results.py` | phase-transition figures |

## Running

```bash
# Stage 0: validation gates (CPU, ~60s)
uv run python -m src.v6_colored_sources.validation

# Stage 1: lag-1 phase transition (GPU recommended)
uv run python -m src.v6_colored_sources.run_experiment --stage 1

# Stage 2: D × W headline figure
uv run python -m src.v6_colored_sources.run_experiment --stage 2
```
