---
author: dmitry
date: 2026-05-05
tags:
  - proposal
  - in-progress
---

## Context

Implement **Experiment 4** from `docs/aniket/experiments/synthetic/notes/txc_synthetic_experiment_proposal_updated.tex` (lines 655–1043) — *"Delayed temporally-colored sources with a provable local bound."* This is the proposal's theorem-first synthetic diagnostic: a regime where local one-token feature-direction recovery is information-theoretically bounded near chance, while temporal recovery from lagged covariance is possible. The goal is to produce the headline phase-transition figure showing TXC recovery jumping at window length `W = D + 1`, with the spectral oracle as ceiling and a local SAE / random-vector baseline as floor.

User decisions (already made):
- **Scope:** Stage 0 + 1 + 2 (data + validation + delay sweep). Stage 3 (σ / eigengap / N=256 sweeps) deferred.
- **Compute:** Run training sweeps on remote `a40_2` (`ssh a40_temp_xc` per memory `reference_a40_txc_1.md`). Stage 0 validation runs locally.
- **Geometry:** `d = N = 128` square. The clean isotropy theorem requires this.

The proposal `.tex` is **untracked** — it lives at the main checkout but not in this worktree. We will inline a short math-only summary into the new package's README so the implementation isn't pinned to a fragile path.

## Reference equations (from proposal)

- Basis: `F ∈ R^{d×N}`, `F^T F = I_N`, `d = N` for Stage 1/2 (proposal lines 680–687).
- Lag-1 generator: `z_{t+1, i} = ρ_i z_{t,i} + sqrt(1 - ρ_i²) η_{t,i}`, `η ~ N(0, I)`, `ρ_i` distinct (lines 712–715).
- Observation: `x_t = F z_t + σ ε_t`, `ε_t ~ N(0, I_d)` (lines 700–703).
- Delayed variant: `D` independent residue classes mod `D`, each running its own AR(1) (line 850).
- Population covariances: `C_0 = (1+σ²) I_d` (line 720); `C_ℓ = 0` for `0 < ℓ < D` (line 858); `C_D = F diag(ρ) F^T` (line 864).
- Recovery score: `Rec(F, F̂) = (1/N) Σ_i max_j |⟨f_i, f̂_j⟩|²` (line 752).
- Chance adjustment: `S_chance ≈ log(H)/N`, `S_adj = (S - S_chance) / (1 - S_chance)` (lines 946–951).
- Phase transition: window-local methods need `W ≥ D + 1` to see the informative lag (lines 869–880).

## Naming and layout

`src/v4_reset_process_validation` already exists, so the next free version slot is **`src/v6_colored_sources`**. Mirror the layout of `src/v5_hmm_sae_baseline`.

```
src/v6_colored_sources/
  __init__.py
  README.md             # short math summary + reference to proposal section
  configs.py            # ColoredSourceConfig dataclass
  colored_sources.py    # data generator (lag-1 + delayed)
  theory.py             # population covariances + spectral oracle
  metrics.py            # squared recovery, chance-adjusted, lagged-cov estimator
  validation.py         # Stage 0 pre-training checks
  data_adapter.py       # bridge to temporal_crosscoders.data.CachedDataSource
  train_runner.py       # patches temporal_crosscoders.config + runs training
  run_experiment.py     # Stage 1 / Stage 2 sweep entrypoints
  plot_results.py       # phase-transition figure
tests/test_v6_colored_sources.py
```

The proposal will be referenced from `src/v6_colored_sources/README.md` (math summary + line citations) — durable even if the `.tex` moves.

## Stage 0 — scaffolding, generator, oracle, validation

**Goal:** generator + spectral oracle that pass the proposal's pre-training checklist (lines 1227–1234) before any training runs.

### Files (Stage 0)

| File | What it does |
|---|---|
| `src/v6_colored_sources/configs.py` | `ColoredSourceConfig(N, d, D, sigma, rho_min, rho_max, n_seq, T_chain, seed)`; helper `rho_schedule()` returns `linspace(rho_min, rho_max, N)`. |
| `src/v6_colored_sources/colored_sources.py` | `generate_orthonormal_basis()`, `sample_ar_chains()`, `generate_dataset(cfg) -> {F, z, x, rho, config}`. |
| `src/v6_colored_sources/theory.py` | `population_C_lag(F, rho, sigma, lag, D)`, `spectral_oracle(x, lag_D, n_components)`. |
| `src/v6_colored_sources/metrics.py` | `empirical_lag_covariance(x, lag)`, `squared_axis_recovery(F, F_hat)`, `chance_adjusted_recovery(S, N, H)`. |
| `src/v6_colored_sources/validation.py` | `check_one_token_isotropy`, `check_short_lag_zero`, `check_oracle_recovers_basis`, `check_shuffle_destroys_oracle`, `check_random_dictionary_chance`. |
| `tests/test_v6_colored_sources.py` | Unit tests for each invariant; fast-path with small `N`/`T`. |

### Key signatures and math

```python
# colored_sources.py
def generate_orthonormal_basis(N: int, d: int, *, seed: int) -> Tensor:
    # Use torch.linalg.qr(torch.randn(d, d)) so F^T F = I exactly.
    # DO NOT use src/shared/orthogonalize.py — it is gradient-based and only
    # approximate, which would invalidate the impossibility theorem.
    # Returns (N, d).

def sample_ar_chains(
    N: int, n_seq: int, T_chain: int, rho: Tensor, D: int, *, seed: int
) -> Tensor:
    # Per-residue-class AR(1):
    #   z[:, r, i]      ~ N(0, 1)                                       # stationary init
    #   z[:, r+kD, i]   = rho[i] * z[:, r+(k-1)D, i] + sqrt(1-rho[i]^2) * eta
    # Residue classes r ∈ {0..D-1} are independent.
    # No burn-in needed because stationary init is exact.
    # Returns (n_seq, T_chain, N).

def generate_dataset(cfg: ColoredSourceConfig) -> dict:
    # Returns {"features": F (N,d), "z": z (n_seq,T,N), "x": x (n_seq,T,d),
    #          "rho": rho (N,), "config": cfg}.
    # x = z @ F + sigma * randn(n_seq, T, d).
```

```python
# theory.py
def population_C_lag(F, rho, sigma, lag, D) -> Tensor:    # (d, d)
    if lag == 0:                 return F.T @ F + sigma**2 * eye(d)   # = (1+σ²) I when N=d
    if 0 < lag < D:              return zeros(d, d)
    if lag == D:                 return F.T @ diag(rho) @ F
    # higher multiples: F.T @ diag(rho ** (lag // D)) @ F

def spectral_oracle(x, lag_D, n_components) -> Tensor:    # (n_components, d)
    C_hat = empirical_lag_covariance(x, lag_D)
    C_sym = 0.5 * (C_hat + C_hat.T)
    eigvals, eigvecs = torch.linalg.eigh(C_sym)
    # Sort by |eigvals| descending; return top n_components rows (each row = direction).
```

```python
# metrics.py
def empirical_lag_covariance(x, lag) -> Tensor:   # (d, d)
    # x: (n_seq, T, d). Pool: C = (1 / (n_seq*(T-lag))) * Σ_s Σ_t outer(x[s,t+lag], x[s,t]).

def squared_axis_recovery(F, F_hat) -> float:
    # Proposal eq (5.6): (1/N) Σ_i max_j |<f_i, f_hat_j>|².  Both inputs unit-rowed.

def chance_adjusted_recovery(S: float, N: int, H: int) -> float:
    # S_adj = max(0, (S - log(H)/N) / (1 - log(H)/N)).
```

### Gating before Stage 1

Run on `D = 2, N = d = 128, σ = 0.1, n_seq = 128, T_chain = 512`. All five must pass:

1. `check_one_token_isotropy`: off-diagonal max < `0.05 (1+σ²)`; eigval max/min ratio < 1.1.
2. `check_short_lag_zero`: `||C_ℓ||_op / ||C_0||_op < 0.05` for `0 < ℓ < D`; `||C_D||_op / ||C_0||_op > 0.5 mean(|ρ|)`.
3. `check_oracle_recovers_basis`: squared recovery > 0.9.
4. `check_shuffle_destroys_oracle`: squared recovery < `2 log(N)/N`.
5. `check_random_dictionary_chance`: empirical mean within ±20% of `log(H)/N`.

Compute: pure CPU torch, well under 60s for the full suite.

## Stage 1 — single-config phase-transition (lag-1)

**Goal:** one figure showing TXC recovery vs window length `W` for `D = 1`, with stacked-SAE baseline and spectral-oracle ceiling.

### Files (new in Stage 1)

| File | What it does |
|---|---|
| `src/v6_colored_sources/data_adapter.py` | `ColoredSourceCache` — duck-typed replacement for `temporal_crosscoders.data.CachedDataSource`: holds pre-generated `act_chains` and exposes `sample_windows(B, T)`. |
| `src/v6_colored_sources/train_runner.py` | Single function `train_pair(cfg, W, k, n_steps, device)` that overrides `temporal_crosscoders.config` constants (`NUM_FEATS=128, HIDDEN_DIM=128, D_SAE=128`), then calls `train_stacked_sae` and `train_txcdr`. Returns `(stacked_history, txc_history, stacked_model, txc_model)`. |
| `src/v6_colored_sources/run_experiment.py` | `run_stage1(cfg, W_grid, n_steps)` — loops over `W`, trains both architectures, computes recovery + chance-adjusted score + oracle ceiling per cell, dumps `results/v6_colored_sources/stage1.json`. |
| `src/v6_colored_sources/plot_results.py` | `plot_phase_transition(results, out)` — `Rec_adj` vs `W`, three layers (TXC solid, stacked-SAE dotted, oracle dashed horizontal). |

### Adapter contract

`temporal_crosscoders.data.CachedDataSource` exposes `act_chains: (num_chains, chain_length, d)` and `sample_windows(B, T) -> (B, T, d)` (verified at `temporal_crosscoders/data.py:140` and `:191`). `CachedWindowIterator` only calls `cache.sample_windows(...)`. We make `ColoredSourceCache` duck-type that surface — no inheritance — and skip the `toy_model` path entirely (`x` is already in observation space).

`CachedWindowIterator.__next__` may call `cache.refresh()` periodically (see `CACHE_REFRESH_INTERVAL` in `temporal_crosscoders/config.py`). Implement `ColoredSourceCache.refresh()` as a deterministic regen using the same seed so `F` stays valid; or set the interval to 0 in `train_runner.py` to disable.

### Config-override approach

`temporal_crosscoders/config.py` hardcodes `NUM_FEATS = 128`, `HIDDEN_DIM = 256`, `D_SAE = NUM_FEATS`. We need `HIDDEN_DIM = NUM_FEATS = 128` for the square geometry. `train_runner.py` does this monkey-patch once per process (assigning the module-level constants before `train_stacked_sae` / `train_txcdr` are called, which import them at function-call time via the `DEVICE`, `BATCH_SIZE`, etc. globals). If that turns out to be brittle (e.g. the model classes capture the constants at import), fall back to writing thin reimplementations in `train_runner.py` that mirror `temporal_crosscoders/train.py:61` and `:139` but accept explicit `d_in`, `d_sae` arguments.

### Stage-1 sweep

| Knob | Value |
|---|---|
| `(N, d)` | `(128, 128)` |
| `D` | `1` |
| `σ` | `0.1` |
| `(ρ_min, ρ_max)` | `(0.1, 0.9)` |
| `n_seq, T_chain` | `128, 512` |
| `H` (dict size) | `128` (square — deferred overcomplete to Stage 3) |
| `k` (active per window) | `8` (TXC), stacked-SAE `k = 8` per position |
| `W` grid | `{1, 2, 4, 8, 16}` |
| `n_steps` | start `30k`; bump to `65k` (TXC default) if recovery hasn't plateaued |

Architectures:
- `W = 1`: stacked SAE only (TXC at `T=1` degenerates).
- `W ≥ 2`: both `train_stacked_sae` and `train_txcdr`.

### Per-cell metrics

- `recovery_squared` (Stage 0 metric, headline)
- `S_adj`
- `recovery_auc` (kept for cross-experiment continuity with v5's `feature_recovery_score`)
- Spectral oracle on the same `x`: `recovery_squared_oracle`, `S_adj_oracle` (ceiling)
- Random-vector baseline: `recovery_squared_random` averaged over 10 draws (floor)

### Gating before Stage 2

- `S_adj` for stacked SAE at `W = 1` is within ±0.05 of 0.
- `S_adj` for TXC at `W = 2` and the oracle both > 0.5.
- Recovery curves saved to `results/v6_colored_sources/stage1.json`.

Compute (Stage 1): 5 W values × ≤2 architectures = 9 training runs. On a40_2 each is ~3–5 min at 30k steps → ~30–45 min total.

## Stage 2 — delay sweep + headline figure

**Goal:** the headline figure: `S_adj` vs `W`, one curve per `D ∈ {1, 2, 4, 8}`, showing a sharp jump at `W = D + 1`.

### Files (Stage 2 changes only)

- `src/v6_colored_sources/run_experiment.py` — add `run_stage2(cfg, D_grid, W_grid, n_steps)` that reuses the Stage-1 cell function across the (D, W) grid. Spectral oracle computed once per `D`. Output: `results/v6_colored_sources/stage2.json`.
- `src/v6_colored_sources/plot_results.py` — add `plot_phase_transition_by_delay(results, out)`: x-axis `W`, y-axis `S_adj`, color = `D`, vertical dashed line at `W = D + 1` per curve, oracle ceiling per `D`, stacked-SAE baseline (lighter shade).

### Stage-2 sweep

`D ∈ {1, 2, 4, 8}`, `W ∈ {1, 2, 4, 8, 16}`. Other knobs frozen at Stage-1 values. **Caveat:** `T_chain = 512` gives only `512 / D` independent lag-`D` samples per chain. For `D = 8` that's 64 lag samples per chain × 128 chains = 8192 lag samples; covariance estimate has standard error `~sqrt(N / 8192) ≈ 0.12` (from the proposal's bound at line 826), which is borderline. Bump `T_chain` to `1024` or `n_seq` to `256` for `D ∈ {4, 8}` cells if Stage-0 oracle check fails at the chosen sample size.

### Cell budget

20 cells × ~1.5 architectures avg = ~30 training runs ≈ 90 min on a40_2.

### Gating before Stage 3 (out of scope here)

- Sharp jump in TXC `S_adj` between `W = D` and `W = D + 1` for at least 3 of 4 `D` values.
- Stacked SAE dominated by TXC at every `W ≥ D + 1` cell.

## Critical files (paths)

Files I will create:
- `/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc/.claude/worktrees/dmitry-synthetic/src/v6_colored_sources/__init__.py`
- `.../src/v6_colored_sources/README.md`
- `.../src/v6_colored_sources/configs.py`
- `.../src/v6_colored_sources/colored_sources.py`
- `.../src/v6_colored_sources/theory.py`
- `.../src/v6_colored_sources/metrics.py`
- `.../src/v6_colored_sources/validation.py`
- `.../src/v6_colored_sources/data_adapter.py`
- `.../src/v6_colored_sources/train_runner.py`
- `.../src/v6_colored_sources/run_experiment.py`
- `.../src/v6_colored_sources/plot_results.py`
- `.../tests/test_v6_colored_sources.py`

Files I will read but not edit (existing infrastructure to call):
- `temporal_crosscoders/data.py:140` (`CachedDataSource`), `:205` (`CachedWindowIterator`)
- `temporal_crosscoders/train.py:61` (`train_stacked_sae`), `:139` (`train_txcdr`)
- `temporal_crosscoders/config.py` (constants we must override at runtime)
- `src/v5_hmm_sae_baseline/metrics.py:14` (`feature_recovery_score`) — call for the AUC continuity metric
- `src/utils/seed.py` (`set_seed`)

Plan-file relocation: per memory `feedback_plan_file_location.md`, after approval move this plan from `~/.claude/plans/ethereal-orbiting-cocke.md` to `docs/dmitry/synthetic/2026-05-05_experiment_4/plan.md` with frontmatter (`author: dmitry`, `date: 2026-05-05`, `tags: [proposal, in-progress]`), strip the H1, and commit.

## Verification (end-to-end)

1. `uv run pytest tests/test_v6_colored_sources.py -v` — Stage 0 invariants, fast.
2. `uv run python -m src.v6_colored_sources.validation --gate stage0` — runs the five gating checks and prints pass/fail. Refuse to proceed if any fail.
3. `uv run python -m src.v6_colored_sources.run_experiment --stage 1` — produces `results/v6_colored_sources/stage1.json` and `plots/phase_transition_stage1.pdf`.
4. `uv run python -m src.v6_colored_sources.run_experiment --stage 2` — produces `results/v6_colored_sources/stage2.json` and `plots/phase_transition_stage2.pdf`.
5. Visual check on the Stage 2 plot: TXC curves jump near `W = D + 1`; stacked-SAE stays near 0 across all `W`; oracle ceiling sits near 1 for every `D`.

## Risks and mitigations

1. **`temporal_crosscoders/config.py` constants captured at import.** If the monkey-patch in `train_runner.py` doesn't take effect because models read `HIDDEN_DIM`/`D_SAE` at module-load time, fall back to thin reimplementations of `train_stacked_sae`/`train_txcdr` in `train_runner.py` with explicit dimension args. Verify on a single `W = 2` cell before launching the full sweep.
2. **`CachedWindowIterator.refresh` desync.** `refresh()` in our adapter must regenerate from the same seed so `F` stays valid. Easier path: set `CACHE_REFRESH_INTERVAL = 0` in the patched config.
3. **Lag-`D` sample-size shortfall.** For `D = 8` with default `T_chain`, the oracle check may fail. Re-run Stage-0 gates per `D` before launching training in Stage-2; bump `T_chain`/`n_seq` if needed.
4. **Squared metric vs `|cos|` AUC.** `v5_hmm_sae_baseline/metrics.feature_recovery_score` reports an AUC of `|cos sim|` thresholds, not the squared metric the proposal uses. Report **both**: the squared metric is the headline; AUC is a continuity metric for cross-experiment comparison.
5. **`orthogonalize` is approximate.** Do not use `src/shared/orthogonalize.py` for `F` — use `torch.linalg.qr(randn)` for exact orthonormality.
6. **Untracked proposal file.** The `.tex` is untracked. The new `README.md` will inline the math and cite line numbers so the implementation isn't pinned to a fragile path.

## Out of scope (Stage 3 — deferred)

- σ sweep (`σ ∈ {0, 0.1, 0.25, 0.5}`).
- Eigengap sweep (`(ρ_min, ρ_max) ∈ {(0.1,0.9), (0.4,0.8), (0.6,0.75)}`).
- `N = 256` robustness check.
- Overcomplete dictionaries (`H = 2N`).
- ReLU signed-pair variant (proposal lines 978–1001).
- Sparse-source ablation (replacing Gaussian sources).

These can be appended as Stage 3 once the headline figure is reproduced.
