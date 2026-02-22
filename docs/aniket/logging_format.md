---
author: aniket
date: 2026-02-21
tags:
  - reference
---

## Logging Format Convention

formatting i use for experiment logging

All log messages use a **bracketed prefix tag** to indicate the message type, followed by the message content. This provides consistent, scannable output.

## Tag Types

| Tag | Purpose | Example |
|-----|---------|---------|
| `[info]` | General status updates, progress info | `[info] initializing toy_model with 50 features in R^100...` |
| `[data]` | Dataset creation and statistics | `[data] generated 25 token_embeddings \| avg_nnz=11.2 \| rho=0.60` |
| `[train]` | Training progress within a run | `[train] step 5000/15000 \| recon_loss=3.42e-02 \| l0=8.3` |
| `[sweep XX/YY]` | Per-config progress during hyperparameter sweeps | `[sweep 03/18] arch=crosscoder \| rho=0.40 \| top_k=10` |
| `[eval]` | Evaluation metrics after training | `[eval] mean_max_cos_sim=0.87 \| dead_latents=3/100 \| fvu=0.12` |
| `[summary]` | Aggregate results after a sweep | `[summary] rho=0.60 \| crosscoder_pareto_area=0.34 \| naive_sae_pareto_area=0.21` |
| `[done]` | Task completion, file saves | `[done] saved pareto_frontier_rho_0.60.png -> ./plots/` |
| `[error]` | Error conditions | `[error] correlation_matrix not positive semi-definite` |
| `[result]` | Final computed values | `[result] crosscoder_advantage=0.13 at rho=0.80` |

## Format Rules

1. **Tag at start**: Always begin with `[tag]` followed by a space
2. **Lowercase tags**: Use lowercase for consistency (`[info]` not `[INFO]`)
3. **Pipe separators**: Use `|` to separate multiple metrics on one line
4. **Scientific notation**: Use `.2e` or `.3e` format for floating-point values
5. **Progress counters**: Format as `XX/YY` with zero-padded indices (`01/18`)
6. **Timing**: Include `time=X.Xs` or `time=X.Xmin` for duration tracking
7. **snake_case everything**: All variable names, file names, directory names, metric names, and config keys must be snake_case (`recon_loss`, `cos_sim_matrix`, `pareto_frontier`, never `reconLoss` or `Pareto-Frontier`)

## Naming Conventions (snake_case)
```text
# files and directories
data/synthetic_data.py
models/naive_sae.py
models/crosscoder.py
plots/pareto_frontier_rho_0.60.png
results/sweep_results.json

# variables and config keys
num_features, hidden_dim, firing_prob
cross_position_rho, l1_coefficient, top_k
recon_loss, l0_sparsity, cos_sim_with_true
w_enc, w_dec, b_enc, b_dec

# metrics
mean_max_cos_sim     # best cosine sim per true feature, averaged
dead_latent_frac     # fraction of latents that never fire
fvu                  # fraction of variance unexplained
pareto_area          # area under pareto frontier curve
```

## Example Output Block
```text
[info] initializing toy_model | num_features=50 | hidden_dim=100 | target_cos_sim=0.00
[data] generated token_embeddings | n_tokens=25 | avg_nnz=11.2 | firing_mag=N(1.0, 0.15)
[data] cross_position_rho=0.60 | correlation_matrix shape=(100, 100) | min_eigenval=1.23e-03

[info] training naive_sae | d_sae=100 | top_k=10 | training_steps=15000
[train] step 05000/15000 | recon_loss=8.41e-02 | l0=10.0 | time=12.3s
[train] step 10000/15000 | recon_loss=3.22e-02 | l0=10.0 | time=24.8s
[train] step 15000/15000 | recon_loss=1.87e-02 | l0=10.0 | time=37.1s
[eval] naive_sae | mean_max_cos_sim=0.72 | dead_latents=5/100 | fvu=1.87e-02

[info] training crosscoder | d_sae=100 | top_k=10 | training_steps=15000
[train] step 05000/15000 | recon_loss=6.13e-02 | l0=10.0 | time=14.1s
[train] step 10000/15000 | recon_loss=1.98e-02 | l0=10.0 | time=28.5s
[train] step 15000/15000 | recon_loss=9.43e-03 | l0=10.0 | time=42.7s
[eval] crosscoder | mean_max_cos_sim=0.91 | dead_latents=1/100 | fvu=9.43e-03

[summary] rho=0.60 | crosscoder_cos_sim=0.91 | naive_sae_cos_sim=0.72 | advantage=0.19
[done] saved feature_recovery_heatmap_rho_0.60.png -> ./plots/
[done] saved sweep_results.json -> ./results/
```
