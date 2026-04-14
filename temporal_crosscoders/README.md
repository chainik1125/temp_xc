# Temporal Crosscoder Sweep

Systematic comparison of **TopK SAE** vs **Temporal Crosscoder (TXCDR)** on synthetic data with controlled temporal structure, sweeping over sparsity `k` and window length `T`.

## Conjecture Under Test

> **optimal_k(TXCDR) >> optimal_k(SAE)**, all other things being equal.

The temporal crosscoder sees a window of T positions simultaneously, pooling evidence across time. This should allow it to usefully activate *more* latents (higher k) without degradation, because correlated features across the window provide mutual confirmation. A standard SAE, seeing one token at a time, should plateau or degrade at lower k since it lacks this cross-timestep disambiguation signal.

### How Results Prove or Disprove This

**SUPPORTED if:**
- For the **markov** dataset (strong temporal correlation), the k that maximizes final AUC is consistently larger for TXCDR than for SAE across multiple T values.
- The `viz.py` optimal-k plot shows TXCDR points above the SAE baseline line.
- The advantage heatmap shows increasingly positive cells as k grows (TXCDR benefits from higher k more than SAE does).

**REFUTED if:**
- Optimal k is similar for both models, or SAE actually prefers higher k.
- The advantage heatmap shows no systematic k-dependent pattern.

**NUANCED outcomes:**
- If the conjecture holds for markov but **not** for iid, that confirms the mechanism is temporal correlation, not

 just architectural capacity.
- If TXCDR optimal k grows with T, that's even stronger evidence: longer windows provide more temporal context, enabling more latents to be productively activated.

## Project Structure

```
config.py          All hyperparameters, sweep grid, wandb settings
models.py          TopKSAE + TemporalCrosscoder architectures
data.py            Toy model, IID/Markov data generators, iterators
metrics.py         Feature recovery AUC, cosine similarity
train.py           Training loops with wandb + local JSON logging
sweep.py           Main sweep runner (CLI, supports single-run mode)
viz.py             Post-hoc visualization and statistical analysis
run_sweep.sh       tmux launcher (sequential or parallel by dataset)
requirements.txt   Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt
export WANDB_MODE=disabled  # or set WANDB_API_KEY

# Test run (1000 steps)
python sweep.py --steps 1000

# Full sweep (1M steps, ~hours on GPU)
python sweep.py

# Parallel via tmux (one pane per dataset)
chmod +x run_sweep.sh
./run_sweep.sh --parallel

# Generate plots after sweep completes
python viz.py
```

## Single-Run Mode (for manual tmux parallelism)

```bash
# Run specific combos in different tmux panes / GPUs:
CUDA_VISIBLE_DEVICES=0 python sweep.py --dataset iid --model sae --steps 1000000 &
CUDA_VISIBLE_DEVICES=0 python sweep.py --dataset iid --model txcdr --steps 1000000 &
CUDA_VISIBLE_DEVICES=1 python sweep.py --dataset markov --steps 1000000 &
```

## Sweep Grid

| Parameter | Values |
|-----------|--------|
| Dataset | `iid`, `markov` |
| k (active latents/position) | 1, 2, 4, 8 |
| T (window length) | 1, 2, 4, 8, 10 |
| Models | SAE (T-independent), TXCDR |
| Steps | 1,000,000 |
| Skip condition | k × T ≥ NUM_FEATS (100) |

SAE is run once per (dataset, k) since it doesn't see windows.

## Outputs

- `logs/*.json` — per-run training histories (step, loss, AUC, recovery)
- `logs/sweep_summary.json` — final metrics for all runs
- `viz_outputs/heatmap_advantage_*.png` — **the key plot**: TXCDR(k,T) − best SAE
- `viz_outputs/optimal_k_analysis.png` — direct conjecture test
- `viz_outputs/auc_vs_k_*.png` — AUC scaling with k
- `viz_outputs/convergence_*.png` — training curves
- `viz_outputs/summary_table.txt` — numerical summary with verdict
