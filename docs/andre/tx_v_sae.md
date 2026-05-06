# Temporal Crosscoder Sweep — Results Summary

Refer to ```temporal_crosscoders``` root folder for the code. 

To replicate:
1. go to ```temporal_crosscoders```
2. change ```config.py``` 
3. run ```python sweep.py``` -- records to wandb 
4. run ```python viz.py``` -- visualize 

**Setup**: TopK SAE vs shared-latent Temporal Crosscoder (ckkissane-style), 80k steps, d=200, h=100, FEAT_PROB=0.05, k∈{2,5,10,25}, T∈{2,5}, iid + markov data, `effective_k = k` for both models.

## Key Findings

1. **SAE dominates the crosscoder across the entire grid** — the shared-latent bottleneck forces k features to explain T×5 activations simultaneously, systematically underperforming a per-token SAE with the same k. ![advantage iid][hm-iid] ![advantage markov][hm-mkv]

2. **IID: SAE achieves near-perfect recovery (AUC=0.975) at k=5**, matching the ~5 true active features, while the best TXCDR reaches only 0.75 at T=2 k=10. ![auc vs k iid][auc-iid]

3. **Markov: the gap narrows at high k** — TXCDR(k=25, T=5) finally matches SAE(k=25) at AUC≈0.87, the only cell in the entire sweep where the crosscoder is not behind. ![auc vs k markov][auc-mkv]

4. **Higher T hurts at low k and helps at high k** — with few latents the compression tax of T positions is fatal, but at k=25 the crosscoder has enough budget that temporal context adds marginal value on markov data. ![raw auc markov][raw-mkv]

5. **The optimal-k conjecture is technically "supported" on IID (ratio=2×) but vacuously** — the TXCDR prefers higher k because it's *failing* to recover features (its best AUC is 0.75 vs SAE's 0.97), not because temporal context enables productive use of more latents. ![optimal k][opt-k]

6. **Convergence curves confirm the crosscoder is not undertrained** — both models plateau by 40–60k steps; the TXCDR deficit is architectural ![convergence iid][conv-iid] ![convergence markov][conv-mkv]

### Hyperparams

```
# ─── Toy model geometry ─────────────────────────────────────────────────────────
NUM_FEATS = 50      # number of ground-truth features (also d_sae)
HIDDEN_DIM = 100      # representation dimension d
FEAT_PROB = 0.05     # Bernoulli activation probability
FEAT_MEAN = 1.0       # magnitude mean
FEAT_STD = 0.15       # magnitude std

# ─── Data generation schemes ────────────────────────────────────────────────────
# Only iid and markov (Scheme C) per user request
DATASETS = ["markov", "iid"]
#DATASETS = ['markov']  # for quick testing; comment out to restore full sweep

# Markov chain parameters (Scheme C)
MARKOV_ALPHA = 0.95   # stay-on probability
MARKOV_BETA = 0.03    # turn-on probability

# ─── Training ───────────────────────────────────────────────────────────────────
TRAIN_STEPS = 80_000 # 1m steps for convergence
LOG_INTERVAL = 1_500      # log metrics every N steps
EVAL_BATCH = 128         # larger batch for stable eval
BATCH_SIZE = 1          # training batch size (both SAE and TXCDR)
LEARNING_RATE = 3e-4
ADAM_BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
SEED = 42

# ─── Sweep grid ─────────────────────────────────────────────────────────────────
SWEEP_K = [2, 5, 10, 25][::-1]  # base active latents per token-position
SWEEP_T = [2, 5][::-1]       # window lengths

def should_skip(k: int, T: int) -> bool:
    """Skip configurations where k*T >= NUM_FEATS (underdetermined)."""
    return txcdr_effective_k(k, T) > NUM_FEATS

# ─── Model sizing ───────────────────────────────────────────────────────────────
D_SAE = NUM_FEATS     # SAE / crosscoder latent dimension = number of true features

def sae_effective_k(k: int) -> int:
    """Active latents for the SAE.  k = per-token budget."""
    return k

def txcdr_effective_k(k: int, T: int) -> int:
    """Active latents for the crosscoder.  k per position × T positions."""
    return k
```

## Next Steps 

While I didn't test it, the issue might be the dead neurons. We likely will need to update the loss function such that each token contributes equally to the distribution of TopK activations over the latent vector. 

[hm-iid]: sweep_tx_v_sae/heatmap_advantage_iid.png
[hm-mkv]: sweep_tx_v_sae/heatmap_advantage_markov.png
[auc-iid]: sweep_tx_v_sae/auc_vs_k_iid.png
[auc-mkv]: sweep_tx_v_sae/auc_vs_k_markov.png
[raw-mkv]: sweep_tx_v_sae/heatmap_raw_auc_markov.png
[opt-k]: sweep_tx_v_sae/optimal_k_analysis.png
[conv-iid]: sweep_tx_v_sae/convergence_iid.png
[conv-mkv]: sweep_tx_v_sae/convergence_markov.png