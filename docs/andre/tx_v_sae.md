# Temporal Crosscoder Sweep — Results Summary

Refer to ```temporal_crosscoders``` root folder for the code. 

**Setup**: TopK SAE vs shared-latent Temporal Crosscoder (ckkissane-style), 80k steps, d=200, h=100, FEAT_PROB=0.05, k∈{2,5,10,25}, T∈{2,5}, iid + markov data, `effective_k = k` for both models.

## Key Findings

1. **SAE dominates the crosscoder across the entire grid** — the shared-latent bottleneck forces k features to explain T×5 activations simultaneously, systematically underperforming a per-token SAE with the same k. ![advantage iid][hm-iid] ![advantage markov][hm-mkv]

2. **IID: SAE achieves near-perfect recovery (AUC=0.975) at k=5**, matching the ~5 true active features, while the best TXCDR reaches only 0.75 at T=2 k=10. ![auc vs k iid][auc-iid]

3. **Markov: the gap narrows at high k** — TXCDR(k=25, T=5) finally matches SAE(k=25) at AUC≈0.87, the only cell in the entire sweep where the crosscoder is not behind. ![auc vs k markov][auc-mkv]

4. **Higher T hurts at low k and helps at high k** — with few latents the compression tax of T positions is fatal, but at k=25 the crosscoder has enough budget that temporal context adds marginal value on markov data. ![raw auc markov][raw-mkv]

5. **The optimal-k conjecture is technically "supported" on IID (ratio=2×) but vacuously** — the TXCDR prefers higher k because it's *failing* to recover features (its best AUC is 0.75 vs SAE's 0.97), not because temporal context enables productive use of more latents. ![optimal k][opt-k]

6. **Convergence curves confirm the crosscoder is not undertrained** — both models plateau by 40–60k steps; the TXCDR deficit is architectural ![convergence iid][conv-iid] ![convergence markov][conv-mkv]

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