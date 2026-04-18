---
author: Claude
date: 2026-03-30
tags:
  - results
---

## Experiment 2: ReLU + L1 Pareto frontier

**Models.** Same architectures as Experiment 1, but using **ReLU + L1** sparsity instead of TopK. Both the SAE and TFA's novel component use a ReLU encoder with an L1 penalty $\lambda \|z\|_1$ on the latent codes; L0 emerges from training rather than being fixed. We call the SAE baseline in this experiment the **ReLU SAE** to distinguish it from the Shared SAE in Experiment 1. Swept $\lambda$ over 15 log-spaced values (ReLU SAE: $5 \times 10^{-3}$ to $20$; TFA: $0.15$ to $60$; TXCDR: $3.2 \times 10^{-2}$ to $32$). Each $\lambda$ produces one (L0, NMSE) point; the Pareto frontier is the lower envelope.

![Pareto with AUC](../../../../src/v2_temporal_schemeC/results/auc_and_crosscoder/exp2_pareto_auc.png)

Left: L0 vs NMSE Pareto. Centre: L0 vs AUC Pareto. Right: NMSE vs AUC scatter. **Solid lines with large markers** are Pareto frontiers; **faded dots** are dominated runs.

**Findings.**

1. **TFA's novel-L0 frontier lies below the ReLU SAE's in the binding regime** (1.3--2.2x advantage at L0 = 7--12), consistent with Experiment 1. On total L0, TFA's frontier is strictly *above* the ReLU SAE's — TFA uses 20--32 total active codes for NMSE that the ReLU SAE achieves with fewer purely sparse codes. This confirms that TFA's NMSE advantage at matched novel L0 comes at the cost of a dense channel that inflates total representation complexity.

2. **TXCDR and TXCDRv2 achieve the best feature recovery** (decoder-averaged AUC). TXCDR $T$=2 peaks at AUC = 0.99 and TXCDRv2 $T$=2 reaches AUC = 0.99, both at low L1. (In L1 mode with $k = $ None, TXCDR and TXCDRv2 are identical — the $k \times T$ distinction only applies under TopK.) Their NMSE floor (~0.003) is an order of magnitude worse than the ReLU SAE's best ($7 \times 10^{-6}$). The crosscoders learn interpretable feature directions but the shared-latent bottleneck limits reconstruction quality.

3. **Stacked SAE has low decoder-averaged AUC** despite good per-position reconstruction. At matched L0, the Stacked SAE's decoder-averaged AUC (0.4--0.7) is substantially lower than TXCDR's (0.8--0.94), confirming that independent per-position decoders do not learn aligned feature representations.

4. **All windowed models collapse at moderate L1** (NMSE $\approx 0.5$), indicating narrow usable L1 ranges compared to the SAE and TFA.

5. **TFA-pos achieves the best NMSE--AUC tradeoff among per-token models.** At its best operating point (L1 $= 0.16$, novel L0 $= 14.8$), TFA-pos achieves NMSE $= 1.2 \times 10^{-4}$ with AUC $= 0.95$ --- lower NMSE than TFA ($4.3 \times 10^{-4}$) at comparable AUC. TFA-pos's AUC peaks at 0.95 (vs TFA's 0.81 and SAE's 0.91).
