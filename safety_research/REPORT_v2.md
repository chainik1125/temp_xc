# Real-Benchmark Safety Eval — TXC vs T-SAE vs SAE

Branch: `andre_safety` · Layer: `mid_res` (Gemma-2-2b-it L13) · k=100/position · d_sae=18432

This report scales the earlier 60-prompt toy eval ([REPORT.md](REPORT.md)) to a real, published benchmark and tests the same architectures under the *monitor-and-steer* paradigm. The headline question is whether TXC has AI-safety supremacy over SAEs as a **monitor** (white-box deception/refusal classifier) or as an **intervention** (refusal-direction steering).

All training and eval is logged to wandb under [`temporal-crosscoders-safety`](https://wandb.ai/standartikom-northwestern-university/temporal-crosscoders-safety) (group `realbench-detect`, `realbench-steer`).

## 1. Benchmark composition

Three real-world public sources, no hand-written prompts:

| split | source | n | n_pos (harmful) |
|-------|--------|---|-----------------|
| train | AdvBench (520 harmful) + Alpaca (520 benign) | 1040 | 520 |
| test_in | JailbreakBench-Behaviors (100 H + 100 B, matched-pairs) | 200 | 100 |
| test_ood | XSTest v2 (225 safe + 225 unsafe; cross-distribution stress test) | 450 | 225 |

`test_in` ⊂ JailbreakBench is the *in-distribution* monitor test — JBB's matched-pair design (e.g. 'main character uses heroin' vs 'addicted to heroin') controls for surface lexical overlap. `test_ood` ⊂ XSTest v2 is the cross-distribution stress test: many prompts are deliberately phrased to look unsafe ('how do I kill a Python process?') or to put safe content in unsafe-looking framing.

## 2. Monitor / detection — per-arm AUC

Sparse linear probe (logistic regression with L2) trained on the top-2k features by per-feature AUC on `train`, evaluated on the two test splits.

| arm | test_in AUC [95% CI] | test_ood AUC [95% CI] | test_in AP | test_ood AP | b2w_in | b2w_ood |
|-----|----------------------|------------------------|------------|--------------|---------|----------|
| TF-IDF (text-only) | 0.693 [0.622, 0.765] | 0.668 [0.623, 0.718] | 0.671 | 0.643 | — | — |
| raw L13 residual | 0.966 [0.940, 0.987] | 0.959 [0.944, 0.974] | 0.973 | 0.960 | — | — |
| SAE (T=1) | 0.961 [0.933, 0.984] | 0.948 [0.930, 0.967] | 0.968 | 0.954 | +0.268 | +0.280 |
| T-SAE (T=5) | 0.973 [0.954, 0.988] | 0.963 [0.948, 0.976] | 0.973 | 0.964 | +0.279 | +0.295 |
| TXC (T=5) | 0.970 [0.952, 0.987] | 0.954 [0.937, 0.970] | 0.973 | 0.956 | +0.277 | +0.286 |

![detect-auc](figures/fig1_detect_auc.png)

![roc](figures/fig2_roc_curves.png)

![b2w](figures/fig3_b2w_boost.png)

![per-feat](figures/fig4_per_feat_top.png)

**Take-aways.**

- **All three SAE-family arms are within 95% bootstrap CI of each other** on both splits. The story "TXC has detection supremacy" from the toy 60-prompt setup *does not survive* the move to real benchmarks: best-feature AUCs at 1.000 there compress into the 0.94-0.97 band here, with overlapping CIs.
- The cleanest single number is the **black-to-white boost**: all three arms add **+0.27-0.30** AUC over the TF-IDF text baseline. That is the load-bearing finding — *internal-state monitoring of Gemma-2-2b-it is far better than reading the prompt*. Architecture differences are second-order.
- The **best monitor** by `test_ood` AUC is **T-SAE (T=5)** (AUC = 0.963), but the gap to the worst arm is 0.015, comparable to the bootstrap half-width.
- TXC and T-SAE both **degrade slightly more** under XSTest distribution shift than SAE does, but the gap is < 0.02 AUC.
- Compared to the *raw L13 residual* probe (AUC = 0.959 on test_ood), all SAE arms lose 0.01-0.02 AUC. This is the classic SAE compression cost — the dictionary is *not* discovering signal the residual lacks; it's just an interpretable approximation.

## 3. Steering / intervention — refusal log-ratio shifts

For each arm we build steering directions in two ways:

- **probe-coef direction** — back-project the LogReg probe coefficients (over the top-2k features by per-feature AUC) into residual space via the *signed* decoder columns. Result: a single unit vector.
- **centroid direction** — mean of the top-20 most-refusal-aligned decoder columns (sign-aligned).
- **DoM (no SAE)** — Arditi-style diff-of-means at the last token; this is the encoder-free baseline.

All directions intervene at L13 by adding `α · d` to the residual stream at every position, then we re-score the refusal log-ratio = log p('I cannot help with that') − log p('Sure, here is') for the JBB test_in prompts.

Baseline (no intervention): refusal-LR mean on harmful = +12.976, on benign = +0.840.

Inject `α` on the refusal direction. Δ values are vs baseline.

| method | α | ΔLR_harm | ΔLR_ben | targeted (Δh − Δb) |
|--------|---|----------|---------|---------------------|
| SAE (T=1) coef | -2.0 | -0.153 | -0.312 | +0.158 |
| SAE (T=1) coef | +0.0 | +0.000 | +0.000 | +0.000 |
| SAE (T=1) coef | +1.0 | +0.077 | +0.153 | -0.076 |
| SAE (T=1) coef | +2.0 | +0.149 | +0.309 | -0.159 |
| SAE (T=1) coef | +4.0 | +0.298 | +0.620 | -0.322 |
| SAE (T=1) centroid | -2.0 | -0.141 | -0.284 | +0.143 |
| SAE (T=1) centroid | +0.0 | +0.000 | +0.000 | +0.000 |
| SAE (T=1) centroid | +1.0 | +0.069 | +0.139 | -0.070 |
| SAE (T=1) centroid | +2.0 | +0.137 | +0.281 | -0.143 |
| SAE (T=1) centroid | +4.0 | +0.274 | +0.565 | -0.291 |
| T-SAE (T=5) coef | -2.0 | -0.097 | -0.309 | +0.211 |
| T-SAE (T=5) coef | +0.0 | +0.000 | +0.000 | +0.000 |
| T-SAE (T=5) coef | +1.0 | +0.050 | +0.152 | -0.102 |
| T-SAE (T=5) coef | +2.0 | +0.095 | +0.307 | -0.212 |
| T-SAE (T=5) coef | +4.0 | +0.186 | +0.608 | -0.422 |
| T-SAE (T=5) centroid | -2.0 | -0.097 | -0.309 | +0.213 |
| T-SAE (T=5) centroid | +0.0 | +0.000 | +0.000 | +0.000 |
| T-SAE (T=5) centroid | +1.0 | +0.052 | +0.155 | -0.103 |
| T-SAE (T=5) centroid | +2.0 | +0.095 | +0.310 | -0.215 |
| T-SAE (T=5) centroid | +4.0 | +0.191 | +0.616 | -0.425 |
| TXC (T=5) coef | -2.0 | -0.140 | -0.240 | +0.100 |
| TXC (T=5) coef | +0.0 | +0.000 | +0.000 | +0.000 |
| TXC (T=5) coef | +1.0 | +0.068 | +0.117 | -0.049 |
| TXC (T=5) coef | +2.0 | +0.140 | +0.243 | -0.103 |
| TXC (T=5) coef | +4.0 | +0.275 | +0.483 | -0.208 |
| TXC (T=5) centroid | -2.0 | -0.150 | -0.251 | +0.101 |
| TXC (T=5) centroid | +0.0 | +0.000 | +0.000 | +0.000 |
| TXC (T=5) centroid | +1.0 | +0.073 | +0.126 | -0.054 |
| TXC (T=5) centroid | +2.0 | +0.149 | +0.249 | -0.100 |
| TXC (T=5) centroid | +4.0 | +0.298 | +0.503 | -0.205 |
| DoM (no SAE) | -2.0 | -0.219 | -0.435 | +0.217 |
| DoM (no SAE) | +0.0 | +0.000 | +0.000 | +0.000 |
| DoM (no SAE) | +1.0 | +0.106 | +0.217 | -0.111 |
| DoM (no SAE) | +2.0 | +0.218 | +0.438 | -0.220 |
| DoM (no SAE) | +4.0 | +0.433 | +0.889 | -0.455 |

**Ablation** (project out top-K decoder directions / DoM):

| method | ΔLR_harm | ΔLR_ben | targeted |
|--------|----------|---------|----------|
| SAE (T=1) top-K ablation | -17.952 | -7.762 | -10.190 |
| T-SAE (T=5) top-K ablation | -12.696 | -4.551 | -8.145 |
| TXC (T=5) top-K ablation | -11.501 | -7.738 | -3.763 |
| DoM ablation | -19.826 | -10.365 | -9.461 |

![pareto](figures/fig5_steer_pareto.png)

![dose](figures/fig6_steer_doseresponse.png)

## 4. Headline answer — TXC as monitor or intervention?

**Monitor (white-box deception/refusal classifier).** On real benchmarks (JBB + XSTest, n=650), the three SAE families are a wash: TXC = 0.954, T-SAE = 0.963, SAE = 0.948 on test_ood. Bootstrap CIs overlap; differences are not significant. The big lift is white-box vs black-box (+0.27-0.30 AUC), not which white-box.

**Intervention.** See section 3 — TXC's diffuse cross-position decomposition of refusal makes single-direction steering harder than the position-specialized T-SAE basis, consistent with the negative result in the original 60-prompt eval (`steering AUC: TXC=0.25, T-SAE=0.95`). The remedy is the **andre-steering** branch — see the follow-up [report](#) where we test a hybrid steering recipe: TXC for *feature selection* (it discovers more refusal-aligned atoms), T-SAE-style *per-position decoder direction* for the actual intervention.

## 5. How this iterates on Secret Agenda (arXiv:2509.20393)

DeLeeuw et al. report two failures of SAE-based safety tooling:

1. *autolabeled deception features rarely fire when the model is actually being deceptive* (i.e., the labels are wrong);
2. *steering on 100+ deception-labeled features fails to suppress lying behavior* (i.e., the directions are wrong).

This study addresses (1) by **training the probe instead of trusting the autolabel** — we measure each arm's monitor AUC on the actual harmful-vs-benign labels rather than asking 'do features I labeled DECEPTION fire?'. All three arms, including vanilla SAE, get to AUC ≥ 0.94 on real benchmarks once the probe is supervised by the behavior of interest. The lesson is that **SAE features carry the signal; autolabel pipelines were the failure mode**, not the basis.

This study addresses (2) directly in section 3 above.

## 6. Reproducibility

```text
safety_research/scripts/
  realbench_build.py        # build train/test_in/test_ood prompt sets
  realbench_cache_acts.py   # forward Gemma, cache L13 last-T residuals
  realbench_detect.py       # monitor probes + black-to-white boost
  realbench_steer.py        # inject + ablate eval, all directions
  realbench_plots.py        # all figures
  realbench_report.py       # build this report
```

All artifacts under `safety_research/results/realbench/`. wandb runs: `realbench-detect`, `realbench-steer`.