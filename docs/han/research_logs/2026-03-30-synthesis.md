## Synthesis

### Summary of findings

1. **TFA achieves 3--7x lower NMSE than a standard SAE at matched novel sparsity in the binding regime**, consistent across both TopK (Experiment 1) and ReLU+L1 (Experiment 2) sparsity mechanisms. The advantage peaks at 6.7x near $k = \mathbb{E}[L_0] = 10$ and disappears once the SAE has enough capacity ($k > 12$).

2. **Most of this advantage is architectural capacity, not temporal structure** (Experiment 1, finding 5). TFA-shuffled — trained on data with all temporal correlations destroyed — captures most of TFA's advantage over the Shared SAE (2.4--5.7x improvement vs the Shared SAE's baseline). TFA improves only 1.2--1.3x beyond TFA-shuffled. Our linear decomposition attributes roughly 88--97% of the total NMSE gap to architecture and 3--12% to temporal structure, though these estimates are from a single seed and the decomposition does not account for possible architecture-temporal interactions.

3. **Temporal structure provides a small benefit without positional encoding, and a substantial benefit with it** (Experiment 1, findings 5--8). Without positional encoding, TFA's temporal fraction is 3--12%. With positional encoding (TFA-pos), the fraction rises to 17--26%. At $k = 8$, TFA-pos achieves 2$\times$ lower NMSE than TFA and 19% higher AUC. This shows that the attention mechanism *can* exploit temporal correlations when given positional information, but cannot do so from content alone in our toy model.

4. **TFA's predictable component does not distinguish temporal transitions** (Experiments 3, 3b): continuation and onset projections are virtually identical, and prediction strength shows no monotonic relationship with temporal persistence $\rho$. The predictable component does partially distinguish ON from OFF features (with high false-positive rates), but this discrimination does not depend on temporal history. This result is robust to stronger tests: conditioning on 5-position run-length history (long continuation $1\!1\!1\!1\!1$ vs sudden onset $0\!0\!0\!0\!1$) gives ratios of 0.98--1.02, and quadrupling the sequence length to $T = 256$ moves the ratio even closer to 1.0 (Experiment 3b). The likely explanation is that content-based matching dominates: with $\pi = 0.5$ and 20 features, any two tokens share ~5 features by chance, so the attention finds broadly similar context tokens regardless of whether the specific feature under test was active in context (see "Why content-based matching dominates" below).

5. **NMSE and feature recovery (AUC) are dissociated** (Experiments 1, 2). Good reconstruction does not imply good feature recovery. The SAE at $k = 20$ achieves near-perfect NMSE ($2 \times 10^{-6}$) but AUC = 0.59 --- it reconstructs via superposition without recovering the true feature directions. Conversely, the temporal crosscoder (TXCDR, $T = 2$) at $k = 15$ achieves AUC = 0.94 (near-perfect feature recovery) but NMSE = 0.042 --- its shared-latent bottleneck prevents good reconstruction while its per-position decoder columns align well with the true features. TFA sits between the two: AUC peaks at 0.81 ($k = 10$) then declines, mirroring the SAE's pattern. This dissociation is consistent with the "Sparse but Wrong" thesis (Chanin et al., 2025): reconstruction-optimized models may learn superposed representations that do not correspond to the true features.

### Why the attention mechanism helps without temporal structure

We hypothesise that TFA's attention mechanism provides reconstruction capacity through two pathways that do not require temporal order:

**Content-based retrieval.** The query (derived from the current token's encoding) and keys (from context tokens) interact via dot-product attention. Even when context tokens are randomly ordered, tokens that share active features with the current token should produce higher attention weights. The value projection then retrieves a weighted reconstruction biased toward the current token's content. With $\pi = 0.5$ and 20 features, any two tokens share ~5 features on average ($n \pi^2 = 5$), providing some signal for content-based matching even in shuffled sequences.

**Projection scaling.** After attention produces a predicted direction $D z_{\text{pred}}$, TFA scales it by $\text{proj\_scale} = \langle D z_{\text{pred}}, x \rangle / \|D z_{\text{pred}}\|^2$, which is the scalar projection of the current input onto the predicted direction. This adapts the predictable component's magnitude to the current token regardless of how the direction was obtained. Even a constant attention output (which would occur if context were completely uninformative) gets scaled to match the current input, providing a rank-1 reconstruction "for free." Note that proj\_scale is not the reason the Experiment 3b long/sudden ratio is $\approx 1.0$ --- even though $x_t$ is the same in both cases (the toy model is a memoryless embedding), the attention *direction* differs, and proj\_scale projects $x_t$ onto that direction, so different directions should yield different projections. The real reason is that content-based matching produces broadly similar directions regardless of whether the specific feature was in context (see below).

Together, these mechanisms give TFA an additional dense reconstruction channel with ~20 active codes that does not count toward the novel L0 budget. Note that this is not equivalent to an optimal rank-20 projection (which would give perfect reconstruction on our 20-feature data); TFA-shuffled at $k = 3$ still has NMSE $= 0.09$, so the predictable component captures only a fraction of the information that an oracle projection would. Nevertheless, this channel is sufficient to dramatically outperform the Shared SAE. We have not directly verified these mechanisms (e.g., by inspecting attention weights on shuffled data); they remain a plausible explanation for TFA-shuffled's strong performance.

### Why extra parameters alone do not help

The Wide Shared SAE (dictionary width 100, 8,140 params — roughly matching TFA's 8,200) performs comparably to or *worse* than the standard Shared SAE (width 40, 3,280 params) at $k \geq 8$. With 100 dictionary atoms in a 40-dimensional space, the extra atoms create redundant directions that compete during training, and TopK selection from a larger pool does not help when the underlying data has only 20 ground-truth features.

This rules out the simplest capacity hypothesis — that TFA wins merely because it has more parameters. TFA's advantage comes from the specific computational structure of the attention mechanism (query-key interaction, value aggregation, projection scaling), not from parameter count per se. We note, however, that the wide SAE tests only one form of extra capacity (wider dictionary); other ways of adding parameters to the SAE (e.g., deeper encoders, multiple encoder heads) were not tested.

### Why content-based matching dominates temporal signal

Experiment 3c directly confirms that content-based matching, not temporal prediction, drives the attention direction. The shuffled model achieves 74% variance explained (vs 94% temporal, 2.5% random), meaning that context tokens matched by content alone --- without any temporal order --- provide most of the directional signal. The temporal model's additional ~20% comes from training distribution match, not from per-feature temporal prediction: the per-feature alignment $|\cos(Dz, \mathbf{f}_i)|$ is identical for long continuations and sudden onsets in *both* temporal and shuffled models (Experiment 3c finding 1).

The mechanism is clear: with $\pi = 0.5$ and 20 features, $x_t$ has ~10 active features, and any context token shares ~5 of them ($n\pi^2 = 5$). The attention's query-key matching is dominated by these ~5 shared features, not by the single feature $i$ under test. Whether feature $i$ specifically appeared in context barely perturbs the attention weights, because it's one signal among many. The resulting direction $D z_{\text{pred}}$ is broadly distributed across features (per-feature alignment only ~0.23 even for the best model), confirming it's a content-matching direction rather than a feature-specific prediction.

This interpretation predicts that the temporal signal should become visible in regimes where content-based matching is weaker --- e.g., very low $\pi$ (sparse features, less incidental overlap). It also predicts that the 3--12% temporal fraction from the shuffle diagnostic is largely a training distribution effect rather than genuine temporal exploitation --- consistent with Experiment 3c's finding that temporal training improves direction quality uniformly across all run-length categories.

### Positional encoding unlocks temporal exploitation

The TFA-pos results (Experiment 1, findings 5--8) demonstrate that the lack of positional information in our toy model is a genuine limitation, not merely an academic concern. When TFA's attention receives sinusoidal positional encoding in its Q/K inputs, the temporal fraction rises from 3--12% (TFA) to 17--26% (TFA-pos). At $k = 8$, TFA-pos achieves 2$\times$ lower NMSE than TFA while also achieving substantially higher AUC (0.91 vs 0.72).

The mechanism is clear: without positional info, TFA's attention treats all context tokens identically --- it can only match by content. With positional encoding, the attention can learn "attend more to the immediately preceding tokens," which in a Markov chain are more likely to share the current token's active features (due to temporal correlation $\rho$). This is precisely the temporal exploitation pathway the TFA paper envisions, but it requires positional information that our toy model's memoryless representations do not provide by default.

Importantly, TFA-pos-shuffled still performs comparably to TFA-shuffled (both lack temporal correlations in training), confirming that the NMSE improvement comes from temporal structure, not from the positional encoding itself. The encoding provides the *mechanism* for temporal exploitation; the temporal data provides the *signal*.

This finding has implications for interpreting TFA on real LM activations: at middle layers, the residual stream already encodes positional information from the transformer's own positional encoding and prior attention layers, so TFA's attention has access to this signal by default. Our TFA-pos results suggest that this positional information is an important ingredient in TFA's ability to capture temporal structure on real data.

### Relationship to the TFA paper's claims

Our results do not directly contradict the TFA paper because the paper does not claim NMSE superiority --- it claims interpretive value of the pred/novel decomposition on real LLM activations. Our findings show that on synthetic data with high feature density ($\pi = 0.5$), content-based matching dominates temporal signal in TFA's predictable component. The TFA paper operates on real LM activations where features are much sparser and the residual stream already encodes temporal context, both of which would strengthen the temporal channel relative to what we observe.

The TFA paper's qualitative successes on stories, garden-path sentences, and in-context learning may reflect both of these differences: real LM activations have sparser feature representations and richer within-token temporal context than our toy model provides.

### Implications for temporal SAE design

The shuffle diagnostic (Experiment 1, finding 5) reveals that TFA's attention mechanism is a powerful reconstruction tool even without temporal structure. This suggests two directions:

- **For reconstruction under sparsity constraints:** attention-augmented SAEs may be useful even in non-temporal settings, as a way to provide dense reconstruction capacity alongside sparse codes.
- **For genuinely temporal feature discovery:** architectures should be designed so that the temporal pathway *cannot* function as a general-purpose reconstruction channel. This might mean restricting the predictable component to use only past tokens' codes (not the current token's query), or penalizing the predictable component's total L0 to prevent it from acting as a dense channel.

### Implications for toy model design

Our toy model places TFA in a regime where content-based matching is strong ($\pi = 0.5$, high feature overlap between tokens) and temporal signal is weak (per-feature persistence, not event-level structure). Two directions would make the toy model a better testbed for temporal exploitation:

1. **Sparser features.** Reducing $\pi$ (e.g., to 0.05--0.1) reduces incidental feature overlap between tokens, making content-based matching less powerful and temporal signal relatively more visible. The tradeoff is that this changes the binding regime ($\mathbb{E}[L_0]$ decreases).
2. **Causal mixing.** Passing the sparse activations through a small causal transformer before feeding them to the SAE would embed temporal context into each token's representation, as real middle-layer activations do. This would give TFA's proj\_scale step something temporal to amplify, and would test whether TFA can exploit within-token temporal information rather than only cross-position information.

## Limitations

- **High feature density ($\pi = 0.5$) favours content-based matching.** With 20 features and $\pi = 0.5$, any two tokens share ~5 features by chance, giving TFA's attention strong content-based matching signal that overwhelms per-feature temporal signal. Real LM features are much sparser, which would reduce incidental overlap and make temporal correlations relatively more visible. See "Why content-based matching dominates" in the Synthesis.

- **Memoryless representations (partially addressed).** The toy model is a linear embedding --- each $x_t$ depends only on features active at time $t$, with no causal mixing from prior positions. TFA-pos partially addresses this by injecting positional encoding into the attention, showing that positional information roughly doubles the temporal fraction (3--12% $\to$ 17--26%). However, real transformer representations carry richer positional information from prior attention layers, so $x_t$ itself differs between the 11111 and 00001 cases in ways our toy model cannot capture. Causal mixing remains an important untested direction.

- **Per-feature autocorrelation only.** The TFA paper claims event-level structure (co-occurring feature groups persisting across tokens). Our data has independent features with no event-level correlations. TFA's attention may be better suited to multi-feature structure; these conclusions may not generalize. (An event-structured data experiment is in progress --- see `run_event_shuffle_diagnostic.py`.)

- **Single seed (42).** No variance estimates across seeds.

- **Training config asymmetries.** The Shared SAE uses Adam (lr 3e-4, no weight decay); TFA uses AdamW (lr 1e-3, weight decay 1e-4, gradient clipping, cosine schedule). The shuffle diagnostic controls for this (TFA vs TFA-shuffled use identical configs), but the absolute Shared SAE vs TFA gap may be confounded.

- **NMSE, not support switching.** The TFA paper's Proposition 4.2 concerns support switching (disjoint codes for nearby inputs), not reconstruction error. A code-stability analysis would more directly test this claim.

## Reproduction

```bash
TQDM_DISABLE=1 python src/v2_temporal_schemeC/run_b1_topk_sweep.py
TQDM_DISABLE=1 python src/v2_temporal_schemeC/run_b1_b2_pareto.py
TQDM_DISABLE=1 python src/v2_temporal_schemeC/run_temporal_decomposition_v2.py
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_temporal_decomposition_v3.py
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_attention_direction_analysis.py
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_shuffle_diagnostic_fast.py
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_auc_and_crosscoder.py
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_txcdr_sweep_T.py 5
TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_tfa_pos_experiments.py
TQDM_DISABLE=1 python src/v2_temporal_schemeC/plot_exp1_exp2.py
```

Results: `src/v2_temporal_schemeC/results/{b1_topk_sweep, b1_b2_pareto, temporal_decomposition_v2, temporal_decomposition_v3, attention_direction_analysis, shuffle_diagnostic, auc_and_crosscoder, txcdr_T5, tfa_pos}/`.

[^1]: All TXCDR numbers in this document are from the unified reproduction framework trained for 80K steps (the correct training budget per original specification). The original document's TXCDR values were from a 30K-step run; the 80K-step values shown here are the authoritative reproduction.
