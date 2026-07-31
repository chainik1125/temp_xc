## Shared-SAE pooling on C7 backtracking

This eval-only experiment asks whether the reported C7 advantage of temporal
crosscoders survives giving an ordinary SAE the same five-token input window.

### Confound being repaired

The canonical C7 evaluator reads `model.config.T`. For the TopK SAE this is
one, so it slices the six cached pre-sentence activations to the final token
before encoding. TXC-base receives the final five. The existing Stacked SAE
does receive five tokens, but it trains an independent dictionary at each
position and then max-pools equal integer feature IDs; those IDs are not
aligned across dictionaries.

Here a *single frozen SAE* encodes each of the final five positions. Because
the dictionary is shared, feature ID `f` has the same decoder direction at
every position and can be pooled coherently.

### Frozen inputs

- Checkpoint: `f437e623fabc37ec`, TopK SAE, seed 42, 20,000 steps,
  `d_sae=32768`, `k=20` per token.
- Data: canonical C7 `sentence_acts_L10.npz`, 25,204 sentences, cached window
  of six Llama-3.1-8B BASE layer-10 residual activations.
- Evaluated window: the final five cached positions, matching TXC-base.
- Probe: train-fold feature selection followed by L1 logistic regression,
  grouped five-fold CV by question ID, PR-AUC at `S={1,2,4,8,16,32}`.

### Pre-registered arms and decision

- Validation: final-token code must reproduce the existing TopK SAE curve to
  absolute tolerance 0.002.
- Primary: temporal mean and temporal max of the five aligned code vectors.
- Diagnostics: first/final token, each individual position, geometric recency
  and reverse-recency weighting.
- Sparsity: untruncated pooling has support at most `5*20=100`, equal to the
  TXC-base nominal window code budget. Mean/max/recency are also evaluated
  after top-20 truncation.
- Steering gate: proceed to new judged generations only if a primary fixed
  pool reproduces the old baseline and matches or beats matched-20k TXC-pro at
  `S=8`.

Mean and max are exactly invariant to within-window permutation, so a shuffle
test cannot establish ordering for those arms. Recency versus reverse-recency
is included only as a positional diagnostic, not as a post-hoc primary arm.

