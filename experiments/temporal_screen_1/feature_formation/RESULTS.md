## Bottom line

The first feature-formation calibration is technically successful but
mechanistically negative.

Both the SAE-free residual readout and the conventional-SAE readout become
strongly predictive only one or two tokens before a judged genuine
backtracking sentence. A readout fixed in Ward's known -13 to -8 token
precursor band does not yield a stable, well-calibrated feature that grows
through the event.

This is more consistent with a changing representational trajectory that
culminates in textual expression than with one conventional residual feature
smoothly forming over tens of tokens.

## Setup

- Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`.
- Representation: residual output after layer 10.
- Data: 268 existing Ward rollouts with a usable genuine-backtracking event.
- Positive anchor: the first usable genuine-backtracking sentence.
- Neutral anchor: a position in the same rollout at least 96 tokens from every
  judged event, usually 160 tokens from the positive anchor.
- Evaluation offsets: -64 through +16 tokens around each anchor.
- Known precursor band: -13 through -8 tokens.
- Cross-validation unit: complete positive/neutral pair.

The positive and neutral anchors are intentionally from the same rollout.
This controls prompt, rollout style, length, and global backtracking
propensity, but it does not make the two local contexts semantically
exchangeable.

## SAE-free result

The SAE-free representation is the fixed 32-dimensional Rademacher projection
from the earlier blind weak-label experiment. The projection was chosen
without event labels.

| Curve | Far-pre AUC | Ward-band AUC | Immediate-pre AUC |
| --- | ---: | ---: | ---: |
| Positionwise local residual | 0.606 | 0.586 | 0.808 |
| Positionwise trailing six-token window | 0.703 | 0.695 | 0.825 |
| Fixed Ward-band local readout | 0.520 | 0.566 | 0.606 |
| Fixed Ward-band six-token readout | 0.539 | 0.624 | 0.610 |

The positionwise local log-loss gain is approximately 0.007 nats far before
the event, -0.001 in the Ward band, and 0.159 immediately before the event.
Its observed peak is at offset -1.

The fixed six-token readout has a visible Ward-band AUC shoulder, but its
proper-score gain is negative throughout most of that region. It ranks some
examples correctly while being badly calibrated and is not a robust forming
feature.

## Conventional-SAE result

An unsupervised single-token TopK SAE was trained on 76,800 sampled layer-10
residuals:

- width 8,192;
- \(k=32\);
- 1,200 optimizer steps;
- no behaviour labels during SAE training;
- centered FVU 0.480;
- 29.6% of features active in an 8,192-token audit.

The reconstruction and activity diagnostics make this a rough first-pass SAE,
not a paper-grade dictionary.

| Curve | Far-pre AUC | Ward-band AUC | Immediate-pre AUC |
| --- | ---: | ---: | ---: |
| Positionwise best held-out top-16 features | 0.630 | 0.652 | 0.900 |
| Fixed Ward-band single feature | 0.523 | 0.534 | 0.563 |
| Fixed Ward-band top-16 features | 0.538 | 0.596 | 0.649 |

The positionwise SAE curve peaks at offset -1 with AUC 0.919 and a 0.356-nat
log-loss gain. In contrast, the transported single feature has negative
proper-score gain even in the Ward band. The feature selected in the discovery
band is not stable across folds: feature 5943 is selected in three of five
folds, while two other features are selected once each.

## Interpretation

Three claims are supported:

- Feature formation can be operationalized and measured with both raw
  residuals and conventional SAE features.
- At layer 10, imminent backtracking becomes very locally decodable just
  before the judged sentence begins.
- The current data do not support one stable conventional SAE feature
  gradually forming across Ward's precursor band.

Three stronger claims are not supported:

- that the -1 token peak is causal rather than the first expression of the
  behaviour;
- that the weak Ward-band shoulder is specific rather than ordinary local
  trajectory structure;
- that temporal crosscoders will outperform conventional SAEs on this basis.

The split between strong positionwise prediction and weak transported-feature
prediction is nevertheless informative. It suggests that the predictive
direction changes through time. A temporal crosscoder could package such a
trajectory even when no single SAE feature transports cleanly, but that
possibility needs a direct architecture comparison and causal intervention.

## Next decisive experiment

For sampled checkpoints, estimate the branch value

\[
V_t = \mathbb{E}[\text{eventual behaviour metric}\mid S_t]
\]

from multiple continuations. Measure how well \(V_t\) is captured by the local
residual, a residual window, and conventional SAE features. At the same
checkpoints, mask attention lag bands and test whether causal lag mass
contracts as local decodability rises.

A formation claim should require both:

\[
\text{local sufficiency}(t)\uparrow
\qquad\text{and}\qquad
\text{causal memory radius}(t)\downarrow.
\]

The next SAE should also pass a stricter reconstruction/activity gate before
its negative result is trusted.

## Artifacts

- `estimators.py`: SAE-free positionwise and transported estimators.
- `sae_features.py`: sparse conventional-SAE feature estimators.
- `run_sae_free.py`: cached Ward SAE-free calibration.
- `modal_ward_sae.py`: activation extraction, SAE training, and evaluation.
- `results/ward_sae_free.json`: complete SAE-free curves.
- `results/ward_sae_features.json`: complete conventional-SAE curves.
- `results/ward_feature_formation_band_summary.json`: controlled band means.
- `results/ward_feature_formation_curves.png`: combined figure.
