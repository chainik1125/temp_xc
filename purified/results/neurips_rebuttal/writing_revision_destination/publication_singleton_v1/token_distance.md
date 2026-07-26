# Predicting deletion distance from pre-deletion activations

![capped_token_label](token_distance.png)

Protocol `klicke-deletion-raw-activation-gate-v1`; 6,224 events from 2,510 writers. The best ordered model is T=6 (log loss 1.239, balanced accuracy 0.475). Positive paired gaps mean the control is worse than ordered history.

| Control at best T | Control − ordered log loss | 95% CI |
|---|---:|---:|
| Best single offset | 0.201 | [0.176, 0.226] |
| Last token | 0.201 | [0.177, 0.226] |
| Order-invariant summary | 0.289 | [0.263, 0.315] |
| Shuffled history (refit) | 0.416 | [0.386, 0.447] |
| Second differences | 0.105 | [0.083, 0.125] |

The right panel uses an equal-writer bootstrap, so prolific writers cannot dominate the uncertainty estimate.
