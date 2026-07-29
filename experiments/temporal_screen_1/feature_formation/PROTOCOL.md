## Question

When does behaviour-relevant information become a locally available residual
feature, rather than remaining distributed across several sequence positions?

This pilot compares two measurements on the same frozen Ward Backtracking
rollouts:

- an *SAE-free* readout from a fixed random projection of the layer-10
  residual stream;
- a *conventional-SAE* readout from TopK SAE latents trained without behaviour
  labels on the same activation distribution.

## Calibration panel

The calibration deliberately uses the existing genuine-backtracking sentence
positions. For each rollout with a usable event:

- the first genuine event supplies the event anchor;
- a token in the same rollout, at least 96 tokens from every labelled event,
  supplies a neutral anchor;
- both anchors contribute the same relative offsets from -64 through +16;
- cross-validation holds out complete event/neutral pairs;
- the primary formation interval is strictly pre-event;
- the known Ward discovery band, -13 through -8 tokens, is used only to fit a
  fixed transported readout.

Using event positions makes this a calibration of the proposed quantity, not
yet a task-agnostic screen.

## Curves

For a residual or SAE-latent panel \(X\), the positionwise curve trains a new
held-out linear readout at every relative offset. It asks whether *some*
linearly available representation predicts an imminent event.

The transported curve trains one readout in the predeclared Ward band and
reuses that exact readout at every offset. It asks whether one fixed precursor
representation grows, persists, or decays through time.

The main scores are held-out ROC-AUC and log-loss improvement over the balanced
class prior. Formation thresholds are reported descriptively from the positive
log-loss-gain curve; the complete non-monotone curve remains primary.

## Interpretation gates

The pilot is useful only if:

- the synthetic estimator recovers a planted gradual feature and stays flat on
  a null panel;
- pre-event discrimination rises before token zero rather than only after the
  labelled phrase begins;
- the fixed precursor readout shows a coherent transported curve;
- the SAE result is not driven by dead features or unacceptable
  reconstruction error;
- neutral anchors and held-out pair splitting prevent rollout identity from
  solving the task.

Neither probe curve is by itself causal. A subsequent checkpoint-branching run
must estimate the eventual behaviour value and test attention-lag masks before
the formation curve can support a mechanistic claim.
