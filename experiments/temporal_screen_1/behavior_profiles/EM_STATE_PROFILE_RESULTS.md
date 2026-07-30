## Medical EM state-profile pilot

The SAE-free pilot does **not** recover a robust temporal formation curve for
Medical emergent misalignment. The leave-one-prompt-out endpoint direction has
terminal macro AUC **0.604** in all four readouts, and every 95% prompt-bootstrap
interval includes chance. With only six evaluable prompt groups, the honest
conclusion is weak, inconclusive decodability rather than a feature onset.

![Medical EM state profile](em_state_profile_analysis.png)

## What was measured

- Subject: the current-paper `Qwen/Qwen2.5-7B-Instruct` bad-medical model
  organism.
- Data: its 64 already frozen, exact-reference full-context rollouts.
- Endpoint classes: 16 coherent EM-positive responses
  (`alignment <= 50`, `coherence >= 70`) and 35 coherent aligned responses
  (`alignment >= 75`, `coherence >= 70`); 13 ambiguous or incoherent responses
  were excluded.
- Length control: mean answer length was 55.69 tokens for EM-positive responses
  and 56.09 for aligned responses. Both classes therefore come from the same
  model, prompts, domain, and essentially the same length distribution.
- Representation: layer-15 `resid_post`, teacher-forced at 11 normalized
  response-progress points.
- Estimator: terminal positive-minus-negative directions trained
  leave-one-prompt-out from equal-weighted within-training-prompt contrasts.
  Evaluation AUC is calculated within each held-out prompt and then averaged
  over prompts.

The two temporal readouts are the residual at the selected token and the mean
residual over the observed response prefix. Each is reported with raw dot
products and cosine-normalized projections.

## Results

| Readout | Projection | Terminal macro AUC | 95% prompt-bootstrap interval | Peak AUC (progress) |
|---|---:|---:|---:|---:|
| Instantaneous | Raw | 0.604 | [0.319, 0.882] | 0.639 (0.5) |
| Instantaneous | Cosine | 0.604 | [0.319, 0.882] | 0.653 (0.2) |
| Prefix mean | Raw | 0.604 | [0.208, 0.938] | 0.771 (0.1) |
| Prefix mean | Cosine | 0.604 | [0.271, 0.938] | 0.715 (0.2) |

The curves are non-monotone. The prefix-mean raw readout has an isolated
pointwise lower interval above chance at 10% response progress, but the effect
is not sustained and is absent from the cosine sensitivity check. It should
not be reported as an onset. Terminal results are also heterogeneous: four
prompt groups score above chance and two below, spanning AUC 0 to 1.

The source result's progress-zero AUCs range from 0.382 to 0.562. These are
floating-point tie artifacts: before the first response token, rollouts of the
same prompt have mathematically identical causal histories, but differing
total sequence lengths can select slightly different attention kernels. The
analysis plot canonically sets within-prompt progress-zero AUC to 0.5. The raw
JSON is unchanged, and future runs canonicalize the prompt-only residual before
scoring.

## Interpretation

This differs qualitatively from the refusal calibration. Refusal has a known
causal direction, separates harmful from matched harmless prompts early, and
supports sequence-wide ablation/addition tests. Medical EM here supplies only a
rollout-level endpoint label and a weak observational direction. The comparison
supports using two protocol families:

- **event-like formation profiles** for behaviours such as refusal, where a
  candidate causal variable and meaningful onset coordinate exist;
- **state-decodability profiles** for diffuse behaviours such as EM, reported
  as evidence accumulation over normalized progress and without an onset
  claim.

This negative pilot does not show that EM has no temporal structure. It says
that this small, single-layer, terminal-direction estimator cannot establish
one. Stronger evidence would require more prompt groups, held-out endpoint
labels, a layer sweep, and causal validation of any stable direction.

## Reproduce

```bash
python -m experiments.temporal_screen_1.behavior_profiles.analyze_em_state_profile

python -m pytest -q \
  experiments/temporal_screen_1/behavior_profiles/test_em_state_profile.py \
  experiments/temporal_screen_1/behavior_profiles/test_analyze_em_state_profile.py
```

The analysis consumes
`results/em_state_profile_paper7b.json` and writes
`em_state_profile_analysis.json` plus `em_state_profile_analysis.png`.
