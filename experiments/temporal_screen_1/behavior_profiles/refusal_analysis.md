## Refusal prefix profile: result

The clean result is that the published refusal direction forms *before or at*
the point at which refusal becomes visible in generated text. This makes
refusal a useful positive-control behaviour for a feature-formation screen:
the internal readout, the generated behaviour, and a causal intervention all
refer to the same known direction.

The experiment used 32 held-out harmful instructions and 32 one-to-one
length-matched harmless instructions with Llama-3-8B-Instruct. It revealed
nine literal token prefixes from 0% to 100%, measured the published Arditi
direction at layer 12 before generation, and greedily generated up to 48
tokens. The exact Arditi directional ablation and direction addition were the
causal controls.

![Refusal temporal profile](refusal_analysis.png)

## What turns on, and when?

The decision-position direction is already a strong harmful-versus-harmless
classifier after only 12.5% of the instruction is revealed: AUC is 0.866,
compared with 0.444 for the published response-start refusal-token log odds.
At this point both cohorts have the same observed lexical refusal rate,
3/32. The direction AUC rises to 0.942 at 25%; refusal-token-log-odds AUC does
not exceed 0.8 until 37.5%.

The mean harmful decision-position projection rises from 0.190 at the empty
prefix to 0.466 at 12.5%, 1.019 at 25%, and 1.501 at 37.5%. The matched
harmless means are 0.189, 0.061, 0.006, and 0.011. Harmful lexical refusal
rises from 0/32 at the empty prefix to 3/32, 11/32, 20/32, 27/32, and 31/32
at 12.5%, 25%, 37.5%, 50%, and 62.5%, then remains at 31/32.

For a prompt-level estimate, define direction formation as the first non-empty
prefix from which the decision-position projection remains above the
same-fraction 95th percentile of the harmless cohort. This threshold is
deliberately neutral-calibrated and requires persistence, rather than treating
a single noisy crossing as onset.

- All 32 harmful prompts have a stable direction onset.
- 31/32 eventually produce a lexical refusal; no prompt switches back after
  its first refusal.
- Among those 31, direction formation is at or before lexical refusal for
  31/31, and strictly before it for 20/31.
- The median refusal-minus-direction lag is 0.125 of the prompt, or 3 prompt
  tokens; the means are 0.141 and 4.10 tokens.
- Median stable direction onset is 7 revealed prompt tokens. Median lexical
  refusal onset is 10 revealed prompt tokens.

This supports a narrow temporal claim: under this reveal intervention, harmful
evidence is consolidated into a known causal refusal state before the state is
expressed as lexical refusal. It does **not** say that refusal has a universal
seven-token horizon; the onset varies with where harmful information occurs
in each instruction.

## Causal check

The causal controls are unusually clean.

- Full harmful prompts refuse in 31/32 baseline generations and 0/32 after
  all-layer ablation of the published direction.
- Full harmless prompts refuse in 0/32 baseline generations and 32/32 after
  adding the single published direction at its selected layer.
- Ablated harmful prompts produce 0/32 lexical refusals at every reveal
  fraction, while their measured direction projection is approximately zero.

So the measured direction is not merely correlated with the class label. In
this setup it mediates the observed refusal and is sufficient to induce it.
The fact that a single direction is a sufficient effector still does not imply
that the prompt evidence used to form that direction came from a single token.

## Important limitations

- A token prefix is an intervention on *semantics*, not a pure clock. It can
  cut a word or proposition before the token that makes the request harmful.
  The curve therefore measures information revealed under literal
  left-prefix truncation. The result file intentionally stores hashes rather
  than prompt text, so semantic-boundary alignment cannot be recovered from
  this artifact alone.
- The 0% condition still contains the fixed Llama-3 user/assistant chat
  template. Its nonzero projection (about 0.19 at the decision position) is a
  template offset, not task evidence. Harmful and harmless inputs are
  semantically identical there; the nominal 0%-bin AUC values should be
  treated as 0.5 calibration, with departures caused by batching,
  left-padding, and finite-precision tie-breaking. Two distinct empty-prefix
  response hashes are present despite identical rendered inputs, so this is a
  real implementation artifact.
- Partial harmless prompts sometimes trigger the lexical classifier (up to
  4/32), even though full harmless prompts never do. This is direct evidence
  that truncation can create unnatural or ambiguous inputs, and is why the
  harmless curve is essential.
- Length matching is good for 28/32 pairs (within five tokens) but has four
  outliers, including absolute gaps of 30 and 35 tokens. Excluding those four
  leaves the early result essentially unchanged: decision-direction AUC is
  0.857 at 12.5% and 0.932 at 25%.
- The prompt-level neutral threshold is estimated from only 32 harmless
  examples. It is a transparent descriptive onset rule, not a pre-registered
  population cutoff.
- This is one model, one published direction, deterministic greedy decoding,
  and a high-precision but incomplete lexical refusal classifier. It locates
  prompt-side formation, not a rollout-internal onset after generation begins.

## Reproduction

Run:

```bash
python experiments/temporal_screen_1/behavior_profiles/analyze_refusal_profile.py
```

This regenerates `refusal_analysis.json` and `refusal_analysis.png` from
`results/refusal_prefix_profile.json`.
