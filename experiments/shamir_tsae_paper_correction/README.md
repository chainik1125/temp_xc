## Corrected Shamir baseline and probe-width comparison

The original `shamir_camera_ready_k_sweep` arm named `tsae` actually used
`temporal_crosscoders.han_tsae.TemporalSAE`: an attention-based model vendored
from TemporalFeatureAnalysis, with an added InfoNCE loss. It is not the published
Bhalla T-SAE. The original checkpoints and numerical results are preserved, but
that curve is excluded from the corrected figure.

`run.py` uses the existing per-token `TSAEPaper` port from
`experiments/temporal_screen/txc_wins/tsae_bhalla.py`. This is a Matryoshka
BatchTopK SAE with high-level adjacent-token InfoNCE, AuxK, and threshold
inference. The upstream implementation is
[TemporalMatryoshkaBatchTopKSAE](https://github.com/AI4LIFE-GROUP/temporal-saes/blob/main/dictionary_learning/dictionary_learning/trainers/temporal_sequence_top_k.py).
The local port records its adaptations; this run is not an exact reproduction of
the paper's LLM training schedule or hyperparameters.

Six models were trained: h=1,2 × seeds 0,1,2, q=11, k=5, width 2048. Each model
is trained once on adjacent-token pairs and then held fixed for every probe W.
Settings: 6000 steps, batch128 pairs, Adam LR .001, gradient clipping1,
20% high-level features (rounded), contrastive coefficient1, AuxK coefficient1/32,
decoder normalization, threshold EMA .999 after step1000, dead threshold10M tokens.
No hyperparameters were selected using the evaluation labels. Both loss
group weights are1; the decoder bias starts at0, as in the existing local port.

`readout_sweep.py` implements the user's comparison:

- TXC loads its W-dependent trained model and encodes W inputs into one shared code.
- T-SAE uses one per-token model per h/seed; the probe reads W concatenated codes.
- TFA reuses its model trained with W=10. It always encodes the same ten-token
  context; only the number of final token codes read by the probe varies.
- Single-token TopK SAE uses one model and one single-token probe, shared across W.
- Stacked TopK SAE concatenates W codes from that same token SAE.

Each episode uses the endpoint of its original W=10 anchor. Smaller readouts
are nested suffixes ending at that position. Thus changing W does not also change
the sampled episode or endpoint. Representation training, probe training, and
evaluation remain episode-disjoint, with 4096 episodes each. Probes retain the
original 1500-step Adam recipe and the original probe seeds.

TFA has access to ten observations even when its probe reads one token's code.
The W≤h information ceiling therefore does not apply to TFA in this comparison.
The same nominal k does not match actual L0 or parameter counts across methods.
T-SAE threshold inference is checked to be tokenwise, and baseline model hashes
are checked to be identical across all readout widths. The sweep completed 60
evaluation jobs, yielding 210 primary rows plus 42 single-token T-SAE diagnostics,
with no failed jobs or flagged information-ceiling controls.

Remote root: `simplex2:/data/users/dmitry/shamir_tsae_paper_correction_20261107/`.
`full/` contains the six corrected T-SAE models and initial probes;
`readout_sweep/` contains the final matched-endpoint probes, predictions, CSV,
plans, and validation. Only GPU5 was used. Frozen source copies and hashes are in
`source/` and `source_manifest.json`. Large checkpoints remain remote.

Local numerical evidence and frozen source copies are in the sibling paper
repository under `notes/shamir_rerun_audit/clean_q11_k5_k10/fixed_model_readout/`.
The camera-ready figure uses that bundle's `readout_sweep/cells.csv`.
