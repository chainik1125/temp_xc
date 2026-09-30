# Paper-ready backtracking detection figures

The main figure is `detection_headline_matched_scaled_C1.pdf`. Use vector PDF in LaTeX; SVG is editable and PNG is a 400-dpi preview. No manuscript file is changed. PDFs are generated at the actual 5.5-inch NeurIPS text width or the existing 2.585 x 1.98-inch C7 half-width slot. Do not shrink the full-width panels into a half-width slot. Font sizes are 7-8 pt at native size, with embedded TrueType fonts.

## Shared caption details

Backtracking detection after 300,000 dictionary-training steps for each of three seeds (1, 2, 42). Average precision is averaged equally across five folds grouped by normalized problem text: 25,204 sentences from 300 archive IDs representing 213 distinct prompts. Features are selected and probes fitted on training folds only. The dashed gray reference is the constant-score baseline, the mean held-out-fold prevalence (0.1257). All methods use offsets -12 through -8. These prompt-grouped results are distinct from the historical question-ID split. S counts total selected coordinates: in the 32K core comparison, TXC and shared readouts have 32,768 candidates; independent Stacked SAE has 5 x 32,768 position-specific candidates. Parameter counts and reconstructed-token exposure are not matched by equal S or equal optimizer steps.

## Headline at S=8

Large symbols and horizontal error bars show the mean and sample standard deviation across dictionary seeds. Small dots show individual seeds, with vertical offsets only for visibility. All eight 32K method/readout comparisons are shown in a fixed order; no pooling rule is selected after seeing the results. The dots are not bars, so the AP axis is narrowed to the observed range.

## Budget curves

Lines show the three-seed mean, and shaded bands show sample standard deviation, not confidence intervals. From left to right, the three panels use the shared encoder's last-token, mean-pool, and max-pool readouts; TXC and independent Stacked SAE repeat as fixed references, not independent measurements. Markers and line patterns identify architectures in grayscale. All curve figures share the same AP limits.

## Width sensitivity

T-SAE 32K and 16K (32,768 versus 16,384 candidates) are compared using identical seed sets and training duration. Lines and bands indicate the mean and sample SD. These results do not establish a globally optimal T-SAE configuration.

## Probe scaling

`matched_scaled_C1` is the predeclared primary: each feature is divided by its training-fold population standard deviation, without centering, before ranking and fitting a C=1 probe. `historical_raw_C1` is a separate raw-feature sensitivity under the new canonical-prompt split; it is not a historical-result replay.

## Scope

These are detection figures. Steering figures require deferred judge labels and validation-based magnitude selection; no unjudged steering result is presented as an effect. Paired prompt-bootstrap CIs, when available, are separate from the seed SD shown here and condition on fixed trained dictionaries/probes.
