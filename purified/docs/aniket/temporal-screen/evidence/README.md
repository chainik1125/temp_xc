# Temporal Screen evidence ledger

This directory records the full-text literature audit used to design the
Temporal Screen. A source counts as reviewed only when its card cites
paper-level evidence beyond the abstract. Duplicate appearances in the
annotated reading list point to one source card.

Every card records:

- the paper's scientific question and observable;
- the temporal object, estimand, estimator, and null comparison;
- assumptions about stationarity, linearity, direction, and sampling;
- whether the quantity is aligned to a downstream target;
- dependence units, splits, and uncertainty treatment;
- the result relevant to the Temporal Screen;
- author-stated limitations and additional failure modes;
- the screen decision the evidence can and cannot support;
- a section, theorem, figure, or page pointer to the primary source.

Papers that could directly instantiate part of the screen additionally record
their algorithmic inputs and outputs, material hyperparameters, computational
requirements, estimator bias, available implementations, required adaptation
to grouped language-model activations, and a synthetic falsification case.

The audit is deliberately method-neutral. Correlation decay, spectral
statistics, target-conditioned prediction, and temporal interventions remain
competing screen families until the contradiction matrix and synthetic
falsification suite are complete.
