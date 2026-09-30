# Validation-only judge spot check

Before test judging, eight saved validation judgments were inspected: four positive backtracking counts, two zero counts, and two low-coherence cases. Cases were selected deterministically by request hash within these categories from the labels then available. This is an assistant spot check, not a random sample, a blinded human study, or an accuracy estimate.

The inspected labels distinguish arithmetic/approach revisions from repeated answers and identify two strongly repetitive continuations as low coherence. One GCD example is borderline: the model changes an intermediate result in its final-answer explanation without explicitly acknowledging its earlier error, and the judge counts this as a correction. This illustrates why human rubric calibration is still needed before camera-ready claims about label reliability.

No labels, prompts, magnitude-selection rules, or test cohorts were changed after this inspection. All original responses remain archived. `validation_spotcheck.jsonl` contains the exact inspected examples and request identities.
