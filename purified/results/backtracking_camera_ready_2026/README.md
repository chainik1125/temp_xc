# Camera-ready backtracking results

The `launch/` directory records the September 29, 2026 campaign setup and
validation. The `cells/` directory now contains **all 15 completed 300K
detection results**, including the three seeds of each registered dictionary.
Large checkpoints and activation caches remain on the pod.

The campaign runs under `/workspace/backtracking` on the user-provided
four-H100 pod. All 27 steering arms have now finished generation, with no
paid judging performed. Its final compact archive is
`/workspace/backtracking/backtracking_compact_results.zip`. The final status
will be in `results/completion.json` on that pod; no status is inferred merely
from a successful launch.

The full compact result archive has also been saved locally as
`backtracking_compact_results.zip` (about 145 MB), outside Git because it
contains raw generations and all OOF arrays. Its manifest verifies every
member. The reproducible code, compact result tables and current paper
figures are tracked separately. Model weights and activation datasets were
not downloaded. Checkpoints still need durable storage before the pod is
stopped or deleted.

`publication/detection/` contains title-free Nord figures at their intended
paper sizes, with vector PDF, editable SVG and PNG previews. The main figure
is `detection_headline_matched_scaled_C1.pdf`; readout curves and T-SAE width
sensitivity have separate full-width figures. `CAPTIONS.md` and
`include_figures.tex` provide caption notes and inclusion commands. Source
tables, three-seed values, hashes and exact dimensions are retained alongside
the figures. The manuscript has not been changed.

The two originally unconverged Stacked seed-2 fits at S=16/32 now converge
after increasing only the probe iteration cap. Original results and full
repair receipts are retained under that cell's `numerical_repairs/` directory;
S=8 and all unaffected fits are unchanged. All primary probes now have zero
convergence warnings. Historical diagnostic warnings remain preserved.

The primary detector now groups 300 question IDs into 213 canonical problem
texts. `prompt_grouping_audit.json` shows that the old question-ID folds put
9,631 of 25,204 test sentences (38.2%) alongside identical problem text in
their training folds; the new prompt-grouped split eliminates that overlap.

The generation pilot passed exact zero-hook equality on eight validation
examples. Seven of eight cut/continue suffixes matched exactly; the remaining
BF16 prefix-recomputation difference is retained in `phase1_zero_check.json`.
This is not a claim of bit-for-bit historical generation reproduction.

Paid API judging is deferred. Saved test candidate generations must not be
graded or selected before validation chooses the intervention magnitude.

See the [protocol and execution README](../../experiments/backtracking_camera_ready_2026/README.md).
