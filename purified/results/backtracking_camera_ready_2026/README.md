# Camera-ready backtracking results

Start with [the illustrated results walkthrough](RESULTS.md).

The `launch/` directory records the September 29, 2026 campaign setup and
validation. The `cells/` directory now contains **all 15 completed 300K
detection results**, including the three seeds of each registered dictionary.
All 15 final checkpoint weights/configs are backed up privately on Hugging Face; activation caches remain on the pod.

The campaign runs under `/workspace/backtracking` on the user-provided
four-H100 pod. All 27 steering arms finished generation before paid judging began.
The pre-judging compact archive is
`/workspace/backtracking/backtracking_compact_results.zip`. The GPU completion receipt
is `results/completion.json` on that pod; no status is inferred merely
from a successful launch.

The full compact result archive has also been saved locally as
`backtracking_compact_results.zip` (about 145 MB), outside Git because it
contains raw generations and all OOF arrays. Its manifest verifies every
member. The reproducible code, compact result tables and current paper
figures are tracked separately. Model weights and activation datasets were
not downloaded. The private checkpoint backup is hash-verified at revision
`e0b19fd1cbd739be38f76d066441c848b5354b92` in
`aniketdesh/temporal-crosscoders-backtracking-300k-2026-09-30`.
See `checkpoint_backup_receipt.json` for all 35 verified files. No pod stop or
delete action was taken.

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

The user authorized paid OpenAI judging on September 30. The resumable
`judge_openai.py` runner uses `gpt-6-luna` with low reasoning, the original
rubrics, exact-request caching, and a $9 cap. Validation selects and freezes
each arm's signed magnitude before any test candidates are released for judging.
Current judging status is recorded separately from the earlier GPU completion
receipt; no effect is claimed from a launch alone.

See the [protocol and execution README](../../experiments/backtracking_camera_ready_2026/README.md).

## Completed Luna judging

All 27 steering arms have completed validation-gated test judging with
`gpt-6-luna`. Estimated token cost is $1.181680675, including cache writes and
reasoning tokens; this is a usage-based estimate, not an invoice. The original
20/100 validation/test split and the original rubrics are unchanged. All 27
zero-intervention runs produce the same token sequences on each test question.

`publication/steering/` contains per-question and per-seed tables, paired
comparisons, title-free Nord PDF/SVG/PNG figures, and external caption notes.
The main steering figure shows conditional paired question-bootstrap 95%
intervals, with individual seed means as small points. The validation curve
bands and the detection figure bars remain sample SD across seeds. These
uncertainty measures are explicitly distinguished in the walkthrough.

The full judged archive is saved locally as `backtracking_judged_results.zip`
outside Git. `judged_local_backup_receipt.json` verifies every member.
`judged_backup_receipt.json` records the separate private HF upload; the
earlier checkpoint backup revision remains an immutable reference.
The temporary OpenAI credential was removed after completion.
