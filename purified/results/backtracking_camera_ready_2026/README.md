# Camera-ready backtracking results

The `launch/` directory records the September 29, 2026 campaign setup and
validation. **These are launch artifacts, not completed 300K results.**

The GPU experiment queue is running under `/workspace/backtracking` on the
user-provided four-H100 pod. Its final compact archive is configured to be
`/workspace/backtracking/backtracking_compact_results.zip`. The final status
will be in `results/completion.json` on that pod; no status is inferred merely
from a successful launch.

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
