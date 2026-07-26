"""ACTMIX RLHF — shared decomposition + metrics.

The single implementation lives in `temp_bench.evals.rlhf` (protocol
2.0.0 port) so the canonical-runner arm and this exploration's
paper-match case-study cannot diverge; this module re-exports it
(experiments → src layering). `papermatch.py` was first run against
the identical function bodies defined here — the post-refactor re-run
is diffed as an identity check (see results/).
"""

from temp_bench.evals.rlhf import (   # noqa: F401
    _shuffle_windows,
    aggregate_response_mean,
    preference_auc,
    mass_at_k,
    length_pearson_topk,
)
