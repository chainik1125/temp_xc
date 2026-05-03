"""Cross-cutting helpers: seeding only.

Run-id allocation, leaderboard append, and checkpoint manifest now live
in :mod:`temp_bench.cache` (single canonical pathway, see
``docs/paper/framework.md``). Cache-key computation lives in
:mod:`temp_bench.config`.
"""

from temp_bench.utils.seed import set_seed  # noqa: F401
