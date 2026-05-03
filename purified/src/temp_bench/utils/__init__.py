"""Cross-cutting helpers.

- :mod:`.seed`       — :func:`set_seed`
- :mod:`.gpu_locks`  — :func:`claim_gpu`, :func:`claim_gpus`,
  :func:`cleanup_stale`, :func:`gpu_lock_status` for the
  Primary + Pool GPU sharing protocol (PROTOCOL.md § 11.1)

Run-id allocation, leaderboard append, and checkpoint manifest live in
:mod:`temp_bench.cache`. Cache-key computation lives in
:mod:`temp_bench.config`.
"""

from temp_bench.utils.seed import set_seed  # noqa: F401
from temp_bench.utils.gpu_locks import (  # noqa: F401
    claim_gpu,
    claim_gpus,
    cleanup_stale,
    gpu_lock_status,
)
