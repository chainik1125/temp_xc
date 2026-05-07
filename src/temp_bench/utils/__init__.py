"""Cross-cutting helpers.

- :mod:`.seed`       — :func:`set_seed`
- :mod:`.tokens`     — :func:`get_token`, :func:`require_token`,
  :func:`token_status` — single canonical resolution chain for HF,
  Anthropic, and GitHub tokens across local and RunPod.

GPU sharing on multi-agent pods is now a CONVENTION (no lockfile
manager) — see PROTOCOL.md § 12. Each agent has a primary GPU pinned
by ``scripts/set_[pipeline].sh``; peer GPUs are fair game when peer
is idle (verify by reading peer's briefing + ``nvidia-smi``).

Run-id allocation, leaderboard append, and checkpoint manifest live in
:mod:`temp_bench.cache`. Cache-key computation lives in
:mod:`temp_bench.config`.
"""

from temp_bench.utils.seed import set_seed  # noqa: F401
from temp_bench.utils.tokens import (  # noqa: F401
    get_token,
    require_token,
    token_status,
    tokens_dir,
)
