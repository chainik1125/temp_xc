"""NLP activation caching + batch iterators for real-LM components.

Used by C3 (sparse probing), C4 (qualitative latents), C5 (steering), C7
(backtracking) — all components that need cached residual-stream
activations from a HuggingFace LM.

Public API:

- :func:`build_activation_cache` — build & save the cache for a
  datasource entry. Idempotent: if the cache exists with the right
  shape, no recomputation.
- :func:`batch_iter_from_act_cache` — return a deterministic callable
  ``(batch_size: int) -> Tensor`` for the canonical SAE trainer.

Both are keyed by ``act_cache_key`` from
:func:`temp_bench.config.compute_act_cache_key`. Cache layout:

    results/act_cache/<act_cache_key>/
      ├ acts.npy       # (n_seqs, seq_len, d_in) fp16
      ├ token_ids.npy  # (n_seqs, seq_len) int64
      └ meta.json      # the datasource spec (for verification)

Ported in spirit (not line-by-line) from
`origin/han-phase7-unification @ 94119bc0:src/data/nlp/cache_activations.py`,
shedding wandb / sweep / CLI cruft to fit the unified framework.
"""

from temp_bench.data.nlp.cache import (
    batch_iter_from_act_cache,
    build_activation_cache,
)

__all__ = ["build_activation_cache", "batch_iter_from_act_cache"]
