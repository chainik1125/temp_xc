"""Training loop. Single trainer; arch picks its loss + sparsity.

The trainer is responsible for:
  - mixed-precision (bf16 on H100/H200; fp16 fallback on A40)
  - dead-feature reset (anti-dead stack used by TXC-base + TXC-pro)
  - checkpointing every N steps + final
  - leaderboard.jsonl emission on completion (via runner.run_cell)

Per-arch overrides (loss, sparsity, contrastive, matryoshka) live in
the arch's ``train_step`` method on :class:`TempBenchArch`. See
PROTOCOL.md § 11 *Code reuse contract*.
"""

from temp_bench.training.sae_trainer import train_sae  # noqa: F401
