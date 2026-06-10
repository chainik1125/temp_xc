"""Training loop. Single trainer; arch picks its loss + sparsity.

The trainer is responsible for:
  - mixed-precision (bf16 on H100/H200; fp16 fallback on A40)
  - dead-feature reset (anti-dead stack used by TXC-base + TXC-pro)
  - checkpointing every N steps + final
  - leaderboard.jsonl emission on completion

Per-arch overrides (loss, sparsity, etc.) are folded into ``ArchConfig``
and consumed by the trainer through callbacks.
"""
