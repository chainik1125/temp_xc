"""GPU-streaming sample generators for a CPU-resident anchor buffer.

Used by hill-climb V2 (Subseq H8 T_max=16) on the local 5090. With the
14 GB anchor buffer kept in CPU RAM (not GPU VRAM), the full 32 GB of
the 5090 is available for W + Adam + activations. Per-step we slice
the small (B, ..., d) sub-tensor on CPU then transfer it to GPU.

Mirrors the GPU-resident generators in
`experiments/phase5b_t_scaling_explore/_train_utils.py` 1:1 in shape
and semantics. Only difference: buf is on CPU, and the per-batch
result is pinned + asynchronously transferred to GPU.

Cost per step at b=4096:
  - subseq H8 T_max=16, 1+K=4 (shifts=[1, 4, 8]):
      slice shape (4096, 4, 16, 2304) fp16 = 1.21 GB
      PCIe 5.0 transfer: ~20 ms (≈ 60 GB/s effective)
  - small relative to ~250 ms forward+backward at this scale.
"""
from __future__ import annotations

import torch


def make_multidistance_pair_gen_cpu_to_gpu(
    buf_cpu: torch.Tensor,
    T: int,
    shifts: list[int],
    device: str = "cuda",
):
    """CPU buffer + GPU stream variant of make_multidistance_pair_gen_gpu.

    Args:
      buf_cpu: (N, L, d) fp16 tensor, CPU-resident. Will be pinned to enable
               non-blocking transfers.
      T: window length.
      shifts: list of strictly-positive lag values for InfoNCE pairs.
      device: target device for batches.

    Returns:
      gen_fn(B) -> (B, 1+K, T, d) float32 on GPU.
    """
    if not buf_cpu.is_pinned():
        buf_cpu = buf_cpu.pin_memory()
    N, L, d = buf_cpu.shape
    max_shift = max(shifts)
    assert L >= T + max_shift, f"need L>={T+max_shift}; got L={L}"
    cuda_dev = torch.device(device)

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,))
        off = torch.randint(0, L - T - max_shift, (batch_size,))
        rng = torch.arange(T)
        outs = []
        for s in [0] + list(shifts):
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)            # CPU
            w = buf_cpu[seq.unsqueeze(1).expand(-1, T), pos]            # CPU fp16
            outs.append(w)
        stacked = torch.stack(outs, dim=1)                              # (B, 1+K, T, d) CPU fp16
        return stacked.to(cuda_dev, non_blocking=True).float()
    return gen


def make_window_gen_cpu_to_gpu(
    buf_cpu: torch.Tensor,
    T: int,
    device: str = "cuda",
):
    """CPU buffer variant of make_window_gen_gpu. Returns (B, T, d) float32 GPU."""
    if not buf_cpu.is_pinned():
        buf_cpu = buf_cpu.pin_memory()
    N, L, d = buf_cpu.shape
    n_wins = L - T + 1
    cuda_dev = torch.device(device)

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,))
        off = torch.randint(0, n_wins, (batch_size,))
        rng = torch.arange(T)
        pos = off.unsqueeze(1) + rng.unsqueeze(0)
        w = buf_cpu[seq.unsqueeze(1).expand(-1, T), pos]
        return w.to(cuda_dev, non_blocking=True).float()
    return gen


def preload_single_cpu(layer_key: str | None = None,
                       n_seqs: int | None = None) -> torch.Tensor:
    """Load the L12 anchor cache as a CPU-resident fp16 tensor (pinned)."""
    import numpy as np
    from experiments.phase7_unification._paths import (
        CACHE_DIR, ANCHOR_LAYER_KEY, PRELOAD_SEQS,
    )
    if layer_key is None:
        layer_key = ANCHOR_LAYER_KEY
    if n_seqs is None:
        n_seqs = PRELOAD_SEQS
    arr = np.load(CACHE_DIR / f"{layer_key}.npy", mmap_mode="r")
    sub = np.asarray(arr[:n_seqs], dtype=np.float16)
    t = torch.from_numpy(sub)
    return t.pin_memory()
