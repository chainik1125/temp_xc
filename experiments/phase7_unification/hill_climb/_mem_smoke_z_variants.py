"""GPU memory smoke test for Z's SubseqH8 variants.

Runs 5 forward+backward iterations at paper-canonical (b=4096, d_in=2304,
d_sae=18432, fp32) on a SYNTHETIC random buffer — no activation cache
needed. Reports peak VRAM and rough step time so we can verify the new
variants fit the 5090 (32 GB) before waiting on the cache rebuild.

Usage:
    .venv/bin/python -m experiments.phase7_unification.hill_climb._mem_smoke_z_variants

Targets at paper-canonical b=4096:
  - SubseqRankedH8 T_max=20 t_sample=5: ought to fit comfortably (~12-18 GB)
    [original SubseqH8 at T_max=20 OOM'd at >32 GB]
  - SubseqSharedH8 T_max=20: trivially fits (<10 GB)
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

from experiments.phase7_unification._paths import DEFAULT_D_IN, DEFAULT_D_SAE
from src.architectures.phase7_subseq_z_variants import (
    SubseqRankedH8,
    SubseqSharedH8,
)


N_ITERS = 5
BATCH = 4096


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def smoke_variant(name: str, model: nn.Module, T_max: int,
                   shifts: tuple[int, ...]) -> None:
    print(f"\n=== {name} (T_max={T_max} shifts={shifts}) ===", flush=True)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    model = model.to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Synthetic random buffer — same shape the real generator produces:
    # (B, 1+K, T_max, d) with K = len(shifts).
    K = len(shifts)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # b_dec init from a single batch's anchor window.
    init_x = torch.randn(BATCH, T_max, DEFAULT_D_IN, device="cuda")
    if hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x)
    del init_x
    torch.cuda.empty_cache()

    print(f"  after init_b_dec peak: {_peak_gb():.2f} GB", flush=True)

    times = []
    for step in range(N_ITERS):
        x = torch.randn(BATCH, 1 + K, T_max, DEFAULT_D_IN, device="cuda")
        t0 = time.time()
        loss, _, z = model(x, alpha=1.0)
        opt.zero_grad()
        loss.backward()
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        print(f"  step {step}: loss={loss.item():.2f} l0={l0:.1f} "
              f"dt={dt*1000:.0f}ms peak={_peak_gb():.2f} GB",
              flush=True)
        del x, loss, z

    # 5090 has 32 GB total. Threshold for "comfortable fit" = 26 GB
    # (leave 6 GB headroom for cache + workspace fragmentation in real run).
    peak = _peak_gb()
    median_step_s = sorted(times)[len(times) // 2]
    print(f"  → peak={peak:.2f} GB | median step={median_step_s*1000:.0f}ms",
          flush=True)
    if peak < 26.0:
        print(f"  VERDICT: fits 5090 ✓ (headroom: {32 - peak:.1f} GB)",
              flush=True)
    elif peak < 32.0:
        print(f"  VERDICT: borderline (only {32 - peak:.1f} GB headroom; "
              f"add cache+workspace and may OOM)",
              flush=True)
    else:
        print(f"  VERDICT: would OOM in real training (need {peak:.1f} GB > 32 GB)",
              flush=True)

    # Free for next variant
    del model, opt
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — this smoke test requires a GPU")
    print(f"Device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)",
          flush=True)
    print(f"d_in={DEFAULT_D_IN}, d_sae={DEFAULT_D_SAE}, batch={BATCH}, "
          f"iters={N_ITERS}", flush=True)

    # Variant A — the headline candidate
    T_max, t_sample = 20, 5
    shifts = tuple(sorted(set(s for s in (1, T_max // 4, T_max // 2)
                              if 1 <= s <= T_max - 1)))
    h = int(DEFAULT_D_SAE * 0.2)
    model_a = SubseqRankedH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T_max=T_max, t_sample=t_sample, k=500,
        shifts=shifts, matryoshka_h_size=h, alpha=1.0,
    )
    smoke_variant(f"SubseqRankedH8 T_max={T_max} t_sample={t_sample}",
                  model_a, T_max, shifts)

    # Variant B — shared encoder, much smaller
    model_b = SubseqSharedH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T_max=T_max, k=500,
        shifts=shifts, matryoshka_h_size=h, alpha=1.0,
    )
    smoke_variant(f"SubseqSharedH8 T_max={T_max}",
                  model_b, T_max, shifts)


if __name__ == "__main__":
    main()
