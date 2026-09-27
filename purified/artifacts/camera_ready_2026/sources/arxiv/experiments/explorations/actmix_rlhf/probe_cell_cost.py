"""INSTRUMENTATION (writes nothing): per-step wall cost vs T for the § 8 pf arm.

Not an experiment — no leaderboard row, no checkpoint, no manifest. It
exists to replace a guessed grid-cost model with a measured one: the
pilot gives ONE point (T2 @ batch 1024) and the T-scaling is what the
fleet is provisioning against.

Two candidate cost models disagree by ~2.7x over the 21-cell grid:
  (a) compute-dominated: cost ~ tokens/step = batch x T
  (b) feed-dominated:    cost ~ batch only  (the trainer hands the arch
      (B, seq_len=128, d_in) sequences and the arch slices its own
      T-windows, so the H2D batch is T-INDEPENDENT at fixed batch)
Model (b) predicts flat wall across T at batch 1024 and a ~2x/4x drop
at the T10/T16 batch steps; model (a) predicts a linear climb.

Usage (runs on the agent's pinned GPU; keep it short — it co-resides
with live training):
  .venv/bin/python -m experiments.explorations.actmix_rlhf.probe_cell_cost \
      --steps 12 --warmup 3
"""

from __future__ import annotations

import argparse
import time

import torch

from temp_bench.core.config import import_by_path, load_arch, load_datasource
from temp_bench.core.trainer import _build_refill_source, _infer_d_in
from temp_bench.data.sequence_buffer import SequenceBuffer
from experiments.explorations.actmix_rlhf.cells import (
    PF_ARCH, PF_DATASOURCE, _pf_batch)


def probe(T, steps, warmup, refill, d_in, device):
    batch = _pf_batch(T)
    cls = import_by_path(load_arch(PF_ARCH).class_path)
    arch = cls(d_in=d_in, d_sae=18432, T=T, k_pos=100 * T)
    arch.to(device).train()
    optim = torch.optim.Adam(arch.parameters(), lr=1e-4)
    it = SequenceBuffer(refill, seq_len=128, device=device, seed=0)

    feed_t, step_t = [], []
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = it(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optim.zero_grad(set_to_none=True)
        m = arch.train_step(x)
        m["loss"].backward()
        optim.step()
        arch.post_step()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= warmup:
            feed_t.append(t1 - t0)
            step_t.append(t2 - t1)
    mem = torch.cuda.max_memory_allocated(device) / 2**30
    del arch, optim, it
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    n = len(feed_t)
    return {
        "T": T, "batch": batch,
        "feed_s": sum(feed_t) / n, "compute_s": sum(step_t) / n,
        "total_s": (sum(feed_t) + sum(step_t)) / n,
        "gpu_gib": mem,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--Ts", type=int, nargs="+", default=[2, 8, 16])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_spec = load_datasource(PF_DATASOURCE)
    d_in = _infer_d_in(data_spec)
    refill = _build_refill_source(data_spec, seed=0)

    print(f"[probe] device={device} d_in={d_in} steps={args.steps} "
          f"warmup={args.warmup}")
    rows = []
    for T in args.Ts:
        r = probe(T, args.steps, args.warmup, refill, d_in, device)
        rows.append(r)
        print(f"[probe] T={r['T']:>2} batch={r['batch']:>4}  "
              f"feed={r['feed_s']:.3f}s  compute={r['compute_s']:.3f}s  "
              f"total={r['total_s']:.3f}s  peak_gpu={r['gpu_gib']:.1f}GiB  "
              f"=> 25k steps = {r['total_s'] * 25000 / 3600:.2f} h")

    base = rows[0]
    print("\n[probe] scaling vs T=%d (feed-dominated predicts ~flat at "
          "fixed batch):" % base["T"])
    for r in rows:
        print(f"  T={r['T']:>2}: total x{r['total_s'] / base['total_s']:.2f}  "
              f"feed x{r['feed_s'] / base['feed_s']:.2f}  "
              f"compute x{r['compute_s'] / base['compute_s']:.2f}")


if __name__ == "__main__":
    main()
