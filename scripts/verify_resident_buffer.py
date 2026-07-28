"""Equivalence receipt for TEMP_BENCH_BUFFER_RESIDENT=1.

The resident path gathers batches from a device-resident fp16 copy of the
activation cache instead of copying them across the bus every step. It is
only safe to use if it changes *nothing* about the data the model sees.

Two properties make that provable rather than probable:

  * the ``np.random.default_rng(seed)`` index stream is untouched, so both
    paths request the same sequences in the same order;
  * fp16 -> fp32 is an exact widening, so the same elements arrive with
    the same bits.

This script checks the claim empirically: N batches from each path,
compared with ``torch.equal`` (bitwise, not allclose). Writes nothing.

    .venv/bin/python scripts/verify_resident_buffer.py <datasource> [n_batches] [batch]
"""
from __future__ import annotations

import os
import sys
import time

import torch

from temp_bench.core.config import load_datasource


def _build(ds_name: str, seed: int, resident: bool):
    prev = os.environ.get("TEMP_BENCH_BUFFER_RESIDENT")
    os.environ["TEMP_BENCH_BUFFER_RESIDENT"] = "1" if resident else "0"
    try:
        from temp_bench.data.real_lm import build_refill
        return build_refill(load_datasource(ds_name), seed=seed)
    finally:
        if prev is None:
            os.environ.pop("TEMP_BENCH_BUFFER_RESIDENT", None)
        else:
            os.environ["TEMP_BENCH_BUFFER_RESIDENT"] = prev


def main() -> int:
    ds = sys.argv[1] if len(sys.argv) > 1 else "gemma_2_2b_base_l12_phase7"
    n_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    seed = 42

    print(f"datasource={ds} batches={n_batches} batch_size={batch} seed={seed}")

    t0 = time.time()
    host = _build(ds, seed, resident=False)
    print(f"host refill built in {time.time() - t0:.1f}s")

    t0 = time.time()
    res = _build(ds, seed, resident=True)
    print(f"resident refill built in {time.time() - t0:.1f}s "
          f"(one-time device load)")

    ok = True
    h_times: list[float] = []
    r_times: list[float] = []
    for i in range(n_batches):
        t0 = time.time(); bh = host(batch); h_times.append(time.time() - t0)
        t0 = time.time(); br = res(batch)
        if br.device.type == "mps":
            torch.mps.synchronize()
        elif br.device.type == "cuda":
            torch.cuda.synchronize()
        r_times.append(time.time() - t0)

        same_shape = bh.shape == br.shape
        same_dtype = bh.dtype == br.dtype
        identical = same_shape and same_dtype and torch.equal(bh, br.cpu())
        ok &= identical
        print(f"  batch {i}: shape {tuple(bh.shape)} host={bh.dtype}/"
              f"{bh.device.type} resident={br.dtype}/{br.device.type} "
              f"bitwise_identical={identical}")
        if not identical and same_shape:
            d = (bh - br.cpu()).abs()
            print(f"    max|diff|={d.max().item()}  n_diff={(d > 0).sum().item()}")

    hm = sum(h_times) / len(h_times)
    rm = sum(r_times) / len(r_times)
    print(f"\nmean host   refill: {hm * 1000:8.1f} ms")
    print(f"mean resident refill: {rm * 1000:8.1f} ms   "
          f"({hm / rm:.1f}x faster)" if rm > 0 else "")
    print(f"\nVERDICT: {'PASS — batches are bitwise identical' if ok else 'FAIL — DO NOT USE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
