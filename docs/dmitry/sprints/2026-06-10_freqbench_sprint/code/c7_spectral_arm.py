"""c7_spectral_arm.py — train the spectral (multiband) crosscoder on the c7
FineWeb Llama cache, matched to train_llama_txc.py's txc_bare settings.

Reads their cache format: data/llama_3_1_8b/<hook>/L<layer>/activations.fp16.npy
(N, 128, 4096). Trains SpectralTXC (T=5, bands {0},{1,2},{3,4}, k split
proportionally) with their loop hyperparameters (Adam, lr, grad clip 1.0,
per-step decoder renorm, fp32 batches from fp16 GPU buffer, no input norm).

Usage:
  python c7_spectral_arm.py --hook resid --layer 10 --d-sae 32768 \
      --k-pos 20 --T 5 --steps 10000 --batch 1024 --lr 3e-4
Output: data/llama_3_1_8b/<hook>/L<layer>/spectral_txc/{ckpt.pt, log.json}
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from fb_core import SpectralTXC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default="resid")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--d-sae", type=int, default=32768)
    ap.add_argument("--k-pos", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cache-root", default="data/llama_3_1_8b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev = torch.device("cuda")
    root = f"{args.cache_root}/{args.hook}/L{args.layer}"
    out = f"{root}/spectral_txc"
    os.makedirs(out, exist_ok=True)
    A = np.load(f"{root}/activations.fp16.npy", mmap_mode="r")
    N, L, d = A.shape
    buf = torch.tensor(np.asarray(A), device=dev)          # fp16, ~30 GB
    print(f"cache {A.shape} on GPU", flush=True)

    T, k_win = args.T, args.k_pos * args.T
    # 3 DCT bands on T=5: DC {0}, low {1,2}, high {3,4};
    # atoms and budget split proportional to band sizes (1:2:2)
    h1 = args.d_sae // 5
    bands = [[0], [1, 2], [3, 4]]
    h_per = [h1, 2 * h1, args.d_sae - 3 * h1]
    k1 = max(1, k_win // 5)
    k_per = [k1, 2 * k1, k_win - 3 * k1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SpectralTXC(d, T, bands, h_per, k_per).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_wins = L - T + 1
    t0 = time.time()
    log = []
    for step in range(args.steps):
        seq = torch.randint(0, N, (args.batch,), device=dev)
        off = torch.randint(0, n_wins, (args.batch,), device=dev)
        pos = off[:, None] + torch.arange(T, device=dev)[None]
        x = buf[seq[:, None].expand(-1, T), pos].float()
        x_hat, zs = model(x)
        loss = (x_hat - x).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model._normalize_decoder()
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                var = (x - x.mean(0, keepdim=True)).pow(2).sum(-1).mean()
                fvu = (loss / var).item()
            log.append({"step": step, "loss": loss.item(), "fvu": fvu,
                        "sec": time.time() - t0})
            print(f"step {step} fvu={fvu:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    torch.save(model.state_dict(), f"{out}/ckpt.pt")
    json.dump({"args": vars(args), "bands": bands, "h_per": h_per,
               "k_per": k_per, "log": log}, open(f"{out}/log.json", "w"),
              indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
