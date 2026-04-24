"""Re-calibrate BatchTopK inference threshold from forward passes on fineweb.

Issue (A2 of handover): txcdr_t15_batchtopk and txcdr_t20_batchtopk finished
training with negative `threshold` buffers (-1.81 and -49.78), caused by
early-training batches where B·k > count(positive preactivations). The
momentum=0.99 EMA then took too long to recover. At eval time the negative
threshold disables sparsity — every positive preactivation passes, yielding
~dense latents rather than the ≈k/d_sae target.

Fix: for each arch, load the ckpt, run it in `train()` mode (which executes
BatchTopK's per-batch exact top-k + EMA update) over 200 unlabeled fineweb
batches WITHOUT backward. This converges the EMA to the correct batch-level
cutoff. Save a new ckpt with suffix `_recal` so original is preserved.

Usage:
  .venv/bin/python analysis/recalibrate_batchtopk_threshold.py txcdr_t15_batchtopk
  .venv/bin/python analysis/recalibrate_batchtopk_threshold.py txcdr_t20_batchtopk
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/workspace/temp_xc")
sys.path.insert(0, str(REPO))

from experiments.phase5_downstream_utility.probing.run_probing import (  # noqa: E402
    _load_model_for_run,
)

CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
BUF_PATH = REPO / "data/cached_activations/gemma-2-2b-it/fineweb/resid_L13.npy"

N_CALIB_BATCHES = 200
BATCH_SIZE = 256


def make_window_gen(buf: np.ndarray, T: int, seed: int = 0):
    """Yield (B, T, d_in) float32 tensors from random window starts."""
    rng = np.random.default_rng(seed)
    N, L, d = buf.shape
    while True:
        # random seq idx, then random start position with T fitting
        seq_idx = rng.integers(0, N, size=BATCH_SIZE)
        start = rng.integers(0, L - T + 1, size=BATCH_SIZE)
        wins = np.empty((BATCH_SIZE, T, d), dtype=np.float32)
        for i in range(BATCH_SIZE):
            wins[i] = buf[seq_idx[i], start[i]:start[i] + T]
        yield torch.from_numpy(wins)


def recalibrate(arch: str, device: str = "cuda"):
    rid = f"{arch}__seed42"
    ckpt_path = CKPT_DIR / f"{rid}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    print(f"Loading {arch}...")
    # Load original state separately to preserve for resave
    ckpt_raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model, arch_name, meta = _load_model_for_run(rid, ckpt_path, torch.device(device))

    # Print pre-recal threshold(s)
    print("  Pre-recal thresholds:")
    preds = []
    for name, buf in model.named_buffers():
        if name.endswith("threshold") and buf.numel() == 1:
            preds.append((name, float(buf.item())))
            print(f"    {name}: {float(buf.item()):.4f}")
    if not preds:
        print("  No threshold buffers found — not a BatchTopK model?")
        return

    # Inspect input requirements. For TXCDR-style: expects (B, T, d_in).
    T = meta.get("T", 5)
    print(f"  T={T}, loading fineweb buffer (mmap)...")
    buf = np.load(BUF_PATH, mmap_mode="r")
    gen = make_window_gen(buf, T)

    model.train()  # enable BatchTopK's training-path (exact top-k + EMA)
    print(f"  Running {N_CALIB_BATCHES} calibration batches (no backward)...")
    # Use encode() directly — skips reconstruction loss + any contrastive
    # branches, only traverses the BatchTopK module that updates `threshold`.
    with torch.no_grad():
        for i, X in enumerate(gen):
            if i >= N_CALIB_BATCHES:
                break
            X = X.to(device)
            _ = model.encode(X)
            if (i + 1) % 50 == 0:
                th_now = [float(b.item()) for n, b in model.named_buffers()
                          if n.endswith("threshold") and b.numel() == 1]
                print(f"    batch {i+1:3d}/{N_CALIB_BATCHES}  thresholds={[f'{t:.3f}' for t in th_now]}")

    model.eval()
    print("  Post-recal thresholds:")
    posts = []
    for name, buf_t in model.named_buffers():
        if name.endswith("threshold") and buf_t.numel() == 1:
            posts.append((name, float(buf_t.item())))
            print(f"    {name}: {float(buf_t.item()):.4f}")

    # Build new ckpt dict — preserve original meta/arch, update state_dict
    new_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out = {**ckpt_raw, "state_dict": new_state}
    new_path = CKPT_DIR / f"{arch}_recal__seed42.pt"
    torch.save(out, new_path)
    print(f"  Saved {new_path}")
    return {"pre": preds, "post": posts, "new_ckpt": str(new_path)}


def main():
    archs = sys.argv[1:] or ["txcdr_t15_batchtopk", "txcdr_t20_batchtopk"]
    for arch in archs:
        print(f"\n{'='*60}\n{arch}\n{'='*60}")
        recalibrate(arch)


if __name__ == "__main__":
    main()
