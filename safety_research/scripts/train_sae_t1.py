"""
Train a vanilla SAE (T=1, stacked_sae with one position) on cached mid_res
activations from Gemma-2-2b-it. Provides the missing 'SAE' arm of the
SAE / T-SAE / TXC comparison.

Reuses settled hyperparams (k=100, d_sae=18432, lr=3e-4) from the andre branch
sweep so we do not re-tune. Trains for STEPS to match T-SAE / TXC checkpoints.
"""
from __future__ import annotations

import os
import sys
import time

import torch

# point at the existing NLP package
NLP_DIR = "/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP"
sys.path.insert(0, NLP_DIR)
os.chdir(NLP_DIR)  # config.py uses relative paths from its location

from config import (  # type: ignore  # noqa: E402
    D_SAE, LEARNING_RATE, ADAM_BETAS, GRAD_CLIP, DEVICE,
    LAYER_SPECS, run_name, CHECKPOINT_DIR, LOG_DIR,
)
from data import CachedActivationSource, WindowIterator  # type: ignore # noqa: E402
from fast_models import FastStackedSAE  # type: ignore # noqa: E402


LAYER = "mid_res"
K = 100
T = 1
STEPS = 5000   # half of TRAIN_STEPS=10000; cached activations stabilise fast
BATCH = 1024


def main() -> None:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    spec = LAYER_SPECS[LAYER]
    d_act = spec["d_act"]
    rn = run_name("stacked_sae", LAYER, K, T)
    print(f"Run: {rn}  d_act={d_act}  d_sae={D_SAE}  k={K}  T={T}  steps={STEPS}")

    source = CachedActivationSource(LAYER)
    iterator = WindowIterator(source, BATCH, T=T)

    model = FastStackedSAE(d_in=d_act, d_sae=D_SAE, T=T, k=K).to(DEVICE)
    compiled = torch.compile(model, mode="reduce-overhead")
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    history = []
    t0 = time.time()
    for step in range(STEPS):
        x = next(iterator)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, x_hat, u = compiled(x)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()

        if step % 250 == 0 or step == STEPS - 1:
            with torch.no_grad():
                fvu = ((x_hat - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
                window_l0 = (u > 0).float().sum(dim=-1).mean().item()
            row = dict(step=step, loss=float(loss.item()),
                       fvu=fvu, window_l0=window_l0)
            history.append(row)
            print(f"  step={step:5d}  loss={row['loss']:.4f}  "
                  f"fvu={row['fvu']:.4f}  L0={row['window_l0']:.0f}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINT_DIR, f"{rn}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"Saved checkpoint: {ckpt}")

    os.makedirs(LOG_DIR, exist_ok=True)
    import json
    log_path = os.path.join(LOG_DIR, f"{rn}.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=1)
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
